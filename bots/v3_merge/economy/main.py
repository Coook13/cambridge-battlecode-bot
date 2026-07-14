import time

from cambc import Controller, EntityType, GameError, Position
from builder import DefenderState, HealerState, run_defender, run_healer
from economy import EconomyState, run_economy
from constants import (ACTION_RADIUS_SQ, CORE_MAX_HP,
                       BREACH_VISION_RADIUS_SQ,
                       COMPETITION_MODE,
                       CORE_VISION_RADIUS_SQ,
                       GUNNER_VISION_RADIUS_SQ,
                       LAUNCHER_VISION_RADIUS_SQ,
                       MAP_DEBUG_CORE_ONLY,
                       MAP_UPDATE_EVERY_N_BUILDER,
                       MAP_UPDATE_EVERY_N_CORE,
                       MAP_UPDATE_EVERY_N_TURRET,
                       MAP_MAX_SIZE, MAP_MIN_SIZE,
                       marker_tile_candidates,
                       RESET_LOGS_ON_START,
                       MSG_DEFENSE_COMPLETE, ROLE_DEFENDER,
                       ROLE_ECONOMY, ROLE_HEALER, SENTINEL_VISION_RADIUS_SQ,
                       SHOW_MAP_DEBUG,
                       decode_marker, encode_marker)
from local_map import UnitLocalMap, VisionLocalMap
from logger import (consume_tick_log_overhead_us, init_logger, log_event,
                    log_local_map,
                    log_timing_event, reset_run_logs)
from logger import reset_tick_log_overhead
from symmetry import detect_symmetry


class Player:
    def __init__(self):
        init_logger()
        self.initialised = False
        self.core_pos = None
        self.symmetry = None
        self.local_map = None

        self.builder_state = None
        self.spawned_builder = False
        self.marker_placed = False
        self.role = None
        # Emergency healer tracking (core only)
        self.healer_spawned = False
        self.healer_marker_placed = False
        # Economy builder tracking (core only)
        self.defense_complete = False
        self.defense_marker_confirmed = False
        self.economy_spawned = False
        self.economy_wait_logged = False
        self.economy_ready_logged = False
        self.economy_spawn_round = None
        self.spawned_bot_roles = {}
        self.spawned_bot_history = []
        self.defender_bot_ids = set()
        self.healer_bot_ids = set()
        self.economy_bot_ids = set()
        self.target_defender_count = 1
        self.target_economy_count = 1
        self.min_rounds_between_role_spawns = 5
        self.last_role_spawn_round = None
        self._timing_meta = {}
        self._map_created_round = None
        self.map_w = None
        self.map_h = None

    def _role_marker_positions(self):
        if self.core_pos is None:
            return ()
        if not isinstance(self.map_w, int) or not isinstance(self.map_h, int):
            return ()

        coords = marker_tile_candidates(
            self.core_pos.x,
            self.core_pos.y,
            self.map_w,
            self.map_h,
        )
        return tuple(Position(x, y) for x, y in coords)

    def _is_turret_entity(self, etype: EntityType) -> bool:
        return etype in (
            EntityType.GUNNER,
            EntityType.SENTINEL,
            EntityType.BREACH,
            EntityType.LAUNCHER,
        )

    def _vision_radius_sq_for_entity(self, etype: EntityType | None) -> int:
        if etype == EntityType.CORE:
            return CORE_VISION_RADIUS_SQ
        if etype == EntityType.GUNNER:
            return GUNNER_VISION_RADIUS_SQ
        if etype == EntityType.SENTINEL:
            return SENTINEL_VISION_RADIUS_SQ
        if etype == EntityType.BREACH:
            return BREACH_VISION_RADIUS_SQ
        if etype == EntityType.LAUNCHER:
            return LAUNCHER_VISION_RADIUS_SQ
        return CORE_VISION_RADIUS_SQ

    def _is_adjacency_ready(self) -> bool:
        if self.local_map is None:
            return False
        if not getattr(self.local_map, "enable_adjacency", False):
            return True
        return bool(getattr(self.local_map, "_adjacency_bootstrapped", True))

    def _spawn_economy_builder(self, c: Controller):
        if self.core_pos is None:
            return None

        cx, cy = self.core_pos.x, self.core_pos.y
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                if dx == 0 and dy == 0:
                    # Keep core centre free for the defender.
                    continue
                sx, sy = cx + dx, cy + dy
                sp = Position(sx, sy)
                try:
                    if c.can_spawn(sp):
                        spawned_id = c.spawn_builder(sp)
                        log_event(
                            c.get_current_round(),
                            c.get_id(),
                            "core",
                            f"({cx},{cy})",
                            "spawned_economy_builder",
                            sx=sx,
                            sy=sy,
                            sid=spawned_id,
                        )
                        return spawned_id
                except GameError:
                    continue
        return None

    def _spawn_builder_at_core_with_fallback(
        self,
        c: Controller,
        primary_event: str,
        fallback_event: str,
    ):
        if self.core_pos is None:
            return None

        cx, cy = self.core_pos.x, self.core_pos.y
        spawn_pos = Position(cx, cy)
        try:
            if c.can_spawn(spawn_pos):
                spawned_id = c.spawn_builder(spawn_pos)
                log_event(
                    c.get_current_round(),
                    c.get_id(),
                    "core",
                    f"({cx},{cy})",
                    primary_event,
                    sx=cx,
                    sy=cy,
                    sid=spawned_id,
                )
                return spawned_id
        except GameError:
            pass

        for nxt in c.get_nearby_tiles():
            try:
                if c.can_spawn(nxt):
                    spawned_id = c.spawn_builder(nxt)
                    log_event(
                        c.get_current_round(),
                        c.get_id(),
                        "core",
                        f"({cx},{cy})",
                        fallback_event,
                        sx=nxt.x,
                        sy=nxt.y,
                        sid=spawned_id,
                    )
                    return spawned_id
            except GameError:
                continue

        return None

    def _register_spawned_builder_role(
        self,
        unit_id: int,
        role: int,
        rnd: int,
    ):
        if not isinstance(unit_id, int):
            return

        self.spawned_bot_roles[unit_id] = role
        self.spawned_bot_history.append((int(rnd), int(unit_id), int(role)))

        if role == ROLE_DEFENDER:
            self.defender_bot_ids.add(unit_id)
            self.spawned_builder = True
        elif role == ROLE_HEALER:
            self.healer_bot_ids.add(unit_id)
            self.healer_spawned = True
        elif role == ROLE_ECONOMY:
            self.economy_bot_ids.add(unit_id)
            self.economy_spawned = True

        self.last_role_spawn_round = int(rnd)

    def _spawn_round_cooldown_ready(self, rnd: int) -> bool:
        if not isinstance(self.last_role_spawn_round, int):
            return True
        return (int(rnd) - int(self.last_role_spawn_round)) >= int(
            self.min_rounds_between_role_spawns
        )

    def _visible_friendly_builder_ids(self, c: Controller):
        out = set()
        my_team = c.get_team()
        try:
            nearby_units = c.get_nearby_units()
        except GameError:
            return out

        for uid in nearby_units:
            try:
                if c.get_team(uid) != my_team:
                    continue
                if c.get_entity_type(uid) != EntityType.BUILDER_BOT:
                    continue
                out.add(int(uid))
            except GameError:
                continue
        return out

    def _refresh_tracked_builder_roles(self, c: Controller):
        visible_builder_ids = self._visible_friendly_builder_ids(c)
        rnd = c.get_current_round()
        cid = c.get_id()

        for role, tracked_set in (
            (ROLE_DEFENDER, self.defender_bot_ids),
            (ROLE_HEALER, self.healer_bot_ids),
        ):
            for uid in tuple(tracked_set):
                if uid in visible_builder_ids:
                    continue
                tracked_set.discard(uid)
                self.spawned_bot_roles.pop(uid, None)
                role_name = (
                    "defender"
                    if role == ROLE_DEFENDER
                    else "healer"
                )
                if self.core_pos is not None:
                    log_event(
                        rnd,
                        cid,
                        "core",
                        f"({self.core_pos.x},{self.core_pos.y})",
                        "core_tracked_bot_missing_v2",
                        lost_id=uid,
                        role=role_name,
                    )

        self.spawned_builder = len(self.defender_bot_ids) > 0
        self.healer_spawned = len(self.healer_bot_ids) > 0

    def _count_friendly_turrets(self, c: Controller) -> int:
        """Count living friendly turrets visible to this unit.

        Turrets (gunner/sentinel/breach/launcher) are counted by
        ``get_unit_count()`` alongside the core and builder bots. Turrets are
        immobile and — for the core — placed on or near the barrier ring,
        so every friendly turret the team owns is within the core's vision
        radius and observable via ``get_nearby_entities``."""
        my_team = c.get_team()
        count = 0
        try:
            nearby_entities = c.get_nearby_entities()
        except GameError:
            return 0
        for entity_id in nearby_entities:
            try:
                if c.get_team(entity_id) != my_team:
                    continue
                if self._is_turret_entity(c.get_entity_type(entity_id)):
                    count += 1
            except GameError:
                continue
        return count

    def _estimate_economy_count(
        self,
        c: Controller,
        defender_count: int,
        healer_count: int,
    ) -> int:
        try:
            friendly_units_including_core = int(c.get_unit_count())
        except GameError:
            return 0

        # get_unit_count() includes the core + all builder bots + all
        # turrets. The caller already tracks defender/healer builder counts;
        # subtract the turret count too so whatever remains is the set of
        # economy builder bots.
        turret_count = self._count_friendly_turrets(c)
        economy_count = friendly_units_including_core - (
            1 + int(defender_count) + int(healer_count) + int(turret_count)
        )
        return max(0, int(economy_count))

    def _spawn_builder_for_role(self, c: Controller, role: int) -> bool:
        if self.core_pos is None:
            return False

        if role == ROLE_ECONOMY:
            marker_event = "placed_economy_marker"
        elif role == ROLE_DEFENDER:
            marker_event = "placed_defender_marker"
        elif role == ROLE_HEALER:
            marker_event = "placed_healer_marker"
        else:
            return False

        if not self._place_role_marker(c, role, marker_event):
            return False

        if role == ROLE_ECONOMY:
            spawned_id = self._spawn_economy_builder(c)
        elif role == ROLE_DEFENDER:
            spawned_id = self._spawn_builder_at_core_with_fallback(
                c,
                "spawned_defender",
                "spawned_defender_fallback",
            )
        else:
            spawned_id = self._spawn_builder_at_core_with_fallback(
                c,
                "spawned_healer",
                "spawned_healer_fallback",
            )

        if not isinstance(spawned_id, int):
            return False

        rnd = c.get_current_round()
        self._register_spawned_builder_role(spawned_id, role, rnd)
        if role == ROLE_ECONOMY:
            self.economy_spawn_round = rnd
        return True

    def _place_role_marker(self, c: Controller, role: int, event_name: str) -> bool:
        """Place/update role marker on the first available candidate tile."""
        marker_positions = self._role_marker_positions()
        if not marker_positions or self.core_pos is None:
            return False

        marker_val = encode_marker(role, self.core_pos.x, self.core_pos.y)

        for mp in marker_positions:
            try:
                bid = c.get_tile_building_id(mp)
            except GameError:
                continue

            if bid is None or bid == 0:
                continue

            try:
                if c.get_entity_type(bid) != EntityType.MARKER:
                    continue
                if c.get_team(bid) != c.get_team():
                    continue
                existing_val = c.get_marker_value(bid)
                if existing_val == marker_val:
                    return True
            except GameError:
                continue

        for mp in marker_positions:
            try:
                bid = c.get_tile_building_id(mp)
            except GameError:
                continue

            if bid is not None and bid != 0:
                try:
                    if c.get_entity_type(bid) != EntityType.MARKER:
                        continue
                    if c.get_team(bid) != c.get_team():
                        continue
                    if not c.can_destroy(mp):
                        continue
                    c.destroy(mp)
                except GameError:
                    continue

            try:
                if c.can_place_marker(mp):
                    c.place_marker(mp, marker_val)
                    log_event(
                        c.get_current_round(), c.get_id(), "core",
                        f"({self.core_pos.x},{self.core_pos.y})", event_name,
                        mx=mp.x, my=mp.y, val=marker_val,
                    )
                    return True
            except GameError:
                continue

        return False

    def _place_temporary_defender_marker(self, c: Controller) -> bool:
        """Place initial DEFENDER marker on a temporary ring tile."""
        if self.core_pos is None:
            return False
        cx, cy = self.core_pos.x, self.core_pos.y
        marker_val = encode_marker(ROLE_DEFENDER, cx, cy)
        for mp in self._role_marker_positions():
            try:
                if c.can_place_marker(mp):
                    c.place_marker(mp, marker_val)
                    log_event(
                        c.get_current_round(), c.get_id(), "core",
                        f"({cx},{cy})", "placed_marker",
                        mx=mp.x, my=mp.y, val=marker_val,
                    )
                    return True
            except GameError:
                continue
        return False

    def _has_defense_complete_marker(self, c: Controller) -> bool:
        marker_positions = self._role_marker_positions()
        if not marker_positions:
            return False

        for mp in marker_positions:
            try:
                if not c.is_in_vision(mp):
                    continue
                bid = c.get_tile_building_id(mp)
                if bid is None or bid == 0:
                    continue
                if c.get_entity_type(bid) != EntityType.MARKER:
                    continue
                if c.get_team(bid) != c.get_team():
                    continue

                marker_val = c.get_marker_value(bid)
                # Backward compatible with old one-byte completion marker.
                if marker_val == MSG_DEFENSE_COMPLETE:
                    return True

                role, mcx, mcy = decode_marker(marker_val)
                if self.core_pos is None:
                    continue
                if (
                    role == MSG_DEFENSE_COMPLETE
                    and mcx == self.core_pos.x
                    and mcy == self.core_pos.y
                ):
                    return True
            except GameError:
                continue

        return False

    def run(self, c: Controller):
        if not COMPETITION_MODE:
            reset_tick_log_overhead()
            start_ns = time.perf_counter_ns()
        else:
            start_ns = 0
        et = "unknown"
        self._timing_meta = {}
        try:
            etype = c.get_entity_type()
            et = str(etype.name).lower()

            if etype == EntityType.CORE:
                self._run_core(c)
            elif etype == EntityType.BUILDER_BOT:
                self._run_bot(c)
            elif etype == EntityType.GUNNER:
                self._run_gunner(c)
            elif etype == EntityType.LAUNCHER:
                self._run_launcher(c)
        except Exception as e:
            try:
                p = c.get_position()
                pos = f"({p.x},{p.y})"
                rnd = c.get_current_round()
                uid = c.get_id()
                et = str(c.get_entity_type().name).lower()
            except Exception:
                pos = "(?,?)"
                rnd = -1
                uid = -1
                et = "unknown"
            log_event(rnd, uid, et, pos, "fatal_exception",
                      force=True, err=repr(e))

        if COMPETITION_MODE:
            return

        try:
            p = c.get_position()
            pos = f"({p.x},{p.y})"
            rnd = c.get_current_round()
            uid = c.get_id()
        except Exception:
            pos = "(?,?)"
            rnd = -1
            uid = -1

        elapsed_us = (time.perf_counter_ns() - start_ns) // 1000
        log_overhead_us = consume_tick_log_overhead_us()
        startup_reset_logs_us = int(
            self._timing_meta.get("startup_reset_logs_us", 0)
        )
        net_elapsed_us = (
            elapsed_us
            - log_overhead_us
            - startup_reset_logs_us
        )
        if net_elapsed_us < 0:
            net_elapsed_us = 0

        self._timing_meta["tick_raw_us"] = elapsed_us
        self._timing_meta["log_overhead_us"] = log_overhead_us
        log_timing_event(rnd, uid, et, pos, net_elapsed_us,
                         **self._timing_meta)

    def _init_game(self, c: Controller):
        my_team = c.get_team()
        try:
            my_type = c.get_entity_type()
        except GameError:
            my_type = None
        log_type = str(my_type.name).lower(
        ) if my_type is not None else "unknown"

        # Fast path for core unit: its own position is the core center.
        try:
            if c.get_entity_type() == EntityType.CORE:
                self.core_pos = c.get_position()
        except GameError:
            self.core_pos = None

        # For non-core units, find nearby friendly core.
        if self.core_pos is None:
            for bid in c.get_nearby_buildings():
                try:
                    if c.get_entity_type(bid) == EntityType.CORE and c.get_team(bid) == my_team:
                        self.core_pos = c.get_position(bid)
                        break
                except GameError:
                    continue

        if not self.core_pos:
            log_event(c.get_current_round(), c.get_id(),
                      log_type, "(?,?)", "init_no_core_visible")
            return

        map_w = c.get_map_width()
        map_h = c.get_map_height()
        self.map_w = map_w
        self.map_h = map_h

        size_ok = (map_w == map_h) and (MAP_MIN_SIZE <= map_w <= MAP_MAX_SIZE)
        if not size_ok:
            log_event(
                c.get_current_round(),
                c.get_id(),
                log_type,
                f"({self.core_pos.x},{self.core_pos.y})",
                "map_assumption_violation",
                w=map_w,
                h=map_h,
                expected=f"square,{MAP_MIN_SIZE}-{MAP_MAX_SIZE}",
                origin="top_left_0_0",
            )

        full_map_unit = my_type == EntityType.BUILDER_BOT

        if full_map_unit:
            self.symmetry = detect_symmetry(self.core_pos, map_w, map_h)
            self.local_map = UnitLocalMap(
                map_w,
                map_h,
                self.symmetry,
                my_team,
                infer_symmetry=True,
                enable_adjacency=True,
                vision_only=False,
            )
            map_mode = "full"
        else:
            self.symmetry = "NONE"
            try:
                center_pos = c.get_position()
            except GameError:
                center_pos = self.core_pos
            self.local_map = VisionLocalMap(
                map_w,
                map_h,
                my_team,
                center_pos,
                self._vision_radius_sq_for_entity(my_type),
            )
            map_mode = "vision_window"

        if self.local_map is not None:
            self.local_map.set_friendly_core(self.core_pos)
            self._map_created_round = c.get_current_round()
        else:
            self._map_created_round = None

        self.initialised = True
        log_event(
            c.get_current_round(),
            c.get_id(),
            log_type,
            f"({self.core_pos.x},{self.core_pos.y})",
            "init_done",
            symmetry=self.symmetry,
            map_mode=map_mode,
            w=map_w,
            h=map_h,
            origin="top_left_0_0",
            map_debug=SHOW_MAP_DEBUG,
        )

    def _update_local_map(
        self,
        c: Controller,
        entity_type: EntityType,
        stride: int,
    ):
        if self.local_map is None:
            return
        rnd = c.get_current_round()
        if self._map_created_round is not None and rnd <= self._map_created_round:
            if not COMPETITION_MODE:
                self._timing_meta["map_update_deferred_init"] = 1
            return
        if stride > 1 and (rnd % stride) != 0:
            if not COMPETITION_MODE:
                self._timing_meta["map_update_skipped"] = 1
            return

        if COMPETITION_MODE:
            self.local_map.update_from_controller(c)
        else:
            map_update_start_ns = time.perf_counter_ns()
            self.local_map.update_from_controller(c)
            map_update_us = (
                time.perf_counter_ns() - map_update_start_ns
            ) // 1000
            self._timing_meta["map_update_us"] = map_update_us
            self._timing_meta.update(self.local_map.consume_round_metrics())
        if SHOW_MAP_DEBUG and not COMPETITION_MODE:
            try:
                if MAP_DEBUG_CORE_ONLY and entity_type != EntityType.CORE:
                    return
                p = c.get_position()
                log_local_map(
                    rnd,
                    c.get_id(),
                    str(entity_type.name).lower(),
                    f"({p.x},{p.y})",
                    self.local_map,
                    p,
                )
            except GameError:
                pass

    def _run_core(self, c: Controller):
        if not self.initialised:
            # Clear debug/timing logs once at game start.
            if RESET_LOGS_ON_START and not COMPETITION_MODE:
                reset_start_ns = time.perf_counter_ns()
                reset_run_logs()
                self._timing_meta["startup_reset_logs_us"] = (
                    time.perf_counter_ns() - reset_start_ns
                ) // 1000
            self._init_game(c)

        if not self.initialised:
            return

        self._update_local_map(
            c,
            EntityType.CORE,
            MAP_UPDATE_EVERY_N_CORE,
        )

        core_pos = c.get_position()
        cx, cy = self.core_pos.x, self.core_pos.y
        rnd = c.get_current_round()

        self._refresh_tracked_builder_roles(c)

        defender_count = len(self.defender_bot_ids)
        healer_count = len(self.healer_bot_ids)
        economy_count = self._estimate_economy_count(
            c,
            defender_count,
            healer_count,
        )
        self.economy_spawned = economy_count > 0

        desired_healer_count = 0
        core_max = CORE_MAX_HP
        try:
            if core_max > 0 and c.get_hp() < int(0.8 * core_max):
                desired_healer_count = 1
        except GameError:
            pass

        log_event(rnd, c.get_id(), "core",
                  f"({core_pos.x},{core_pos.y})", "core_tick",
                  defense=self.defense_complete,
                  econ=economy_count,
                  defenders=defender_count,
                  healers=healer_count,
                  target_defenders=self.target_defender_count,
                  target_econ=self.target_economy_count,
                  target_healers=desired_healer_count,
                  cd=c.get_action_cooldown())

        # Temporary diagnostic: break down what get_unit_count() is counting.
        if not COMPETITION_MODE:
            try:
                raw_unit_count = int(c.get_unit_count())
            except GameError:
                raw_unit_count = -1
            my_team = c.get_team()
            n_core = 0
            n_builder = 0
            n_gunner = 0
            n_sentinel = 0
            n_breach = 0
            n_launcher = 0
            n_other = 0
            n_enemy = 0
            try:
                for eid in c.get_nearby_entities():
                    try:
                        if c.get_team(eid) != my_team:
                            n_enemy += 1
                            continue
                        et = c.get_entity_type(eid)
                        if et == EntityType.CORE:
                            n_core += 1
                        elif et == EntityType.BUILDER_BOT:
                            n_builder += 1
                        elif et == EntityType.GUNNER:
                            n_gunner += 1
                        elif et == EntityType.SENTINEL:
                            n_sentinel += 1
                        elif et == EntityType.BREACH:
                            n_breach += 1
                        elif et == EntityType.LAUNCHER:
                            n_launcher += 1
                        else:
                            n_other += 1
                    except GameError:
                        continue
            except GameError:
                pass
            turret_count_vis = n_gunner + n_sentinel + n_breach + n_launcher
            friendly_visible = (
                n_core + n_builder + turret_count_vis + n_other
            )
            log_event(
                rnd, c.get_id(), "core",
                f"({core_pos.x},{core_pos.y})",
                "core_unit_count_debug",
                unit_count=raw_unit_count,
                friendly_visible=friendly_visible,
                core=n_core,
                builder=n_builder,
                gunner=n_gunner,
                sentinel=n_sentinel,
                breach=n_breach,
                launcher=n_launcher,
                turrets=turret_count_vis,
                other_friendly=n_other,
                enemy_visible=n_enemy,
                def_tracked=defender_count,
                heal_tracked=healer_count,
                econ_derived=economy_count,
            )

        if c.get_action_cooldown() == 0 and self._spawn_round_cooldown_ready(rnd):
            if economy_count < self.target_economy_count:
                if self._spawn_builder_for_role(c, ROLE_ECONOMY):
                    return

            if defender_count < self.target_defender_count:
                if self._spawn_builder_for_role(c, ROLE_DEFENDER):
                    return

            if healer_count < desired_healer_count:
                if self._spawn_builder_for_role(c, ROLE_HEALER):
                    return

        # Step 4: Defense is complete only when launcher exists AND defender
        # has communicated completion through the marker channel.
        if self.spawned_builder and not self.defense_complete:
            launcher_placed = False
            try:
                for bid in c.get_nearby_buildings():
                    try:
                        if c.get_entity_type(bid) != EntityType.LAUNCHER:
                            continue
                        if c.get_team(bid) != c.get_team():
                            continue
                        lp = c.get_position(bid)
                        if max(abs(lp.x - cx), abs(lp.y - cy)) <= 2:
                            launcher_placed = True
                            break
                    except GameError:
                        continue
            except GameError:
                pass

            marker_confirmed = self._has_defense_complete_marker(c)
            if launcher_placed and marker_confirmed:
                self.defense_complete = True
                self.defense_marker_confirmed = True
                log_event(
                    c.get_current_round(), c.get_id(),
                    "core", f"({cx},{cy})",
                    "defense_complete_confirmed",
                    launcher=1,
                    marker=1,
                )

    def _run_bot(self, c: Controller):
        if not self.initialised:
            self._init_game(c)

        if not self.initialised:
            return

        self._update_local_map(c, EntityType.BUILDER_BOT,
                               MAP_UPDATE_EVERY_N_BUILDER)

        # First tick: read marker to determine role.
        if self.builder_state is None and self.role is None:
            p = c.get_position()
            my_team = c.get_team()
            assigned_role = ROLE_DEFENDER  # default

            for mp in self._role_marker_positions():
                try:
                    if not c.is_in_vision(mp):
                        continue
                    bid = c.get_tile_building_id(mp)
                    if bid is None or bid == 0:
                        continue
                    if c.get_entity_type(bid) != EntityType.MARKER:
                        continue
                    if c.get_team(bid) != my_team:
                        continue

                    marker_val = c.get_marker_value(bid)
                    role, mcx, mcy = decode_marker(marker_val)
                    if self.core_pos is not None:
                        if mcx != self.core_pos.x or mcy != self.core_pos.y:
                            continue

                    if role == ROLE_DEFENDER:
                        assigned_role = ROLE_DEFENDER
                        log_event(
                            c.get_current_round(), c.get_id(), "builder",
                            f"({p.x},{p.y})", "read_marker_defender",
                            val=marker_val, mcx=mcx, mcy=mcy,
                        )
                        break
                    if role == ROLE_HEALER:
                        assigned_role = ROLE_HEALER
                        log_event(
                            c.get_current_round(), c.get_id(), "builder",
                            f"({p.x},{p.y})", "read_marker_healer",
                            val=marker_val, mcx=mcx, mcy=mcy,
                        )
                        break
                    if role == ROLE_ECONOMY:
                        assigned_role = ROLE_ECONOMY
                        log_event(
                            c.get_current_round(), c.get_id(), "builder",
                            f"({p.x},{p.y})", "read_marker_economy",
                            val=marker_val, mcx=mcx, mcy=mcy,
                        )
                        break
                except GameError:
                    continue

            self.role = assigned_role

            if self.role == ROLE_DEFENDER:
                self.builder_state = DefenderState(
                    self.core_pos,
                    c.get_map_width(),
                    c.get_map_height(),
                    self.local_map,
                )
                log_event(c.get_current_round(), c.get_id(),
                          "defender", f"({p.x},{p.y})", "defender_init")
            elif self.role == ROLE_HEALER:
                self.builder_state = HealerState(
                    self.core_pos,
                    self.local_map,
                )
                log_event(c.get_current_round(), c.get_id(),
                          "healer", f"({p.x},{p.y})", "healer_init")
            elif self.role == ROLE_ECONOMY:
                self.builder_state = EconomyState(self.core_pos)
                log_event(c.get_current_round(), c.get_id(),
                          "economy", f"({p.x},{p.y})", "economy_init")
            else:
                log_event(c.get_current_round(), c.get_id(),
                          "builder", f"({p.x},{p.y})", "economy_role_disabled")
                try:
                    c.self_destruct()
                except GameError:
                    pass
                return

        if self.role == ROLE_DEFENDER:
            if isinstance(self.builder_state, DefenderState):
                run_defender(c, self.builder_state)
            return
        elif self.role == ROLE_HEALER:
            if isinstance(self.builder_state, HealerState):
                run_healer(c, self.builder_state)
            return
        elif self.role == ROLE_ECONOMY:
            # Economy behavior is intentionally deferred until adjacency is ready.
            if not self._is_adjacency_ready():
                if not self.economy_wait_logged:
                    p = c.get_position()
                    log_event(
                        c.get_current_round(),
                        c.get_id(),
                        "economy",
                        f"({p.x},{p.y})",
                        "economy_wait_adjacency",
                    )
                    self.economy_wait_logged = True
                return
            if not self.economy_ready_logged:
                p = c.get_position()
                log_event(
                    c.get_current_round(),
                    c.get_id(),
                    "economy",
                    f"({p.x},{p.y})",
                    "economy_adjacency_ready",
                )
                self.economy_ready_logged = True
            if not isinstance(self.builder_state, EconomyState):
                self.builder_state = EconomyState(self.core_pos)
            run_economy(c, self.builder_state, self.local_map)
            return
        else:
            # Economy builder role removed.
            return

    def _run_gunner(self, c: Controller):
        if not self.initialised:
            self._init_game(c)
        if not self.initialised:
            return

        self._update_local_map(
            c,
            EntityType.GUNNER,
            MAP_UPDATE_EVERY_N_TURRET,
        )

        pos = c.get_position()
        rnd = c.get_current_round()
        uid = c.get_id()

        enemy_core_pos = None
        for bid in c.get_nearby_buildings():
            try:
                if c.get_entity_type(bid) != EntityType.CORE:
                    continue
                if c.get_team(bid) == c.get_team():
                    continue
                enemy_core_pos = c.get_position(bid)
                break
            except GameError:
                continue

        if enemy_core_pos is None:
            return

        try:
            if c.can_fire(enemy_core_pos):
                c.fire(enemy_core_pos)
                log_event(
                    rnd,
                    uid,
                    "gunner",
                    f"({pos.x},{pos.y})",
                    "fired_enemy_core",
                    tx=enemy_core_pos.x,
                    ty=enemy_core_pos.y,
                )
        except GameError:
            pass

    def _run_launcher(self, c: Controller):
        """Auto-launch adjacent builder bots to the landing pad."""
        if not self.initialised:
            self._init_game(c)
        if not self.initialised:
            return

        self._update_local_map(
            c,
            EntityType.LAUNCHER,
            MAP_UPDATE_EVERY_N_TURRET,
        )

        pos = c.get_position()
        rnd = c.get_current_round()
        uid = c.get_id()

        if c.get_action_cooldown() > 0:
            return

        cx, cy = self.core_pos.x, self.core_pos.y

        # Designated launcher queue tile: core tile adjacent to launcher,
        # chosen closest to core center.
        wait_candidates = []
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                wx, wy = cx + dx, cy + dy
                ddx = wx - pos.x
                ddy = wy - pos.y
                if ddx * ddx + ddy * ddy <= ACTION_RADIUS_SQ:
                    wait_candidates.append((wx, wy))
        if wait_candidates:
            wait_x, wait_y = min(
                wait_candidates,
                key=lambda t: abs(t[0] - cx) + abs(t[1] - cy),
            )
        else:
            wait_x, wait_y = cx + 1, cy

        # Launch only friendly builder bot standing on designated wait tile.
        my_team = c.get_team()
        for nid in c.get_nearby_units():
            try:
                if c.get_entity_type(nid) != EntityType.BUILDER_BOT:
                    continue
                if c.get_team(nid) != my_team:
                    continue
                bot_pos = c.get_position(nid)
                if bot_pos.x != wait_x or bot_pos.y != wait_y:
                    continue
                # Dynamically scan for passable landing tiles outside
                # the barrier (Chebyshev ≥ 4 from core), within throw r²=26.
                best_target = None
                best_dist = 999
                launcher_scan_radius = int(LAUNCHER_VISION_RADIUS_SQ ** 0.5)
                for dy in range(-launcher_scan_radius, launcher_scan_radius + 1):
                    for dx in range(-launcher_scan_radius, launcher_scan_radius + 1):
                        lx, ly = pos.x + dx, pos.y + dy
                        # Must be within throw range from launcher
                        if dx * dx + dy * dy > LAUNCHER_VISION_RADIUS_SQ:
                            continue
                        # Must be outside barrier (Chebyshev ≥ 4 from core)
                        if max(abs(lx - cx), abs(ly - cy)) < 4:
                            continue
                        lp = Position(lx, ly)
                        try:
                            if c.can_launch(bot_pos, lp):
                                d = abs(lx - cx) + abs(ly - cy)
                                if d < best_dist:
                                    best_dist = d
                                    best_target = lp
                        except GameError:
                            continue
                if best_target is not None:
                    c.launch(bot_pos, best_target)
                    log_event(
                        rnd, uid, "launcher",
                        f"({pos.x},{pos.y})",
                        "launched_bot",
                        bx=bot_pos.x, by=bot_pos.y,
                        tx=best_target.x, ty=best_target.y)
                    return
            except GameError:
                continue
