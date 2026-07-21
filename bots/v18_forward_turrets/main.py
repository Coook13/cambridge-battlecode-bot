import time

from cambc import Controller, Direction, EntityType, GameError, Position
from builder import DefenderState, HealerState, run_defender, run_healer
from economy import EconomyState, run_economy
from miner import MinerState, run_miner
from raider import RaiderState, run_raider
from repair_patrol import RepairPatrolState, run_repair_patrol
from constants import (ACTION_RADIUS_SQ, CORE_MAX_HP,
                       BREACH_VISION_RADIUS_SQ,
                       BUILDER_BOT_BASE_COST,
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
                       ROLE_ECONOMY, ROLE_HEALER,
                       ROLE_MINER, ROLE_RAIDER, ROLE_REPAIR_PATROL,
                       SENTINEL_VISION_RADIUS_SQ,
                       SHOW_MAP_DEBUG,
                       VALID_ROLES,
                       decode_marker, encode_marker)
from local_map import UnitLocalMap, VisionLocalMap
from logger import (consume_tick_log_overhead_us, init_logger, log_event,
                    log_local_map,
                    log_timing_event, reset_run_logs)
from logger import reset_tick_log_overhead
from symmetry import detect_symmetry


def _max_raiders_for_round(rnd: int) -> int:
    """Hard cap on simultaneous raiders by round bracket.

    v17_no1_combo: copy the #1 replay's larger raider waves while keeping
    v14's earlier economy ladder to pay for them.
    """
    if rnd < 200:
        return 2
    if rnd < 400:
        return 4
    if rnd < 800:
        return 6
    return 8


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
        # Combat role tracking (grafted in v3_merged flip).
        self.miner_bot_ids = set()
        self.raider_bot_ids = set()
        self.repair_patrol_bot_ids = set()
        # v4_micky: skip defender entirely.
        #
        # Why this works on default maps:
        #   - The defender chain (build barriers → harvester → launcher →
        #     return → place marker) costs hundreds of Ti and pumps roads/
        #     conveyors aggressively. Each road/conveyor adds to the team
        #     cost-scale (+0.5%/+1% per build).
        #   - On default maps the cost-scale spirals so fast that builder
        #     bots end up costing 200+ Ti before the launcher is finished,
        #     so the launcher never actually gets built. Result:
        #     defense_complete stays False forever, raider/miner spawn
        #     gates never trip, and matches always tie at round 2000 on
        #     identical Ti delivered (faisal's pipeline is byte-identical
        #     in both bots, so v4_micky === v3_merged in every match).
        #   - Skipping the defender frees the entire Ti pool for faisal's
        #     economy. Pool stays high enough that the dormant axionite
        #     refinery pipeline (39 phases starting at economy.py:383,
        #     gated on titanium_pool > 2000) actually wakes up. v4 now
        #     mines 5–40k Ti per match AND delivers refined axionite —
        #     which auto-wins the tiebreak (refined Ax > Ti delivered).
        #
        # Risk:
        #   - No gunner ring = enemy raiders walk in unopposed. On default
        #     maps neither side spawns raiders (same cost-scale problem),
        #     so this is fine. Against aggressive opponents on the ladder
        #     this could regress; AB-tested locally vs starter, version4,
        #     version5 — all wins.
        #
        # AB results (single 1-line change vs v3_merged baseline, 7 maps
        # × 2 seats = 14 matches): 12 wins / 2 losses. Both losses are
        # tight or in the seat that already lost in baseline self-match.
        # Net pair tournament: 5 pair-wins + 2 pair-ties + 0 pair-losses.
        self.target_defender_count = 0
        self.target_economy_count = 1
        # Combat role targets. Miner waves gated by round + Ti floor; raider
        # count capped by round bracket; repair_patrol is a one-shot spawn
        # once economy is running and round >= 80.
        # v5_swarm: SUPPORT-HEAVY strategy. Differentiator vs v4_micky's
        # raider-only path: spawn 2 repair patrols instead of 1 (defense
        # priority), plus a 3rd economy bot earlier (multi-econ priority).
        # Miner waves stay at default cadence — aggressive miner spawning
        # was tested and regressed test1 (Ti budget starvation).
        self.target_miner_count_early = 2      # pre-defense miners
        self.miner_wave_interval = 60          # rounds between post-defense miner waves
        self.target_repair_patrol_count = 2    # v5: 2 repair patrols (was 1)
        self.target_raider_count_max = 8       # hard cap; actual cap by round bracket
        self.repair_patrol_spawned = False
        self._last_miner_wave_round = 0
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

    def _try_builder_forward_turret(self, c: Controller) -> bool:
        """Let any advanced builder leave a forward gunner/sentinel.

        Raider-only turret placement did not fire often enough because other
        roles can be the builders that actually reach enemy infrastructure.
        Keep this gated by round, distance from home, unit cap, local turret
        density, and a verified visible target.
        """
        if self.core_pos is None:
            return False
        if self.role in (ROLE_DEFENDER, ROLE_HEALER):
            return False
        try:
            rnd = c.get_current_round()
            if rnd < 220 or c.get_action_cooldown() > 0:
                return False
            pos = c.get_position()
            if (pos.x - self.core_pos.x) ** 2 + (pos.y - self.core_pos.y) ** 2 < 49:
                return False
            if c.get_unit_count() >= 48:
                return False
            if self._nearby_friendly_pressure_turrets(c, pos) >= 3:
                return False
            ti, _ = c.get_global_resources()
            sentinel_cost, _ = c.get_sentinel_cost()
            gunner_cost, _ = c.get_gunner_cost()
        except GameError:
            return False

        targets = self._visible_forward_turret_targets(c)
        if not targets:
            return False

        candidates = []
        for dx, dy in (
            (0, -1), (1, -1), (1, 0), (1, 1),
            (0, 1), (-1, 1), (-1, 0), (-1, -1),
        ):
            bpos = Position(pos.x + dx, pos.y + dy)
            try:
                bid = c.get_tile_building_id(bpos)
                if bid is not None and bid != 0:
                    continue
            except GameError:
                continue
            dist = min(abs(bpos.x - tp.x) + abs(bpos.y - tp.y) for _prio, tp in targets)
            candidates.append((dist, bpos))

        candidates.sort(key=lambda item: (item[0], item[1].x, item[1].y))

        for _dist, bpos in candidates:
            for priority, target in targets:
                for direction in self._direction_options_8((bpos.x, bpos.y), (target.x, target.y)):
                    if ti >= sentinel_cost + 220 and c.get_unit_count() <= 44:
                        try:
                            if (
                                c.can_build_sentinel(bpos, direction)
                                and c.can_fire_from(bpos, direction, EntityType.SENTINEL, target)
                            ):
                                c.build_sentinel(bpos, direction)
                                return True
                        except GameError:
                            pass
                    if priority <= 3 and ti >= gunner_cost + 90:
                        try:
                            if (
                                c.can_build_gunner(bpos, direction)
                                and c.can_fire_from(bpos, direction, EntityType.GUNNER, target)
                            ):
                                c.build_gunner(bpos, direction)
                                return True
                        except GameError:
                            pass
        return False

    def _visible_forward_turret_targets(self, c: Controller):
        my_team = c.get_team()
        targets = []
        try:
            nearby = c.get_nearby_buildings()
        except GameError:
            return targets
        for bid in nearby:
            try:
                if c.get_team(bid) == my_team:
                    continue
                etype = c.get_entity_type(bid)
                pos = c.get_position(bid)
            except GameError:
                continue
            if etype == EntityType.CORE:
                priority = 0
            elif etype == EntityType.HARVESTER:
                priority = 1
            elif etype in (EntityType.CONVEYOR, EntityType.SPLITTER, EntityType.ARMOURED_CONVEYOR, EntityType.BRIDGE):
                priority = 2
            elif etype in (EntityType.GUNNER, EntityType.SENTINEL, EntityType.BREACH, EntityType.LAUNCHER, EntityType.FOUNDRY):
                priority = 3
            elif etype in (EntityType.ROAD, EntityType.BARRIER):
                priority = 4
            else:
                continue
            targets.append((priority, pos))
        targets.sort(key=lambda item: (item[0], item[1].x, item[1].y))
        return targets[:8]

    def _nearby_friendly_pressure_turrets(self, c: Controller, pos: Position) -> int:
        my_team = c.get_team()
        count = 0
        try:
            nearby = c.get_nearby_buildings()
        except GameError:
            return 0
        for bid in nearby:
            try:
                if c.get_team(bid) != my_team:
                    continue
                if c.get_entity_type(bid) not in (EntityType.GUNNER, EntityType.SENTINEL):
                    continue
                tp = c.get_position(bid)
                if (tp.x - pos.x) ** 2 + (tp.y - pos.y) ** 2 <= 20:
                    count += 1
            except GameError:
                continue
        return count

    def _direction_options_8(self, from_xy, to_xy):
        dx = to_xy[0] - from_xy[0]
        dy = to_xy[1] - from_xy[1]
        mapping = (
            ((0, -1), Direction.NORTH),
            ((1, -1), Direction.NORTHEAST),
            ((1, 0), Direction.EAST),
            ((1, 1), Direction.SOUTHEAST),
            ((0, 1), Direction.SOUTH),
            ((-1, 1), Direction.SOUTHWEST),
            ((-1, 0), Direction.WEST),
            ((-1, -1), Direction.NORTHWEST),
        )
        ranked = []
        for delta, direction in mapping:
            dot = dx * delta[0] + dy * delta[1]
            ranked.append((-dot, direction.name, direction))
        ranked.sort()
        return tuple(direction for _neg_dot, _name, direction in ranked)

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
        elif role == ROLE_MINER:
            self.miner_bot_ids.add(unit_id)
            self._last_miner_wave_round = int(rnd)
        elif role == ROLE_RAIDER:
            self.raider_bot_ids.add(unit_id)
        elif role == ROLE_REPAIR_PATROL:
            self.repair_patrol_bot_ids.add(unit_id)
            self.repair_patrol_spawned = True

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

    def _count_visible_friendly_harvesters(self, c: Controller) -> int:
        """Count friendly harvesters within core vision.

        Used by gating logic that previously read
        self.builder_state.harvester_ids_friendly — that attribute lives on
        EconomyState, not on the core's self.builder_state (which is None).
        Without this helper, the existing combat_econ_ready and
        repair_patrol harvester gates were always 0 from core perspective.
        """
        my_team = c.get_team()
        count = 0
        try:
            for bid in c.get_nearby_buildings():
                try:
                    if c.get_entity_type(bid) != EntityType.HARVESTER:
                        continue
                    if c.get_team(bid) != my_team:
                        continue
                    count += 1
                except GameError:
                    continue
        except GameError:
            return 0
        return count

    def _refresh_tracked_builder_roles(self, c: Controller):
        visible_builder_ids = self._visible_friendly_builder_ids(c)
        rnd = c.get_current_round()
        cid = c.get_id()

        _ROLE_NAME_MAP = {
            ROLE_DEFENDER: "defender",
            ROLE_HEALER: "healer",
            ROLE_MINER: "miner",
            ROLE_RAIDER: "raider",
            ROLE_REPAIR_PATROL: "repair_patrol",
        }
        for role, tracked_set in (
            (ROLE_DEFENDER, self.defender_bot_ids),
            (ROLE_HEALER, self.healer_bot_ids),
            (ROLE_MINER, self.miner_bot_ids),
            (ROLE_RAIDER, self.raider_bot_ids),
            (ROLE_REPAIR_PATROL, self.repair_patrol_bot_ids),
        ):
            for uid in tuple(tracked_set):
                if uid in visible_builder_ids:
                    continue
                tracked_set.discard(uid)
                self.spawned_bot_roles.pop(uid, None)
                role_name = _ROLE_NAME_MAP.get(role, "unknown")
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
        # turrets. The caller already tracks defender/healer/miner/raider/
        # repair_patrol builder counts; subtract the turret count too so
        # whatever remains is the set of economy builder bots.
        turret_count = self._count_friendly_turrets(c)
        miner_count = len(self.miner_bot_ids)
        raider_count = len(self.raider_bot_ids)
        repair_count = len(self.repair_patrol_bot_ids)
        economy_count = friendly_units_including_core - (
            1
            + int(defender_count)
            + int(healer_count)
            + int(turret_count)
            + miner_count
            + raider_count
            + repair_count
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
        elif role == ROLE_MINER:
            marker_event = "placed_miner_marker"
        elif role == ROLE_RAIDER:
            marker_event = "placed_raider_marker"
        elif role == ROLE_REPAIR_PATROL:
            marker_event = "placed_repair_patrol_marker"
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
        elif role == ROLE_HEALER:
            spawned_id = self._spawn_builder_at_core_with_fallback(
                c,
                "spawned_healer",
                "spawned_healer_fallback",
            )
        elif role == ROLE_MINER:
            spawned_id = self._spawn_builder_at_core_with_fallback(
                c,
                "spawned_miner",
                "spawned_miner_fallback",
            )
        elif role == ROLE_RAIDER:
            spawned_id = self._spawn_builder_at_core_with_fallback(
                c,
                "spawned_raider",
                "spawned_raider_fallback",
            )
        elif role == ROLE_REPAIR_PATROL:
            spawned_id = self._spawn_builder_at_core_with_fallback(
                c,
                "spawned_repair_patrol",
                "spawned_repair_patrol_fallback",
            )
        else:
            return False

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
            elif etype in (EntityType.GUNNER, EntityType.SENTINEL, EntityType.BREACH):
                self._run_turret(c)
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

        # v4_micky: emergency defender insurance. If core HP drops below
        # 70% (well below the normal 100%), enable defender spawning so
        # the launcher chain starts. On default maps no enemy attacks
        # reach our core so this never triggers and target stays at 0.
        # Against aggressive ladder opponents this gives us a fallback.
        try:
            if core_max > 0 and c.get_hp() < int(0.7 * core_max):
                if self.target_defender_count < 1:
                    self.target_defender_count = 1
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

        # v14_no1_ladder: replay-inspired economy ladder. The #1 replay
        # snowballs by keeping several builders active and continuously
        # expanding titanium throughput. Keep Ti gates conservative enough
        # that tiny maps still get their refinery/titanium bootstrap first.
        try:
            ti_late, _ = c.get_global_resources()
            if rnd >= 300 and ti_late > 800 and self.target_economy_count < 2:
                self.target_economy_count = 2
            if rnd >= 600 and ti_late > 1200 and self.target_economy_count < 3:
                self.target_economy_count = 3
            if rnd >= 900 and ti_late > 1800 and self.target_economy_count < 4:
                self.target_economy_count = 4
            if rnd >= 1200 and ti_late > 2400 and self.target_economy_count < 5:
                self.target_economy_count = 5
        except GameError:
            pass

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

            # --- Combat role waves (v3_merged flip) -------------------
            # Miner wave: once defense complete, periodic miner spawns
            # gated by round interval AND Ti floor.
            try:
                ti, _ = c.get_global_resources()
                scale = c.get_scale_percent()
                builder_cost = int(BUILDER_BOT_BASE_COST[0] * scale / 100)
            except GameError:
                ti = 0
                builder_cost = 30

            miner_count = len(self.miner_bot_ids)
            raider_count = len(self.raider_bot_ids)

            # Economy-health gate: never siphon Ti from harvester pipeline on
            # small maps. Require >=2 friendly harvesters OR round>=300.
            # Fallback round cap ensures combat still kicks in if economy stalls.
            #
            # v4_micky: count harvesters via vision (not self.builder_state.
            # harvester_ids_friendly, which only exists on EconomyState — the
            # core's self.builder_state is None). Without this, harvester_count
            # was always 0 from core perspective, so combat_econ_ready only
            # fired via rnd>=300 fallback and repair_patrol's harvester>=3 gate
            # NEVER fired at all.
            harvester_count = self._count_visible_friendly_harvesters(c)
            combat_econ_ready = harvester_count >= 2 or rnd >= 300

            # Post-defense miner waves: r >= 70, every 60 rounds, Ti floor,
            # economy stood up. (v5_swarm: kept at default cadence; aggressive
            # variant tested separately and regressed test1.)
            if (
                self.defense_complete
                and rnd >= 70
                and combat_econ_ready
                and (rnd - self._last_miner_wave_round) >= self.miner_wave_interval
                and ti >= builder_cost + 50
            ):
                if self._spawn_builder_for_role(c, ROLE_MINER):
                    return

            # v12_smart_sentinel: keep the first repair patrol early, but
            # gate the second behind a Ti buffer so it cannot steal scale/Ti
            # during foundry bootstrap.
            repair_count = len(self.repair_patrol_bot_ids)
            first_repair_ok = repair_count == 0
            second_repair_ok = repair_count >= 1 and rnd >= 600 and ti >= 1500
            if (
                self.defense_complete
                and rnd >= 80
                and self.economy_spawned
                and harvester_count >= 3
                and repair_count < self.target_repair_patrol_count
                and (first_repair_ok or second_repair_ok)
                and ti >= builder_cost + 30
            ):
                if self._spawn_builder_for_role(c, ROLE_REPAIR_PATROL):
                    return

            # v17_no1_combo: more replay-like raider cadence. The v14 economy
            # ladder should keep enough Ti flowing that these waves are less
            # likely to starve harvesters than the earlier pure-raider tests.
            if (
                self.defense_complete
                and rnd >= 120
                and combat_econ_ready
            ):
                max_raiders = _max_raiders_for_round(rnd)
                force_raider_round = rnd in (
                    200, 280, 360, 440, 520, 600, 700, 800, 900,
                    1000, 1100, 1200, 1300, 1400, 1500, 1600,
                    1700, 1800, 1900,
                )
                ti_floor = builder_cost if force_raider_round else builder_cost + 40
                if raider_count < max_raiders and ti >= ti_floor:
                    if self._spawn_builder_for_role(c, ROLE_RAIDER):
                        return

        # v4_micky: when defender is intentionally off (target_defender_count == 0),
        # bypass the launcher+marker requirement and trip defense_complete from
        # a softer harvester+round signal. Otherwise raiders / repair patrol /
        # miner waves never spawn (all gated on self.defense_complete at
        # main.py:976, 988, 1001). Without this bypass, killing the defender
        # also silently kills all combat.
        if not self.defense_complete and self.target_defender_count == 0:
            harvester_count_def = self._count_visible_friendly_harvesters(c)
            if harvester_count_def >= 3 and rnd >= 80:
                self.defense_complete = True
                self.defense_marker_confirmed = True
                log_event(
                    rnd, c.get_id(), "core", f"({cx},{cy})",
                    "defense_complete_bypass",
                    harvesters=harvester_count_def,
                )

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
                    if role == ROLE_MINER:
                        assigned_role = ROLE_MINER
                        log_event(
                            c.get_current_round(), c.get_id(), "builder",
                            f"({p.x},{p.y})", "read_marker_miner",
                            val=marker_val, mcx=mcx, mcy=mcy,
                        )
                        break
                    if role == ROLE_RAIDER:
                        assigned_role = ROLE_RAIDER
                        log_event(
                            c.get_current_round(), c.get_id(), "builder",
                            f"({p.x},{p.y})", "read_marker_raider",
                            val=marker_val, mcx=mcx, mcy=mcy,
                        )
                        break
                    if role == ROLE_REPAIR_PATROL:
                        assigned_role = ROLE_REPAIR_PATROL
                        log_event(
                            c.get_current_round(), c.get_id(), "builder",
                            f"({p.x},{p.y})", "read_marker_repair_patrol",
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
            elif self.role == ROLE_MINER:
                self.builder_state = MinerState(self.core_pos, c.get_id())
                log_event(c.get_current_round(), c.get_id(),
                          "miner", f"({p.x},{p.y})", "miner_init")
            elif self.role == ROLE_RAIDER:
                self.builder_state = RaiderState(
                    self.core_pos, self.map_w, self.map_h,
                )
                log_event(c.get_current_round(), c.get_id(),
                          "raider", f"({p.x},{p.y})", "raider_init")
            elif self.role == ROLE_REPAIR_PATROL:
                self.builder_state = RepairPatrolState(self.core_pos)
                log_event(c.get_current_round(), c.get_id(),
                          "repair_patrol", f"({p.x},{p.y})",
                          "repair_patrol_init")
            else:
                log_event(c.get_current_round(), c.get_id(),
                          "builder", f"({p.x},{p.y})", "role_unknown")
                try:
                    c.self_destruct()
                except GameError:
                    pass
                return

        if self._try_builder_forward_turret(c):
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
        elif self.role == ROLE_MINER:
            if isinstance(self.builder_state, MinerState):
                run_miner(c, self.builder_state, self.local_map)
            return
        elif self.role == ROLE_RAIDER:
            if isinstance(self.builder_state, RaiderState):
                run_raider(c, self.builder_state, self.local_map)
            return
        elif self.role == ROLE_REPAIR_PATROL:
            if isinstance(self.builder_state, RepairPatrolState):
                run_repair_patrol(c, self.builder_state, self.local_map)
            return
        else:
            # Unknown role — no-op.
            return

    # --- Combat turret logic (ported from xuanming / combat main.py) -------
    # Replaces faisal's weak _run_gunner (which only ever targeted enemy core).
    # Prioritized fire: builder_bot > turret > core > harvester > transport >
    # other > barrier > marker. Same helpers serve GUNNER / SENTINEL / BREACH.

    def _iter_enemy_targets(self, c: Controller):
        my_team = c.get_team()
        seen = set()
        for entity_id in c.get_nearby_entities():
            if entity_id in seen:
                continue
            seen.add(entity_id)
            try:
                if c.get_team(entity_id) == my_team:
                    continue
                yield entity_id
            except GameError:
                continue

    def _enemy_target_priority(self, c: Controller, entity_id: int):
        try:
            etype = c.get_entity_type(entity_id)
        except GameError:
            return 9
        if etype == EntityType.BUILDER_BOT:
            return 0
        if etype in (
            EntityType.GUNNER,
            EntityType.SENTINEL,
            EntityType.BREACH,
            EntityType.LAUNCHER,
        ):
            return 1
        if etype == EntityType.CORE:
            return 2
        if etype == EntityType.HARVESTER:
            return 3
        if etype in (
            EntityType.BRIDGE,
            EntityType.CONVEYOR,
            EntityType.SPLITTER,
            EntityType.ARMOURED_CONVEYOR,
        ):
            return 4
        if etype == EntityType.BARRIER:
            return 6
        if etype == EntityType.MARKER:
            return 8
        return 5

    def _best_turret_target(self, c: Controller, pos: Position):
        best_target = None
        best_key = None
        for entity_id in self._iter_enemy_targets(c):
            try:
                target_pos = c.get_position(entity_id)
            except GameError:
                continue
            try:
                if not c.can_fire(target_pos):
                    continue
            except GameError:
                continue
            prio = self._enemy_target_priority(c, entity_id)
            d2 = pos.distance_squared(target_pos)
            key = (prio, d2, target_pos.x, target_pos.y)
            if best_key is None or key < best_key:
                best_key = key
                best_target = target_pos
        return best_target

    def _best_rotation_direction(self, c: Controller, pos: Position):
        best_dir = None
        best_key = None
        for entity_id in self._iter_enemy_targets(c):
            try:
                target_pos = c.get_position(entity_id)
            except GameError:
                continue
            prio = self._enemy_target_priority(c, entity_id)
            d2 = pos.distance_squared(target_pos)
            face_dir = pos.direction_to(target_pos)
            key = (prio, d2, target_pos.x, target_pos.y)
            if best_key is None or key < best_key:
                best_key = key
                best_dir = face_dir
        return best_dir

    def _run_turret(self, c: Controller):
        if not self.initialised:
            self._init_game(c)
        if not self.initialised:
            return

        etype = c.get_entity_type()
        self._update_local_map(c, etype, MAP_UPDATE_EVERY_N_TURRET)

        pos = c.get_position()
        rnd = c.get_current_round()
        uid = c.get_id()
        turret_name = str(etype.name).lower()

        if c.get_action_cooldown() == 0:
            target = self._best_turret_target(c, pos)
            if target is not None:
                try:
                    c.fire(target)
                    log_event(
                        rnd,
                        uid,
                        turret_name,
                        f"({pos.x},{pos.y})",
                        "fired",
                        tx=target.x,
                        ty=target.y,
                    )
                    return
                except GameError:
                    pass

        if etype == EntityType.GUNNER and c.get_action_cooldown() == 0:
            face_dir = self._best_rotation_direction(c, pos)
            if face_dir is not None:
                try:
                    if c.can_rotate(face_dir):
                        c.rotate(face_dir)
                        log_event(
                            rnd,
                            uid,
                            turret_name,
                            f"({pos.x},{pos.y})",
                            "rotated",
                            d=face_dir.name,
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
