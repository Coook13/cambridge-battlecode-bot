"""Main entry point for the ultimate bot.

Orchestrates the core spawn controller and per-unit role dispatch.
Key enhancements over the starter bot:

    1. PARALLEL STARTUP — Spawns 3 bots in the first 5 rounds (defender
       + 2 quick miners) instead of 1, getting economy flowing ~15 rounds
       earlier.

    2. MARKER-FIRST PROTOCOL — Core places the role marker BEFORE spawning
       the bot. Bot reads marker on tick 1, re-reads on tick 2 to confirm.
       Falls back to DEFENDER if both reads fail.

    3. EXPANDED ROLE DISPATCH — Supports 7 roles: defender, economy, healer,
       miner, raider, scout, repair_patrol.

    4. ADAPTIVE SPAWN POLICY — After initial queue, spawns are resource-gated
       and threat-aware (O(1) decision logic per round).

All roles delegate to their own modules (builder.py, economy.py, miner.py,
raider.py, repair_patrol.py).
"""

import time

from cambc import Controller, EntityType, GameError, Position
from builder import DefenderState, HealerState, run_defender, run_healer, seed_guard_tables
from economy import EconomyState, run_economy
from miner import MinerState, run_miner
from raider import RaiderState, run_raider
from repair_patrol import RepairPatrolState, run_repair_patrol
from constants import (
    ACTION_RADIUS_SQ,
    BUILDER_BOT_BASE_COST,
    CARDINAL_DELTAS,
    COMPETITION_MODE,
    CORE_MAX_HP,
    CORE_VISION_RADIUS_SQ,
    BREACH_VISION_RADIUS_SQ,
    GUNNER_VISION_RADIUS_SQ,
    LAUNCHER_VISION_RADIUS_SQ,
    MAP_DEBUG_CORE_ONLY,
    MAP_UPDATE_EVERY_N_BUILDER,
    MAP_UPDATE_EVERY_N_CORE,
    MAP_UPDATE_EVERY_N_TURRET,
    MSG_DEFENSE_COMPLETE,
    RESET_LOGS_ON_START,
    ROLE_DEFENDER,
    ROLE_ECONOMY,
    ROLE_HEALER,
    ROLE_MINER,
    ROLE_RAIDER,
    ROLE_REPAIR_PATROL,
    SENTINEL_VISION_RADIUS_SQ,
    SHOW_MAP_DEBUG,
    VALID_ROLES,
    decode_marker,
    encode_marker,
    marker_tile_candidates,
)
from local_map import UnitLocalMap, VisionLocalMap
from logger import (
    consume_tick_log_overhead_us,
    init_logger,
    log_event,
    log_local_map,
    log_timing_event,
    reset_run_logs,
    reset_tick_log_overhead,
)
from symmetry import detect_symmetry


_ROLE_NAMES = {
    ROLE_DEFENDER: "defender",
    ROLE_ECONOMY: "economy",
    ROLE_HEALER: "healer",
    ROLE_MINER: "miner",
    ROLE_RAIDER: "raider",
    ROLE_REPAIR_PATROL: "repair_patrol",
}


class CoreSpawnScheduler:
    """Manages the core's bot spawning across game phases.

    Phase 1 (rounds 1-5):  Fixed queue — defender, miner #1, miner #2.
    Phase 2 (post-defense): Economy bot via launcher.
    Phase 3 (round 80+):   Repair patrol (if economy is active).
    Phase 4 (round 70+):   Extra miners every 60 rounds (resource-gated).
    Phase 5 (round 150+):  Raiders (resource-gated).
    Emergency:             Healer when core HP < 80%.

    CPU cost: O(1) per round — simple threshold checks.
    """

    _INITIAL_QUEUE = [
        (1, ROLE_DEFENDER),
        (3, ROLE_MINER),
        (5, ROLE_MINER),
    ]

    _MINER_WAVE_INTERVAL = 60

    def __init__(self):
        self.queue_idx = 0
        self.defense_complete = False
        self.economy_spawned = False
        self.repair_patrol_spawned = False
        self.raider_count = 0
        self.miner_count = 0
        self.healer_active = False
        self.healer_marker_placed = False
        self._last_miner_wave_round = 0

    def should_spawn_healer(self, c, core_pos):
        """Check if healer is needed. O(1)."""
        if self.healer_active:
            return False
        try:
            hp = c.get_hp()
            if hp < 0.8 * CORE_MAX_HP:
                return True
        except GameError:
            pass
        return False

    def get_next_spawn(self, c, round_num):
        """Return role to spawn, or None. O(1)."""

        if self.queue_idx < len(self._INITIAL_QUEUE):
            min_round, role = self._INITIAL_QUEUE[self.queue_idx]
            if round_num >= min_round:
                return role
            return None

        if (
            round_num >= 80
            and not self.repair_patrol_spawned
            and self.economy_spawned
        ):
            return ROLE_REPAIR_PATROL

        if (
            round_num >= 70
            and round_num - self._last_miner_wave_round >= self._MINER_WAVE_INTERVAL
        ):
            try:
                ti, _ = c.get_global_resources()
                scale = c.get_scale_percent()
                builder_cost = int(BUILDER_BOT_BASE_COST[0] * scale / 100)
                if ti >= builder_cost + 40:
                    return ROLE_MINER
            except GameError:
                pass

        if round_num >= 150 and self.raider_count < _max_raiders(round_num):
            try:
                ti, _ = c.get_global_resources()
                scale = c.get_scale_percent()
                builder_cost = int(BUILDER_BOT_BASE_COST[0] * scale / 100)
                if ti >= builder_cost + 60:
                    return ROLE_RAIDER
            except GameError:
                pass

        return None

    def advance_queue(self, role, round_num=0):
        if self.queue_idx < len(self._INITIAL_QUEUE):
            self.queue_idx += 1
        if role == ROLE_MINER:
            self.miner_count += 1
            self._last_miner_wave_round = round_num
        elif role == ROLE_ECONOMY:
            self.economy_spawned = True
        elif role == ROLE_REPAIR_PATROL:
            self.repair_patrol_spawned = True
        elif role == ROLE_RAIDER:
            self.raider_count += 1
        elif role == ROLE_HEALER:
            self.healer_active = True


def _max_raiders(round_num):
    if round_num < 200:
        return 1
    if round_num < 400:
        return 2
    if round_num < 800:
        return 3
    return 4


class Player:
    """Entry point called by the game engine each round."""

    def __init__(self):
        init_logger()
        self.initialised = False
        self.core_pos = None
        self.symmetry = None
        self.local_map = None
        self.map_w = None
        self.map_h = None

        self.role = None
        self.builder_state = None
        self._role_read_tick = 0

        self.scheduler = CoreSpawnScheduler()
        self._pending_marker_role = None
        self._timing_meta = {}
        self._map_created_round = None

    def _role_marker_positions(self):
        if self.core_pos is None or self.map_w is None or self.map_h is None:
            return ()
        coords = marker_tile_candidates(
            self.core_pos.x, self.core_pos.y, self.map_w, self.map_h
        )
        return tuple(Position(x, y) for x, y in coords)

    def _spawn_offsets_for_role(self, role):
        """Spawn preference ordering by role.

        Early miners are deliberately limited to two pre-defense spawns, so we
        only need stable cardinal spreading rather than aggressive early crowding.
        """

        def _rotated_cardinals(start_idx: int):
            base = [(0, 1), (1, 0), (0, -1), (-1, 0)]
            start_idx %= 4
            ordered = base[start_idx:] + base[:start_idx]
            return tuple(ordered + [(1, -1), (1, 1), (-1, 1), (-1, -1), (0, 0)])

        if role == ROLE_MINER:
            return _rotated_cardinals(self.scheduler.miner_count)

        if role == ROLE_RAIDER:
            return _rotated_cardinals(self.scheduler.raider_count)

        if role in (ROLE_DEFENDER, ROLE_HEALER, ROLE_REPAIR_PATROL):
            return (
                (0, 0),
                (0, -1), (1, 0), (0, 1), (-1, 0),
                (1, -1), (1, 1), (-1, 1), (-1, -1),
            )

        return _rotated_cardinals(0)

    def _find_friendly_launcher_xy(self, c):
        try:
            for bid in c.get_nearby_buildings():
                try:
                    if c.get_entity_type(bid) != EntityType.LAUNCHER:
                        continue
                    if c.get_team(bid) != c.get_team():
                        continue
                    lp = c.get_position(bid)
                    return (lp.x, lp.y)
                except GameError:
                    continue
        except GameError:
            pass
        return None

    def _launcher_wait_tile_xy(self, launcher_xy):
        if launcher_xy is None or self.core_pos is None:
            return None

        lx, ly = launcher_xy
        cx, cy = self.core_pos.x, self.core_pos.y
        candidates = []
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                wx, wy = cx + dx, cy + dy
                ddx = wx - lx
                ddy = wy - ly
                if ddx * ddx + ddy * ddy <= ACTION_RADIUS_SQ:
                    candidates.append((wx, wy))
        if not candidates:
            return None
        return min(candidates, key=lambda t: abs(t[0] - cx) + abs(t[1] - cy))

    def _choose_fallback_role(self, c):
        rnd = c.get_current_round()
        uid = c.get_id()

        try:
            pos = c.get_position()
            cur_xy = (pos.x, pos.y)
        except GameError:
            cur_xy = None

        launcher_xy = self._find_friendly_launcher_xy(c)
        wait_xy = self._launcher_wait_tile_xy(launcher_xy)

        if launcher_xy is not None and cur_xy is not None and cur_xy == wait_xy:
            return ROLE_ECONOMY

        allow_patrol = (rnd >= 80 and self.scheduler.economy_spawned)
        bucket = (uid + rnd) % 6

        if launcher_xy is not None and allow_patrol:
            weighted = [
                ROLE_DEFENDER,
                ROLE_MINER,
                ROLE_ECONOMY,
                ROLE_REPAIR_PATROL,
                ROLE_MINER,
                ROLE_ECONOMY,
            ]
        elif launcher_xy is not None:
            weighted = [
                ROLE_DEFENDER,
                ROLE_MINER,
                ROLE_ECONOMY,
                ROLE_MINER,
            ]
        else:
            weighted = [
                ROLE_DEFENDER,
                ROLE_MINER,
                ROLE_DEFENDER,
                ROLE_MINER,
            ]

        return weighted[bucket % len(weighted)]

    def _seed_launcher_hint(self, c):
        if self.builder_state is None:
            return
        launcher_xy = self._find_friendly_launcher_xy(c)
        if launcher_xy is None:
            return

        try:
            if isinstance(self.builder_state, DefenderState):
                self.builder_state.launcher_pos = launcher_xy
        except Exception:
            pass

        try:
            if isinstance(self.builder_state, EconomyState):
                self.builder_state.launcher_xy = launcher_xy
        except Exception:
            pass

    def _vision_radius_sq(self, etype):
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
            log_event(rnd, uid, et, pos, "fatal_exception", force=True, err=repr(e))

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
        startup_us = int(self._timing_meta.get("startup_reset_logs_us", 0))
        net_us = max(0, elapsed_us - log_overhead_us - startup_us)
        self._timing_meta["tick_raw_us"] = elapsed_us
        self._timing_meta["log_overhead_us"] = log_overhead_us
        log_timing_event(rnd, uid, et, pos, net_us, **self._timing_meta)

    def _init_game(self, c: Controller):
        my_team = c.get_team()
        try:
            my_type = c.get_entity_type()
        except GameError:
            my_type = None

        try:
            if my_type == EntityType.CORE:
                self.core_pos = c.get_position()
        except GameError:
            self.core_pos = None

        if self.core_pos is None:
            for bid in c.get_nearby_buildings():
                try:
                    if (
                        c.get_entity_type(bid) == EntityType.CORE
                        and c.get_team(bid) == my_team
                    ):
                        self.core_pos = c.get_position(bid)
                        break
                except GameError:
                    continue

        if not self.core_pos:
            return

        map_w = c.get_map_width()
        map_h = c.get_map_height()
        self.map_w = map_w
        self.map_h = map_h

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
                self._vision_radius_sq(my_type),
            )

        if self.local_map is not None:
            self.local_map.set_friendly_core(self.core_pos)
            self._map_created_round = c.get_current_round()

        self.initialised = True
        log_event(
            c.get_current_round(),
            c.get_id(),
            "core",
            f"({self.core_pos.x},{self.core_pos.y})",
            "init_done",
            symmetry=self.symmetry,
            w=map_w,
            h=map_h,
        )

    def _update_local_map(self, c, entity_type, stride):
        if self.local_map is None:
            return
        rnd = c.get_current_round()
        if self._map_created_round is not None and rnd <= self._map_created_round:
            return
        if stride > 1 and (rnd % stride) != 0:
            return

        if COMPETITION_MODE:
            self.local_map.update_from_controller(c)
        else:
            t0 = time.perf_counter_ns()
            self.local_map.update_from_controller(c)
            us = (time.perf_counter_ns() - t0) // 1000
            self._timing_meta["map_update_us"] = us
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
            if RESET_LOGS_ON_START and not COMPETITION_MODE:
                t0 = time.perf_counter_ns()
                reset_run_logs()
                self._timing_meta["startup_reset_logs_us"] = (
                    time.perf_counter_ns() - t0
                ) // 1000
            self._init_game(c)

        if not self.initialised:
            return

        self._update_local_map(c, EntityType.CORE, MAP_UPDATE_EVERY_N_CORE)

        rnd = c.get_current_round()
        uid = c.get_id()
        cx = self.core_pos.x
        cy = self.core_pos.y

        log_event(
            rnd,
            uid,
            "core",
            f"({cx},{cy})",
            "core_tick",
            q_idx=self.scheduler.queue_idx,
            defense=int(self.scheduler.defense_complete),
            econ=int(self.scheduler.economy_spawned),
            cd=c.get_action_cooldown(),
        )

        if self.scheduler.should_spawn_healer(c, self.core_pos):
            healed = self._spawn_with_marker(c, ROLE_HEALER, "healer", rnd, uid)
            if healed:
                return

        try:
            if c.get_hp() >= CORE_MAX_HP and self.scheduler.healer_active:
                self.scheduler.healer_active = False
                self.scheduler.healer_marker_placed = False
        except GameError:
            pass

        if self._pending_marker_role is not None:
            if c.get_action_cooldown() == 0:
                role = self._pending_marker_role

                if self.scheduler.defense_complete and role in (ROLE_MINER, ROLE_RAIDER):
                    spawned = self._spawn_launcher_queue_builder(c, rnd, uid, role)
                else:
                    spawned = self._try_spawn_bot(c, rnd, uid, role)

                if spawned:
                    self.scheduler.advance_queue(role, rnd)
                    self._pending_marker_role = None
                    log_event(
                        rnd,
                        uid,
                        "core",
                        f"({cx},{cy})",
                        f"spawned_{_ROLE_NAMES.get(role, 'bot')}",
                        role=role,
                    )
            return

        if not self.scheduler.defense_complete:
            self._check_defense_complete(c, rnd, uid)

        if self.scheduler.defense_complete and not self.scheduler.economy_spawned:
            marker_ok = self._place_role_marker(c, ROLE_ECONOMY)
            if marker_ok:
                if c.get_action_cooldown() == 0:
                    spawned = self._spawn_economy_builder(c, rnd, uid)
                    if spawned:
                        self.scheduler.advance_queue(ROLE_ECONOMY, rnd)
                return

        next_role = self.scheduler.get_next_spawn(c, rnd)
        if next_role is not None:
            if self.scheduler.defense_complete and next_role in (ROLE_MINER, ROLE_RAIDER):
                marker_ok = self._place_role_marker(c, next_role)
                if marker_ok and c.get_action_cooldown() == 0:
                    spawned = self._spawn_launcher_queue_builder(c, rnd, uid, next_role)
                    if spawned:
                        self.scheduler.advance_queue(next_role, rnd)
                return

            self._spawn_with_marker(
                c,
                next_role,
                _ROLE_NAMES.get(next_role, "bot"),
                rnd,
                uid,
            )

    def _spawn_with_marker(self, c, role, role_name, rnd, uid):
        cx, cy = self.core_pos.x, self.core_pos.y

        marker_ok = self._place_role_marker(c, role)
        if not marker_ok:
            return False

        if c.get_action_cooldown() == 0:
            spawned = self._try_spawn_bot(c, rnd, uid, role)
            if spawned:
                self.scheduler.advance_queue(role, rnd)
                log_event(
                    rnd,
                    uid,
                    "core",
                    f"({cx},{cy})",
                    f"spawned_{role_name}",
                    role=role,
                )
                return True

        self._pending_marker_role = role
        return False

    def _place_role_marker(self, c, role):
        marker_positions = self._role_marker_positions()
        if not marker_positions or self.core_pos is None:
            return False

        marker_val = encode_marker(role, self.core_pos.x, self.core_pos.y)

        for mp in marker_positions:
            try:
                bid = c.get_tile_building_id(mp)
                if bid is None or bid == 0:
                    continue
                if c.get_entity_type(bid) != EntityType.MARKER:
                    continue
                if c.get_team(bid) != c.get_team():
                    continue
                if c.get_marker_value(bid) == marker_val:
                    return True
            except GameError:
                continue

        for mp in marker_positions:
            try:
                bid = c.get_tile_building_id(mp)
                if bid is not None and bid != 0:
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
                        c.get_current_round(),
                        c.get_id(),
                        "core",
                        f"({self.core_pos.x},{self.core_pos.y})",
                        "placed_role_marker",
                        role=role,
                        mx=mp.x,
                        my=mp.y,
                    )
                    return True
            except GameError:
                continue

        return False

    def _try_spawn_bot(self, c, rnd, uid, role=None):
        cx, cy = self.core_pos.x, self.core_pos.y

        offsets = self._spawn_offsets_for_role(role) if role is not None else (
            (0, -1), (1, 0), (0, 1), (-1, 0),
            (1, -1), (1, 1), (-1, 1), (-1, -1),
            (0, 0),
        )

        for dx, dy in offsets:
            sp = Position(cx + dx, cy + dy)
            try:
                if c.can_spawn(sp):
                    c.spawn_builder(sp)
                    return True
            except GameError:
                continue
        return False

    def _spawn_economy_builder(self, c, rnd, uid):
        """Spawn economy builder only on the launcher queue tile.

        This avoids silently spawning the economy bot into the wrong inside-base
        tile where it may look like it is doing nothing.
        """
        cx, cy = self.core_pos.x, self.core_pos.y

        launcher_xy = self._find_friendly_launcher_xy(c)
        wait_xy = self._launcher_wait_tile_xy(launcher_xy)

        if wait_xy is None:
            log_event(
                rnd,
                uid,
                "core",
                f"({cx},{cy})",
                "economy_waiting_for_launcher",
            )
            return False

        sp = Position(wait_xy[0], wait_xy[1])
        try:
            if c.can_spawn(sp):
                c.spawn_builder(sp)
                log_event(
                    rnd,
                    uid,
                    "core",
                    f"({cx},{cy})",
                    "spawned_economy_builder",
                    sx=sp.x,
                    sy=sp.y,
                )
                return True
        except GameError:
            pass

        log_event(
            rnd,
            uid,
            "core",
            f"({cx},{cy})",
            "economy_spawn_blocked",
            sx=sp.x,
            sy=sp.y,
        )
        return False

    def _spawn_launcher_queue_builder(self, c, rnd, uid, role):
        if self.core_pos is None:
            return False

        cx, cy = self.core_pos.x, self.core_pos.y
        launcher_xy = self._find_friendly_launcher_xy(c)
        wait_xy = self._launcher_wait_tile_xy(launcher_xy)

        if wait_xy is None:
            return False

        sp = Position(wait_xy[0], wait_xy[1])
        try:
            if c.can_spawn(sp):
                c.spawn_builder(sp)
                log_event(
                    rnd,
                    uid,
                    "core",
                    f"({cx},{cy})",
                    "spawned_launcher_queue_builder",
                    role=role,
                    sx=sp.x,
                    sy=sp.y,
                )
                return True
        except GameError:
            pass

        return False

    def _check_defense_complete(self, c, rnd, uid):
        for mp in self._role_marker_positions():
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

                val = c.get_marker_value(bid)
                if val == MSG_DEFENSE_COMPLETE:
                    self.scheduler.defense_complete = True
                    return

                role, mcx, mcy, _ = decode_marker(val)
                if self.core_pos is not None:
                    if (
                        role == MSG_DEFENSE_COMPLETE
                        and mcx == self.core_pos.x
                        and mcy == self.core_pos.y
                    ):
                        self.scheduler.defense_complete = True
                        return

                launcher_found = False
                for bid2 in c.get_nearby_buildings():
                    try:
                        if c.get_entity_type(bid2) != EntityType.LAUNCHER:
                            continue
                        if c.get_team(bid2) != c.get_team():
                            continue
                        lp = c.get_position(bid2)
                        cx = self.core_pos.x
                        cy = self.core_pos.y
                        if max(abs(lp.x - cx), abs(lp.y - cy)) <= 2:
                            launcher_found = True
                            break
                    except GameError:
                        continue

                if launcher_found and (
                    role == MSG_DEFENSE_COMPLETE or val == MSG_DEFENSE_COMPLETE
                ):
                    self.scheduler.defense_complete = True
                    log_event(
                        rnd,
                        uid,
                        "core",
                        f"({self.core_pos.x},{self.core_pos.y})",
                        "defense_complete_confirmed",
                    )
                    return
            except GameError:
                continue

    def _run_bot(self, c: Controller):
        if not self.initialised:
            self._init_game(c)
        if not self.initialised:
            return

        self._update_local_map(c, EntityType.BUILDER_BOT, MAP_UPDATE_EVERY_N_BUILDER)

        if self.builder_state is None and self.role is None:
            self._role_read_tick += 1
            assigned_role = self._read_role_marker(c)

            if assigned_role is None:
                if self._role_read_tick >= 2:
                    assigned_role = self._choose_fallback_role(c)
                    log_event(
                        c.get_current_round(),
                        c.get_id(),
                        "builder",
                        "(?,?)",
                        "marker_read_fallback_role",
                        fallback=assigned_role,
                        name=_ROLE_NAMES.get(assigned_role, "unknown"),
                    )
                else:
                    return

            self.role = assigned_role
            self._init_role(c, assigned_role)

        if self._role_read_tick == 1 and self.role is not None:
            self._role_read_tick = 2
            confirmed = self._read_role_marker(c)
            if confirmed is not None and confirmed != self.role:
                log_event(
                    c.get_current_round(),
                    c.get_id(),
                    "builder",
                    "(?,?)",
                    "role_overridden",
                    old=self.role,
                    new=confirmed,
                )
                self.role = confirmed
                self._init_role(c, confirmed)

        self._dispatch_role(c)

    def _read_role_marker(self, c):
        my_team = c.get_team()
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

                val = c.get_marker_value(bid)
                role, mcx, mcy, _ = decode_marker(val)

                if self.core_pos is not None:
                    if mcx != self.core_pos.x or mcy != self.core_pos.y:
                        continue
                if role not in VALID_ROLES:
                    continue

                log_event(
                    c.get_current_round(),
                    c.get_id(),
                    "builder",
                    "(?,?)",
                    f"read_marker_{_ROLE_NAMES.get(role, '?')}",
                    val=val,
                    role=role,
                )
                return role
            except GameError:
                continue
        return None

    def _init_role(self, c, role):
        p = c.get_position()
        rnd = c.get_current_round()
        uid = c.get_id()

        if role == ROLE_DEFENDER:
            self.builder_state = DefenderState(
                self.core_pos, self.map_w, self.map_h, self.local_map
            )
            if self.scheduler.defense_complete:
                seed_guard_tables(self.builder_state)
                self.builder_state.phase = "guard"

        elif role == ROLE_HEALER:
            self.builder_state = HealerState(self.core_pos, self.local_map)
        elif role == ROLE_ECONOMY:
            self.builder_state = EconomyState(self.core_pos)
        elif role == ROLE_MINER:
            self.builder_state = MinerState(self.core_pos, uid)
        elif role == ROLE_RAIDER:
            self.builder_state = RaiderState(self.core_pos, self.map_w, self.map_h)
        elif role == ROLE_REPAIR_PATROL:
            self.builder_state = RepairPatrolState(self.core_pos)
        else:
            try:
                c.self_destruct()
            except GameError:
                pass
            return

        self._seed_launcher_hint(c)

        log_event(
            rnd,
            uid,
            _ROLE_NAMES.get(role, "bot"),
            f"({p.x},{p.y})",
            f"{_ROLE_NAMES.get(role, 'bot')}_init",
        )

    def _dispatch_role(self, c):
        if self.role == ROLE_DEFENDER:
            if isinstance(self.builder_state, DefenderState):
                run_defender(c, self.builder_state)

        elif self.role == ROLE_HEALER:
            if isinstance(self.builder_state, HealerState):
                run_healer(c, self.builder_state)

        elif self.role == ROLE_ECONOMY:
            if not self._is_adjacency_ready():
                return
            if not isinstance(self.builder_state, EconomyState):
                self.builder_state = EconomyState(self.core_pos)
            run_economy(c, self.builder_state, self.local_map)

        elif self.role == ROLE_MINER:
            if isinstance(self.builder_state, MinerState):
                run_miner(c, self.builder_state, self.local_map)

        elif self.role == ROLE_RAIDER:
            if isinstance(self.builder_state, RaiderState):
                run_raider(c, self.builder_state, self.local_map)

        elif self.role == ROLE_REPAIR_PATROL:
            if isinstance(self.builder_state, RepairPatrolState):
                run_repair_patrol(c, self.builder_state, self.local_map)

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
        if not self.initialised:
            self._init_game(c)
        if not self.initialised:
            return

        self._update_local_map(c, EntityType.LAUNCHER, MAP_UPDATE_EVERY_N_TURRET)

        pos = c.get_position()
        rnd = c.get_current_round()
        uid = c.get_id()

        if c.get_action_cooldown() > 0:
            return

        cx, cy = self.core_pos.x, self.core_pos.y

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
                wait_candidates, key=lambda t: abs(t[0] - cx) + abs(t[1] - cy)
            )
        else:
            wait_x, wait_y = cx + 1, cy

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

                def _landing_penalty(lx: int, ly: int):
                    penalty_enemy_transport = 0
                    penalty_dead_pocket = 0
                    penalty_on_building = 0

                    lp2 = Position(lx, ly)
                    try:
                        bid2 = c.get_tile_building_id(lp2)
                    except GameError:
                        bid2 = None

                    if bid2 is not None and bid2 != 0:
                        try:
                            et2 = c.get_entity_type(bid2)
                            tm2 = c.get_team(bid2)
                            if tm2 != my_team and et2 in (
                                EntityType.ROAD,
                                EntityType.CONVEYOR,
                                EntityType.SPLITTER,
                                EntityType.BRIDGE,
                                EntityType.ARMOURED_CONVEYOR,
                            ):
                                penalty_enemy_transport = 3
                            else:
                                penalty_on_building = 1
                        except GameError:
                            penalty_on_building = 1

                    escape_count = 0
                    for sdx, sdy in CARDINAL_DELTAS:
                        nx = lx + sdx
                        ny = ly + sdy
                        if not (0 <= nx < self.map_w and 0 <= ny < self.map_h):
                            continue
                        if max(abs(nx - cx), abs(ny - cy)) < 4:
                            continue
                        np = Position(nx, ny)
                        try:
                            env = c.get_tile_env(np)
                            if str(env).endswith("WALL") or getattr(env, "name", "") == "WALL":
                                continue
                        except Exception:
                            pass
                        try:
                            nbid = c.get_tile_building_id(np)
                        except GameError:
                            nbid = None
                        if nbid is None or nbid == 0:
                            escape_count += 1
                            continue
                        try:
                            net = c.get_entity_type(nbid)
                            if net in (
                                EntityType.ROAD,
                                EntityType.CONVEYOR,
                                EntityType.SPLITTER,
                                EntityType.BRIDGE,
                                EntityType.ARMOURED_CONVEYOR,
                            ):
                                escape_count += 1
                        except GameError:
                            pass

                    if escape_count == 0:
                        penalty_dead_pocket = 2
                    elif escape_count == 1:
                        penalty_dead_pocket = 1

                    return (
                        penalty_enemy_transport,
                        penalty_dead_pocket,
                        penalty_on_building,
                    )

                best_target = None
                best_score = None
                scan_r = int(LAUNCHER_VISION_RADIUS_SQ ** 0.5)
                for ldy in range(-scan_r, scan_r + 1):
                    for ldx in range(-scan_r, scan_r + 1):
                        lx, ly = pos.x + ldx, pos.y + ldy
                        if ldx * ldx + ldy * ldy > LAUNCHER_VISION_RADIUS_SQ:
                            continue
                        if max(abs(lx - cx), abs(ly - cy)) < 4:
                            continue
                        if not (0 <= lx < self.map_w and 0 <= ly < self.map_h):
                            continue
                        lp = Position(lx, ly)
                        try:
                            if not c.can_launch(bot_pos, lp):
                                continue
                        except GameError:
                            continue

                        penalties = _landing_penalty(lx, ly)
                        score = penalties + (
                            abs(lx - cx) + abs(ly - cy),
                            abs(lx - pos.x) + abs(ly - pos.y),
                            lx,
                            ly,
                        )
                        if best_score is None or score < best_score:
                            best_score = score
                            best_target = lp

                if best_target is not None:
                    c.launch(bot_pos, best_target)
                    log_event(
                        rnd,
                        uid,
                        "launcher",
                        f"({pos.x},{pos.y})",
                        "launched_bot",
                        bx=bot_pos.x,
                        by=bot_pos.y,
                        tx=best_target.x,
                        ty=best_target.y,
                    )
                    return
            except GameError:
                continue