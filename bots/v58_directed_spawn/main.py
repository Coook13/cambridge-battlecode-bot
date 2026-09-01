import heapq
import sys
from cambc import Controller, Direction, EntityType, Environment, Position

CARDINALS = [Direction.NORTH, Direction.EAST, Direction.SOUTH, Direction.WEST]
ALL_DIRS = [d for d in Direction if d != Direction.CENTRE]

_OPP = {
    Direction.NORTH: Direction.SOUTH, Direction.SOUTH: Direction.NORTH,
    Direction.EAST: Direction.WEST, Direction.WEST: Direction.EAST,
    Direction.NORTHEAST: Direction.SOUTHWEST, Direction.SOUTHWEST: Direction.NORTHEAST,
    Direction.NORTHWEST: Direction.SOUTHEAST, Direction.SOUTHEAST: Direction.NORTHWEST,
}
_RIGHT = {
    Direction.NORTH: Direction.EAST, Direction.EAST: Direction.SOUTH,
    Direction.SOUTH: Direction.WEST, Direction.WEST: Direction.NORTH,
}
_LEFT = {
    Direction.NORTH: Direction.WEST, Direction.WEST: Direction.SOUTH,
    Direction.SOUTH: Direction.EAST, Direction.EAST: Direction.NORTH,
}

_DIR_VECTORS = {
    Direction.NORTH: (0, -1),
    Direction.SOUTH: (0, 1),
    Direction.EAST: (1, 0),
    Direction.WEST: (-1, 0),
    Direction.NORTHEAST: (1, -1),
    Direction.NORTHWEST: (-1, -1),
    Direction.SOUTHEAST: (1, 1),
    Direction.SOUTHWEST: (-1, 1),
}

WALKABLE_TYPES = {
    EntityType.ROAD,
    EntityType.CONVEYOR,
    EntityType.SPLITTER,
    EntityType.BRIDGE,
    EntityType.ARMOURED_CONVEYOR,
}

BREAKABLE_WALKABLE_TYPES = {
    EntityType.ROAD,
    EntityType.CONVEYOR,
    EntityType.SPLITTER,
    EntityType.BRIDGE,
}

BREAK_TARGET_STALE_LIMIT = 4
BREAK_STICKY_BONUS = 320
BREAK_RECENT_BROKEN_LIMIT = 8
BREAK_RECENT_BROKEN_ROUNDS = 24
FRIENDLY_BREAK_HARD_PENALTY = 100000
FRIENDLY_BREAK_SOFT_PENALTY = 2400
BREAK_FEED_PRIORITY_BONUS = 700
BREAK_INFRA_PRIORITY_BONUS = 260
BREAK_ROAD_PRIORITY_PENALTY = 60
BREAK_BRIDGE_PRIORITY_BONUS = 950
BREAK_CONVEYOR_PRIORITY_BONUS = 700
BREAK_SPLITTER_PRIORITY_BONUS = 620
BREAK_HOT_TILE_BONUS = 140

# ---- economy phase ----
WAVE_SIZE = 4
WAVE_INTERVAL = 150
SCOUT_WAVE_SLOT = 1

# ---- assault phase ----
RUSH_START = 30
RUSH_BLOCK = 30
RUSH_BOTS_PER_BLOCK = 12
BONUS_MINERS_PER_BLOCK = 1
RESOURCE_GATE = 80

TURRET_PLANT_DIST2 = 18
INFRA_DESTROY_DIST2 = 10
ROAM_DEMOLISH_START = 220

ORBIT_OFFSETS = [
    (3, 0), (-3, 0), (0, 3), (0, -3),
    (2, 2), (-2, 2), (2, -2), (-2, -2),
]

ASSAULT_STALL_TRIGGER = 3
ASSAULT_RECENT_MEMORY = 6
ASSAULT_REPEAT_PENALTY = 45
ASSAULT_DETOUR_STEPS = 8
ASSAULT_DETOUR_SIDE_LEN = 6
ASSAULT_DETOUR_FWD_LEN = 2

PATH_PLAN_ATTACK_DIST2 = 25
PATH_PLAN_OUTWARD_DIST2 = 9
PATH_PLAN_SCOUT_DIST2 = 16
PATH_PLAN_STALL_TRIGGER = 1
ROUTE_BRIDGE_LOOKBACK = 10
ROUTE_BRIDGE_MIN_SKIP = 2
HOME_TURRET_EXCLUSION_DIST2 = 49

# ---- economy tuning ----
SABOTAGE_START_1 = 280
SABOTAGE_START_2 = 380
MIN_HARVESTS_BEFORE_SABOTAGE = 2
SABOTAGE_SAFE_CORE_DIST2 = 16

# ---- scout ----
SCOUT_ARRIVE_DIST2 = 16
SCOUT_LOITER_ROUNDS = 10

# ---- markers ----
MARKER_ENEMY_CORE_FLAG = 1 << 30
MARKER_GO_CENTRE_FLAG = 1 << 29
MARKER_STAY_MINER_FLAG = 1 << 28


def _encode_enemy_pos(x, y):
    return MARKER_ENEMY_CORE_FLAG | ((x & 0xFFF) << 12) | (y & 0xFFF)


def _decode_pos(v):
    return ((v >> 12) & 0xFFF, v & 0xFFF)


class Player:
    def __init__(self):
        # Shared unit lifecycle / identity.
        self.num_spawned = 0
        self.phase = 'outward'
        self.core_pos = None
        self.harvest_count = 0

        # Economy route-back state.
        self.path = []
        self.path_index_by_xy = {}
        self.route_idx = 0
        self.route_pivot = None
        self.route_skip_until_idx = None
        self.pending_bridge_from = None
        self.pending_bridge_to = None
        self.pending_bridge_anchor = None
        self.pending_bridge_stage = None

        # Shared short-range path planner used by both economy and assault code.
        self.path_plan_target = None
        self.path_plan_tiles = []
        self.path_plan_index = 0
        self.path_plan_kind = None

        # Miner exploration state.
        self.spoke_dir = None
        self.last_dir = None
        self.ore_target = None

        # Generic stuck / escape fallback.
        self.stuck_turns = 0
        self.escape_dir = None
        self.escape_steps = 0
        self.escape_return_phase = 'outward'

        self.role = 'miner'

        # Sabotage-only state.
        self.sabotage_dir = None
        self.sabotage_path = []
        self.sabotage_visited = set()
        self.sabotage_phase = 'travel'

        # Enemy-core knowledge shared by core spawning, markers, and assaulters.
        self.enemy_core_pos = None
        self.enemy_candidates = []
        self.enemy_candidate_idx = 0
        self.enemy_bad_targets = set()

        self.enemy_core_guess = None
        self.enemy_symmetry = None

        # Scout / economy role flags.
        self.is_scout = False
        self.scout_done = False
        self.stay_miner = False

        # Core spawn bookkeeping.
        self.bonus_miners_spawned = 0

        # Assault role state.
        self.assault_phase = 'travel'
        self.is_bomber = False
        self.plant_gun_pos = None
        self.plant_gun_facing = None
        self.plant_done_round = -1
        self.assault_fail_turns = 0
        self.assault_recent_positions = []
        self.assault_stall_turns = 0
        self.assault_detour_target = None
        self.assault_detour_steps = 0
        self.assault_detour_flip = 0

        # Sticky break-target memory for bomber pressure.
        self.break_target = None
        self.break_target_kind = None
        self.break_target_stale_turns = 0
        self.recent_broken_tiles = []
        self.break_coord_slot = 1

    # -- Entry dispatch -------------------------------------------------
    def run(self, c: Controller) -> None:
        try:
            etype = c.get_entity_type()
            if etype == EntityType.CORE:
                self._core(c)
            elif etype == EntityType.BUILDER_BOT:
                self._builder(c)
            elif etype in (EntityType.GUNNER, EntityType.SENTINEL, EntityType.BREACH):
                self._turret_fire(c)
        except Exception as e:
            print(
                f"ERR round={c.get_current_round()} id={c.get_id()} role={self.role} phase={self.phase} {e}",
                file=sys.stderr,
            )

    # -- Core spawning --------------------------------------------------
    def _core(self, c):
        if self.core_pos is None:
            self.core_pos = c.get_position()

        self._bootstrap_enemy_core_guess(c)

        if self.enemy_core_pos is None:
            self._try_read_enemy_core_marker(c)

        if c.get_action_cooldown() > 0:
            return

        round_num = c.get_current_round()
        pos = c.get_position()

        target = self._target_spawn_count(round_num)
        if self.num_spawned >= target:
            return

        if round_num >= RUSH_START:
            ti, ax = c.get_global_resources()
            if ti < RESOURCE_GATE and ax < RESOURCE_GATE:
                return

            bonus_target = ((round_num - RUSH_START) // RUSH_BLOCK) + 1
            if self.bonus_miners_spawned < bonus_target:
                for d in ALL_DIRS:
                    sp = pos.add(d)
                    try:
                        if c.can_spawn(sp):
                            c.spawn_builder(sp)
                            if c.can_place_marker(sp):
                                c.place_marker(sp, MARKER_STAY_MINER_FLAG)
                            self.bonus_miners_spawned += 1
                            self.num_spawned += 1
                            return
                    except Exception:
                        pass

            enemy_t = self._current_enemy_target(c)
            ex, ey = enemy_t.x, enemy_t.y
            best_sp = None
            best_d2 = 10**18
            for d in ALL_DIRS:
                sp = pos.add(d)
                try:
                    if c.can_spawn(sp):
                        dx = sp.x - ex
                        dy = sp.y - ey
                        d2 = dx * dx + dy * dy
                        if d2 < best_d2:
                            best_d2 = d2
                            best_sp = sp
                except Exception:
                    pass
            if best_sp is not None:
                c.spawn_builder(best_sp)
                self.num_spawned += 1
            return

        # v58_directed_spawn: pick spawn position closest to enemy core
        # to fix seat A asymmetry in xuanming_v3. Original code used fixed
        # cardinal preference (N/E/S/W cycle) regardless of enemy direction,
        # so seat A spawns away from enemy in pre-rush phase, costing ~30
        # rounds of bot travel time vs seat B who happens to spawn toward
        # enemy. Self-play showed seat A gets crushed 4/4.
        slot_in_wave = self.num_spawned % WAVE_SIZE
        enemy_t = self._current_enemy_target(c)
        ex, ey = enemy_t.x, enemy_t.y

        # Score each candidate spawn tile by distance² to enemy.
        candidates = []
        for d in ALL_DIRS:
            sp = pos.add(d)
            try:
                if c.can_spawn(sp):
                    dx = sp.x - ex
                    dy = sp.y - ey
                    d2 = dx * dx + dy * dy
                    candidates.append((d2, d, sp))
            except Exception:
                pass
        candidates.sort(key=lambda t: t[0])

        for d2, d, sp in candidates:
            try:
                c.spawn_builder(sp)
                if slot_in_wave == SCOUT_WAVE_SLOT and c.can_place_marker(sp):
                    c.place_marker(sp, MARKER_GO_CENTRE_FLAG)
                self.num_spawned += 1
                return
            except Exception:
                pass

    def _target_spawn_count(self, round_num):
        target = ((round_num // WAVE_INTERVAL) + 1) * WAVE_SIZE
        if round_num >= SABOTAGE_START_1:
            target += 1
        if round_num >= SABOTAGE_START_2:
            target += 1
        if round_num >= RUSH_START:
            blocks = ((round_num - RUSH_START) // RUSH_BLOCK) + 1
            target += blocks * RUSH_BOTS_PER_BLOCK
        return target

    # -- Builder dispatch -----------------------------------------------
    def _builder(self, c):
        pos = c.get_position()

        if self.core_pos is None:
            for bid in c.get_nearby_buildings():
                try:
                    if c.get_entity_type(bid) == EntityType.CORE and c.get_team(bid) == c.get_team():
                        self.core_pos = c.get_position(bid)
                        break
                except Exception:
                    pass
        if self.core_pos is None:
            return

        self._bootstrap_enemy_core_guess(c)

        if self.spoke_dir is None:
            self.spoke_dir = CARDINALS[c.get_id() % 4]
            self.break_coord_slot = c.get_id() % 3
            v = self._get_marker_on_tile(c, pos)
            if v is not None:
                if v & MARKER_GO_CENTRE_FLAG:
                    self.is_scout = True
                if v & MARKER_STAY_MINER_FLAG:
                    self.stay_miner = True
            if c.get_current_round() < RUSH_START:
                self.stay_miner = True
            self._ensure_enemy_candidates(c)

        if self._attack_enemy_tile_if_profitable(c, pos, hold_position=False):
            return

        if self.enemy_core_pos is None:
            self._try_read_enemy_core_marker(c)
        self._try_report_enemy_core(c, pos)

        # Attack handoff: after the rush timer, non-stay-miners stop using the
        # economy pipeline and switch into the assaulter pipeline below.
        if (
            c.get_current_round() >= RUSH_START
            and self.role not in ('assaulter', 'saboteur')
            and not self.stay_miner
        ):
            self._activate_assaulter(c)

        if self.is_scout and not self.scout_done and self.role != 'assaulter':
            self._do_scout(c, pos)
            return

        if self.phase == 'outward':
            self._record_stack_path(self.path, pos)

        if self.phase == 'escape':
            self._do_escape(c, pos)
            return
        if self.phase == 'outward':
            self._do_outward(c, pos)
        elif self.phase == 'route':
            self._do_route(c, pos)
        elif self.phase == 'sabotage':
            self._do_sabotage(c, pos)
        elif self.phase == 'assault':
            self._do_assault(c, pos)

    # -- Attack: turret autofire ----------------------------------------
    def _turret_fire(self, c):
        if c.get_action_cooldown() > 0:
            return

        if self.enemy_core_pos is None:
            self._try_read_enemy_core_marker(c)
        if self.enemy_core_pos is None:
            self._scan_for_real_enemy_core(c)

        if self.enemy_core_pos is not None:
            try:
                my_pos = c.get_position()
            except Exception:
                my_pos = None
            for tp in self._enemy_core_tiles_sorted(self.enemy_core_pos, my_pos):
                try:
                    if c.can_fire(tp):
                        c.fire(tp)
                        return
                except Exception:
                    pass

        try:
            if c.get_entity_type() == EntityType.GUNNER:
                t = c.get_gunner_target()
                if t is not None and self._gunner_target_is_hostile(c, t) and c.can_fire(t):
                    c.fire(t)
                    return
        except Exception:
            pass

    # -- Attack entrypoints ---------------------------------------------
    # 1. _activate_assaulter() resets mining state and picks bomber vs planter.
    # 2. _do_assault() routes into bomber or turret-planter behavior.
    # 3. Attack-support helpers below decide hijack sites, enemy-core guesses,
    #    and sticky break targets.
    def _activate_assaulter(self, c):
        # Clear miner-only state before handing the builder over to attack.
        self.role = 'assaulter'
        self.phase = 'assault'
        self.ore_target = None
        self.path = []
        self.path_index_by_xy = {}
        self.route_idx = 0
        self.route_pivot = None
        self.route_skip_until_idx = None
        self.pending_bridge_from = None
        self.pending_bridge_to = None
        self.pending_bridge_anchor = None
        self.pending_bridge_stage = None
        self.last_dir = None
        self.stuck_turns = 0
        self.escape_dir = None
        self.escape_steps = 0
        self._clear_path_plan()

        self.assault_phase = 'travel'
        self.is_bomber = (c.get_id() % 5 == 0)
        self.plant_gun_pos = None
        self.plant_gun_facing = None
        self.plant_done_round = -1
        self.assault_fail_turns = 0
        self.assault_recent_positions = []
        self.assault_stall_turns = 0
        self.assault_detour_target = None
        self.assault_detour_steps = 0
        self.assault_detour_flip = 0
        self.break_coord_slot = c.get_id() % 3
        self._clear_break_target()

    def _do_assault(self, c, pos):
        # Assaulters share target discovery, then split into two roles:
        # bombers break tiles directly, while turret-planters look for hijacks.
        if self.enemy_core_pos is None:
            self._try_read_enemy_core_marker(c)
        self._try_report_enemy_core(c, pos)
        self._maybe_advance_enemy_candidate(c, pos)

        target = self._current_enemy_target(c)
        real_target = self.enemy_core_pos if self.enemy_core_pos is not None else target

        if self.is_bomber:
            self._do_bomber(c, pos, real_target)
            return
        self._do_turret_assault(c, pos, real_target)

    def _do_bomber(self, c, pos, target):
        # Bomber priority:
        # 1. keep breaking the enemy tile we already stand on
        # 2. commit to a sticky nearby break target
        # 3. chase a live enemy feed tile
        # 4. roam onto visible infra later in the game
        # 5. fall back to direct core pressure
        if self.enemy_core_pos is None:
            self._scan_for_real_enemy_core(c)
        core_target = self.enemy_core_pos if self.enemy_core_pos is not None else target

        if self._attack_enemy_tile_if_profitable(c, pos):
            return

        break_target = self._select_break_target(c, pos, core_target)
        if break_target is not None:
            if c.get_move_cooldown() == 0:
                self._assault_move(c, pos, break_target, prefer_safe_tiles=False)
            return

        hijack = self._find_nearest_live_enemy_feed_tile(c, pos, core_target)
        if hijack is not None:
            if c.get_move_cooldown() == 0:
                self._assault_move(c, pos, hijack, prefer_safe_tiles=False)
                return

        if c.get_current_round() >= ROAM_DEMOLISH_START:
            infra = self._nearest_enemy_infra(c, pos)
            if infra is not None and c.get_move_cooldown() == 0:
                self._assault_move(c, pos, infra, prefer_safe_tiles=False)
                return

        if c.get_move_cooldown() == 0:
            self._assault_move(c, pos, core_target, prefer_safe_tiles=False)

    def _do_turret_assault(self, c, pos, target):
        # Turret-planters first try to convert enemy feed into a forward gunner.
        # If no safe plant exists yet, they pressure the same enemy space until
        # a good hijack tile opens up.
        if self.enemy_core_pos is None:
            self._scan_for_real_enemy_core(c)
        core_target = self.enemy_core_pos if self.enemy_core_pos is not None else target

        if self.assault_phase == 'done':
            if c.get_current_round() > self.plant_done_round:
                self.is_bomber = True
                self._do_bomber(c, pos, core_target)
            return

        if c.get_action_cooldown() == 0:
            hijack_build = self._find_empty_hijack_build(c, pos, core_target)
            if hijack_build is not None:
                gun_pos, gun_dir = hijack_build
                # v53_xuan_plus: try sentinel first (#1 team replay had 218
                # forward-deployed sentinels — longer range than gunner).
                # Fall back to gunner if Ti tight or sentinel can't build.
                try:
                    if c.can_build_sentinel(gun_pos, gun_dir):
                        c.build_sentinel(gun_pos, gun_dir)
                        self.plant_gun_pos = gun_pos
                        self.plant_gun_facing = gun_dir
                        self.plant_done_round = c.get_current_round()
                        self.assault_phase = 'done'
                        return
                except Exception:
                    pass
                try:
                    c.build_gunner(gun_pos, gun_dir)
                    self.plant_gun_pos = gun_pos
                    self.plant_gun_facing = gun_dir
                    self.plant_done_round = c.get_current_round()
                    self.assault_phase = 'done'
                    return
                except Exception:
                    pass

        hijack = self._find_nearest_live_enemy_feed_tile(c, pos, core_target)
        if hijack is not None and c.get_move_cooldown() == 0:
            self._assault_move(c, pos, hijack, prefer_safe_tiles=False)
            return

        if self.enemy_core_pos is not None and pos.distance_squared(self.enemy_core_pos) <= TURRET_PLANT_DIST2:
            self.assault_fail_turns += 1
            if self.assault_fail_turns >= 6:
                if self._attack_enemy_tile_if_profitable(c, pos):
                    return
                break_target = self._select_break_target(c, pos, core_target)
                if break_target is not None:
                    if c.get_move_cooldown() == 0:
                        self._assault_move(c, pos, break_target, prefer_safe_tiles=False)
                    return
            if c.get_move_cooldown() == 0:
                orbit = self._assault_orbit_target(c, core_target)
                self._assault_move(c, pos, orbit, prefer_safe_tiles=False)
            return

        if c.get_move_cooldown() == 0:
            self._assault_move(c, pos, core_target, prefer_safe_tiles=False)

    # -- Attack support: hijack / turret placement helpers ---------------
    def _incoming_feed_dirs_for_tile(self, c, tile_pos, team_mode='any'):
        dirs = []
        for d in CARDINALS:
            nb = tile_pos.add(d)
            if not self._in_bounds(c, nb):
                continue
            try:
                bid = c.get_tile_building_id(nb)
            except Exception:
                bid = None
            if bid is None:
                continue

            try:
                team = c.get_team(bid)
                etype = c.get_entity_type(bid)
            except Exception:
                continue

            if team_mode == 'ally' and team != c.get_team():
                continue
            if team_mode == 'enemy' and team == c.get_team():
                continue

            if etype in (EntityType.CONVEYOR, EntityType.ARMOURED_CONVEYOR):
                try:
                    out_dir = c.get_direction(bid)
                except Exception:
                    continue
                dest = nb.add(out_dir)
                if dest.x == tile_pos.x and dest.y == tile_pos.y:
                    dirs.append(d)
            elif etype == EntityType.SPLITTER:
                try:
                    out_dir = c.get_direction(bid)
                except Exception:
                    continue
                for out in (out_dir, _LEFT.get(out_dir), _RIGHT.get(out_dir)):
                    if out is None:
                        continue
                    dest = nb.add(out)
                    if dest.x == tile_pos.x and dest.y == tile_pos.y:
                        dirs.append(d)
                        break
            elif etype in (EntityType.HARVESTER, EntityType.FOUNDRY):
                dirs.append(d)
        return dirs

    def _bridge_feeders_for_tile(self, c, tile_pos, team_mode='any'):
        feeders = []
        try:
            bids = c.get_nearby_buildings()
        except Exception:
            return feeders

        for bid in bids:
            try:
                team = c.get_team(bid)
                etype = c.get_entity_type(bid)
            except Exception:
                continue

            if etype != EntityType.BRIDGE:
                continue
            if team_mode == 'ally' and team != c.get_team():
                continue
            if team_mode == 'enemy' and team == c.get_team():
                continue

            try:
                target = c.get_bridge_target(bid)
            except Exception:
                continue

            if target.x == tile_pos.x and target.y == tile_pos.y:
                try:
                    feeders.append(c.get_position(bid))
                except Exception:
                    pass

        return feeders

    def _gunner_target_is_hostile(self, c, target_pos):
        try:
            uid = c.get_tile_builder_bot_id(target_pos)
        except Exception:
            uid = None
        if uid is not None:
            try:
                return c.get_team(uid) != c.get_team()
            except Exception:
                return False

        try:
            bid = c.get_tile_building_id(target_pos)
        except Exception:
            bid = None
        if bid is None:
            return False

        try:
            etype = c.get_entity_type(bid)
            team = c.get_team(bid)
        except Exception:
            return False

        return etype != EntityType.MARKER and team != c.get_team()

    def _offensive_turret_site_safe(self, c, tile_pos, core_target):
        if self.core_pos is None:
            return False
        if tile_pos.distance_squared(self.core_pos) <= HOME_TURRET_EXCLUSION_DIST2:
            return False
        if core_target is None:
            return True
        return tile_pos.distance_squared(core_target) < tile_pos.distance_squared(self.core_pos)

    def _gunner_line_has_friendly_first_blocker(self, c, gun_pos, gun_dir):
        try:
            tiles = c.get_attackable_tiles_from(gun_pos, gun_dir, EntityType.GUNNER)
        except Exception:
            return False

        tiles.sort(key=lambda tp: gun_pos.distance_squared(tp))
        for tp in tiles:
            try:
                uid = c.get_tile_builder_bot_id(tp)
            except Exception:
                uid = None
            if uid is not None:
                try:
                    return c.get_team(uid) == c.get_team()
                except Exception:
                    return False

            try:
                bid = c.get_tile_building_id(tp)
            except Exception:
                bid = None
            if bid is not None:
                try:
                    etype = c.get_entity_type(bid)
                    team = c.get_team(bid)
                except Exception:
                    return False
                if etype == EntityType.MARKER:
                    continue
                return team == c.get_team()

            try:
                if c.get_tile_env(tp) == Environment.WALL:
                    return False
            except Exception:
                return False

        return False

    def _tile_is_live_enemy_feed_tile(self, c, tile_pos):
        try:
            bid = c.get_tile_building_id(tile_pos)
        except Exception:
            return False
        if bid is None:
            return False

        try:
            team = c.get_team(bid)
            etype = c.get_entity_type(bid)
        except Exception:
            return False

        if team == c.get_team():
            return False
        if etype not in (EntityType.CONVEYOR, EntityType.ARMOURED_CONVEYOR, EntityType.SPLITTER, EntityType.BRIDGE):
            return False

        if self._incoming_feed_dirs_for_tile(c, tile_pos, team_mode='enemy'):
            return True
        if self._bridge_feeders_for_tile(c, tile_pos, team_mode='enemy'):
            return True

        try:
            return c.get_stored_resource(bid) is not None
        except Exception:
            return False

    def _find_nearest_live_enemy_feed_tile(self, c, pos, core_pos):
        best = None
        best_score = 10**18

        try:
            bids = c.get_nearby_buildings()
        except Exception:
            return None

        for bid in bids:
            try:
                etype = c.get_entity_type(bid)
                team = c.get_team(bid)
            except Exception:
                continue

            if team == c.get_team():
                continue
            if etype not in (EntityType.CONVEYOR, EntityType.ARMOURED_CONVEYOR, EntityType.SPLITTER, EntityType.BRIDGE):
                continue

            try:
                bp = c.get_position(bid)
            except Exception:
                continue

            if not self._tile_is_live_enemy_feed_tile(c, bp):
                continue

            score = pos.distance_squared(bp)
            if core_pos is not None:
                score += bp.distance_squared(core_pos)

            try:
                if c.get_stored_resource(bid) is not None:
                    score -= 120
            except Exception:
                pass

            if etype == EntityType.BRIDGE:
                score -= 160
            elif etype in (EntityType.CONVEYOR, EntityType.ARMOURED_CONVEYOR):
                score -= 70
            elif etype == EntityType.SPLITTER:
                score -= 50

            if score < best_score:
                best_score = score
                best = bp

        return best

    def _best_hijack_gun_dir(self, gun_pos, core_pos, feed_dirs):
        dx = core_pos.x - gun_pos.x
        dy = core_pos.y - gun_pos.y

        best_dir = None
        best_score = -10**18

        for d, (vx, vy) in _DIR_VECTORS.items():
            if d in CARDINALS:
                ok = False
                for fd in feed_dirs:
                    if fd != d:
                        ok = True
                        break
                if not ok:
                    continue

            score = dx * vx + dy * vy
            if d not in CARDINALS:
                score += 2

            if score > best_score:
                best_score = score
                best_dir = d

        return best_dir

    def _find_empty_hijack_build(self, c, pos, core_pos):
        best = None
        best_score = 10**18

        try:
            tiles = c.get_nearby_tiles(2)
        except Exception:
            return None

        for tp in tiles:
            if pos.distance_squared(tp) > 2:
                continue
            if not self._in_bounds(c, tp) or self._is_wall(c, tp):
                continue

            try:
                if c.get_tile_building_id(tp) is not None:
                    continue
            except Exception:
                continue

            if not self._offensive_turret_site_safe(c, tp, core_pos):
                continue

            feed_dirs = self._incoming_feed_dirs_for_tile(c, tp, team_mode='enemy')
            if not feed_dirs:
                continue

            gun_dir = self._best_hijack_gun_dir(tp, core_pos, feed_dirs)
            if gun_dir is None:
                continue

            if self._gunner_line_has_friendly_first_blocker(c, tp, gun_dir):
                continue

            try:
                if not c.can_build_gunner(tp, gun_dir):
                    continue
            except Exception:
                continue

            score = tp.distance_squared(core_pos)

            hot = False
            for d in feed_dirs:
                nb = tp.add(d)
                try:
                    bid = c.get_tile_building_id(nb)
                    if bid is not None and c.get_stored_resource(bid) is not None:
                        hot = True
                        break
                except Exception:
                    pass
            if hot:
                score -= 150

            if score < best_score:
                best_score = score
                best = (tp, gun_dir)

        return best

    def _nearest_enemy_infra(self, c, pos):
        best_pos = None
        best_score = 10**18
        try:
            bids = c.get_nearby_buildings()
        except Exception:
            return None

        for bid in bids:
            try:
                team = c.get_team(bid)
                etype = c.get_entity_type(bid)
            except Exception:
                continue
            if team == c.get_team() or etype not in (
                EntityType.ROAD,
                EntityType.CONVEYOR,
                EntityType.SPLITTER,
                EntityType.BRIDGE,
            ):
                continue
            try:
                tp = c.get_position(bid)
            except Exception:
                continue
            score = pos.distance_squared(tp)
            if self.enemy_core_pos is not None:
                score += tp.distance_squared(self.enemy_core_pos)
            if score < best_score:
                best_score = score
                best_pos = tp
        return best_pos

    def _scan_for_real_enemy_core(self, c):
        try:
            for bid in c.get_nearby_buildings():
                if c.get_entity_type(bid) == EntityType.CORE and c.get_team(bid) != c.get_team():
                    self.enemy_core_pos = c.get_position(bid)
                    self.enemy_core_guess = self.enemy_core_pos
                    return
        except Exception:
            pass

    def _enemy_core_tiles_sorted(self, core_pos, ref_pos=None):
        tiles = []
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                tp = Position(core_pos.x + dx, core_pos.y + dy)
                tiles.append(tp)
        if ref_pos is not None:
            tiles.sort(
                key=lambda p: (p.x - ref_pos.x) * (p.x - ref_pos.x)
                + (p.y - ref_pos.y) * (p.y - ref_pos.y)
            )
        return tiles

    def _assault_orbit_target(self, c, core_pos):
        ox, oy = ORBIT_OFFSETS[(c.get_id() + self.assault_fail_turns) % len(ORBIT_OFFSETS)]
        x = max(0, min(c.get_map_width() - 1, core_pos.x + ox))
        y = max(0, min(c.get_map_height() - 1, core_pos.y + oy))
        return Position(x, y)

    # -- Attack support: marker sharing ---------------------------------
    # Attack target discovery starts here. Markers and symmetry give attackers a
    # shared guess for where to march even before the real enemy core is seen.
    def _get_marker_on_tile(self, c, pos):
        try:
            bid = c.get_tile_building_id(pos)
            if bid is None:
                return None
            if c.get_entity_type(bid) != EntityType.MARKER:
                return None
            return c.get_marker_value(bid)
        except Exception:
            return None

    def _try_report_enemy_core(self, c, pos):
        if self.enemy_core_pos is None:
            self._scan_for_real_enemy_core(c)
        if self.enemy_core_pos is not None:
            val = _encode_enemy_pos(self.enemy_core_pos.x, self.enemy_core_pos.y)
            for d in CARDINALS:
                adj = pos.add(d)
                try:
                    if c.can_place_marker(adj):
                        c.place_marker(adj, val)
                        break
                except Exception:
                    pass

    def _try_read_enemy_core_marker(self, c):
        if self.enemy_core_pos is not None:
            return
        try:
            for tp in c.get_nearby_tiles():
                v = self._get_marker_on_tile(c, tp)
                if v is not None and (v & MARKER_ENEMY_CORE_FLAG):
                    ex, ey = _decode_pos(v)
                    self.enemy_core_pos = Position(ex, ey)
                    self.enemy_core_guess = self.enemy_core_pos
                    return
        except Exception:
            pass

    # -- Attack support: symmetry bootstrap ------------------------------
    def _detect_enemy_symmetry(self, c):
        if self.core_pos is None:
            return None

        w = c.get_map_width()
        h = c.get_map_height()

        mid_x = (w - 1) / 2.0
        mid_y = (h - 1) / 2.0

        on_mid_x = abs(self.core_pos.x - mid_x) < 0.51
        on_mid_y = abs(self.core_pos.y - mid_y) < 0.51

        if on_mid_x and not on_mid_y:
            return 'HORIZONTAL'
        if on_mid_y and not on_mid_x:
            return 'VERTICAL'
        return 'ROTATIONAL'

    def _mirror_position_by_symmetry(self, pos, symmetry, map_w, map_h):
        if symmetry == 'VERTICAL':
            return Position(map_w - 1 - pos.x, pos.y)
        if symmetry == 'HORIZONTAL':
            return Position(pos.x, map_h - 1 - pos.y)
        return Position(map_w - 1 - pos.x, map_h - 1 - pos.y)

    def _bootstrap_enemy_core_guess(self, c):
        if self.core_pos is None:
            return

        if self.enemy_core_pos is not None:
            self.enemy_core_guess = self.enemy_core_pos
            return

        if self.enemy_core_guess is not None:
            return

        symmetry = self._detect_enemy_symmetry(c)
        if symmetry is None:
            return

        self.enemy_symmetry = symmetry
        self.enemy_core_guess = self._mirror_position_by_symmetry(
            self.core_pos,
            symmetry,
            c.get_map_width(),
            c.get_map_height(),
        )

    # -- Attack support: enemy-core target selection ---------------------
    # If the real core is still hidden, assaulters rotate through mirrored core
    # candidates until one is confirmed or ruled out.
    def _ensure_enemy_candidates(self, c):
        if self.enemy_candidates or self.core_pos is None:
            return
        w = c.get_map_width()
        h = c.get_map_height()
        x, y = self.core_pos.x, self.core_pos.y

        raw = [
            Position(w - 1 - x, h - 1 - y),
            Position(w - 1 - x, y),
            Position(x, h - 1 - y),
        ]

        seen = {(x, y)}
        uniq = []
        for p in raw:
            key = (p.x, p.y)
            if key not in seen:
                seen.add(key)
                uniq.append(p)

        self.enemy_candidates = uniq or [Position(w // 2, h // 2)]
        self.enemy_candidate_idx = c.get_id() % len(self.enemy_candidates)

    def _current_enemy_target(self, c):
        if self.enemy_core_pos is not None:
            return self.enemy_core_pos

        self._bootstrap_enemy_core_guess(c)
        if (
            self.enemy_core_guess is not None
            and (self.enemy_core_guess.x, self.enemy_core_guess.y) not in self.enemy_bad_targets
        ):
            return self.enemy_core_guess

        self._ensure_enemy_candidates(c)
        if not self.enemy_candidates:
            return Position(c.get_map_width() // 2, c.get_map_height() // 2)

        n = len(self.enemy_candidates)
        for k in range(n):
            idx = (self.enemy_candidate_idx + k) % n
            p = self.enemy_candidates[idx]
            if (p.x, p.y) not in self.enemy_bad_targets:
                self.enemy_candidate_idx = idx
                return p

        self.enemy_bad_targets.clear()
        return self.enemy_candidates[self.enemy_candidate_idx % n]

    def _maybe_advance_enemy_candidate(self, c, pos):
        if self.enemy_core_pos is not None:
            self.enemy_core_guess = self.enemy_core_pos
            return

        self._bootstrap_enemy_core_guess(c)

        if (
            self.enemy_core_guess is not None
            and (self.enemy_core_guess.x, self.enemy_core_guess.y) not in self.enemy_bad_targets
        ):
            if pos.distance_squared(self.enemy_core_guess) <= 20:
                self._scan_for_real_enemy_core(c)
                if self.enemy_core_pos is not None:
                    self.enemy_core_guess = self.enemy_core_pos
            return

        self._ensure_enemy_candidates(c)
        if not self.enemy_candidates:
            return

        target = self._current_enemy_target(c)
        if pos.distance_squared(target) > 20:
            return

        self._scan_for_real_enemy_core(c)
        if self.enemy_core_pos is not None:
            return

        self.enemy_bad_targets.add((target.x, target.y))
        if self.enemy_candidates:
            self.enemy_candidate_idx = (self.enemy_candidate_idx + 1) % len(self.enemy_candidates)

    # -- Attack support: break-target selection --------------------------
    def _standing_on_enemy_walkable(self, c, pos):
        try:
            bid = c.get_tile_building_id(pos)
            if bid is None:
                return False
            return c.get_team(bid) != c.get_team() and c.get_entity_type(bid) in WALKABLE_TYPES
        except Exception:
            return False

    def _is_enemy_walkable_tile(self, c, pos):
        try:
            bid = c.get_tile_building_id(pos)
            if bid is None:
                return False
            return c.get_team(bid) != c.get_team() and c.get_entity_type(bid) in WALKABLE_TYPES
        except Exception:
            return False

    def _is_breakable_enemy_walkable_tile(self, c, pos, bid=None, team=None, etype=None):
        if bid is None:
            try:
                bid = c.get_tile_building_id(pos)
            except Exception:
                return False
        if bid is None:
            return False

        try:
            if team is None:
                team = c.get_team(bid)
            if etype is None:
                etype = c.get_entity_type(bid)
        except Exception:
            return False

        return team != c.get_team() and etype in BREAKABLE_WALKABLE_TYPES

    def _clear_break_target(self):
        self.break_target = None
        self.break_target_kind = None
        self.break_target_stale_turns = 0

    def _prune_recent_broken_tiles(self, c):
        current_round = c.get_current_round()
        kept = []
        for x, y, expire_round in self.recent_broken_tiles:
            if expire_round > current_round:
                kept.append((x, y, expire_round))
        if len(kept) > BREAK_RECENT_BROKEN_LIMIT:
            kept = kept[-BREAK_RECENT_BROKEN_LIMIT:]
        self.recent_broken_tiles = kept

    def _remember_recent_broken_tile(self, c, pos):
        self._prune_recent_broken_tiles(c)
        expire_round = c.get_current_round() + BREAK_RECENT_BROKEN_ROUNDS
        kept = []
        for x, y, old_expire in self.recent_broken_tiles:
            if x == pos.x and y == pos.y:
                continue
            kept.append((x, y, old_expire))
        kept.append((pos.x, pos.y, expire_round))
        if len(kept) > BREAK_RECENT_BROKEN_LIMIT:
            kept = kept[-BREAK_RECENT_BROKEN_LIMIT:]
        self.recent_broken_tiles = kept

    def _recently_broken_tile(self, c, pos):
        self._prune_recent_broken_tiles(c)
        for x, y, _ in self.recent_broken_tiles:
            if x == pos.x and y == pos.y:
                return True
        return False

    def _tile_is_live_break_feed_tile(self, c, tile_pos, bid=None, etype=None, team=None):
        if bid is None:
            try:
                bid = c.get_tile_building_id(tile_pos)
            except Exception:
                return False
        if bid is None:
            return False

        try:
            if team is None:
                team = c.get_team(bid)
            if etype is None:
                etype = c.get_entity_type(bid)
        except Exception:
            return False

        if team == c.get_team():
            return False
        if etype not in (EntityType.CONVEYOR, EntityType.SPLITTER, EntityType.BRIDGE):
            return False

        if self._incoming_feed_dirs_for_tile(c, tile_pos):
            return True

        try:
            return c.get_stored_resource(bid) is not None
        except Exception:
            return False

    def _break_target_kind_for(self, c, tile_pos, bid=None, etype=None, team=None):
        if not self._is_breakable_enemy_walkable_tile(c, tile_pos, bid, team, etype):
            return None
        if bid is None:
            try:
                bid = c.get_tile_building_id(tile_pos)
            except Exception:
                return None
        try:
            if team is None:
                team = c.get_team(bid)
            if etype is None:
                etype = c.get_entity_type(bid)
        except Exception:
            return None

        if etype == EntityType.BRIDGE:
            return 'bridge'
        if etype == EntityType.CONVEYOR:
            return 'conveyor'
        if etype == EntityType.SPLITTER:
            return 'splitter'
        return 'road'

    def _break_priority_score_adjust(self, c, bid, etype):
        score = 0
        if etype == EntityType.BRIDGE:
            score -= BREAK_BRIDGE_PRIORITY_BONUS
        elif etype == EntityType.CONVEYOR:
            score -= BREAK_CONVEYOR_PRIORITY_BONUS
        elif etype == EntityType.SPLITTER:
            score -= BREAK_SPLITTER_PRIORITY_BONUS
        else:
            score += BREAK_ROAD_PRIORITY_PENALTY

        try:
            if c.get_stored_resource(bid) is not None:
                score -= BREAK_HOT_TILE_BONUS
        except Exception:
            pass
        return score

    # Once a bomber is standing on a breakable enemy walkable, keep swinging
    # there
    def _attack_enemy_tile_if_profitable(self, c, pos, hold_position=True):
        if not self._is_breakable_enemy_walkable_tile(c, pos):
            if (
                self.break_target is not None
                and self.break_target.x == pos.x
                and self.break_target.y == pos.y
            ):
                self._remember_recent_broken_tile(c, pos)
                self._clear_break_target()
            return False

        if (
            self.break_target is None
            or self.break_target.x != pos.x
            or self.break_target.y != pos.y
        ):
            self.break_target = Position(pos.x, pos.y)
            self.break_target_kind = self._break_target_kind_for(c, pos)
        self.break_target_stale_turns = 0

        if c.get_action_cooldown() > 0:
            return hold_position

        try:
            if c.can_fire(pos):
                c.fire(pos)
                return True
        except Exception:
            pass
        return hold_position

    def _friendly_pressure_penalty(self, c, pos, target, friendly_positions=None):
        if friendly_positions is None:
            friendly_positions = []
            try:
                for uid in c.get_nearby_units():
                    try:
                        if uid == c.get_id():
                            continue
                        if c.get_team(uid) != c.get_team():
                            continue
                        if c.get_entity_type(uid) != EntityType.BUILDER_BOT:
                            continue
                        friendly_positions.append(c.get_position(uid))
                    except Exception:
                        continue
            except Exception:
                pass

        my_d2 = pos.distance_squared(target)
        penalty = 0
        for other_pos in friendly_positions:
            if other_pos.x == target.x and other_pos.y == target.y:
                return FRIENDLY_BREAK_HARD_PENALTY
            other_d2 = other_pos.distance_squared(target)
            if other_d2 <= 1 and other_d2 < my_d2:
                penalty = FRIENDLY_BREAK_SOFT_PENALTY
        return penalty

    def _break_lane_penalty(self, pos, target, core_target):
        if core_target is None:
            return 0

        if abs(core_target.x - pos.x) >= abs(core_target.y - pos.y):
            lateral = target.y - core_target.y
        else:
            lateral = target.x - core_target.x

        desired = (-2, 0, 2)[self.break_coord_slot % 3]
        return abs(lateral - desired) * 10

    # Sticky break scoring prefers live enemy logistics near the enemy core and
    # avoids target churn unless the current target goes stale.
    def _select_break_target(self, c, pos, core_target):
        self._prune_recent_broken_tiles(c)

        if self.break_target is not None and pos.distance_squared(self.break_target) <= 2:
            if not self._is_breakable_enemy_walkable_tile(c, self.break_target):
                self._remember_recent_broken_tile(c, self.break_target)
                self._clear_break_target()

        friendly_positions = []
        try:
            for uid in c.get_nearby_units():
                try:
                    if uid == c.get_id():
                        continue
                    if c.get_team(uid) != c.get_team():
                        continue
                    if c.get_entity_type(uid) != EntityType.BUILDER_BOT:
                        continue
                    friendly_positions.append(c.get_position(uid))
                except Exception:
                    continue
        except Exception:
            pass

        best_pos = None
        best_kind = None
        best_score = 10**18

        try:
            bids = c.get_nearby_buildings()
        except Exception:
            bids = []

        for bid in bids:
            try:
                team = c.get_team(bid)
                etype = c.get_entity_type(bid)
            except Exception:
                continue
            if team == c.get_team() or etype not in BREAKABLE_WALKABLE_TYPES:
                continue

            try:
                tile_pos = c.get_position(bid)
            except Exception:
                continue

            kind = self._break_target_kind_for(c, tile_pos, bid, etype, team)
            if kind is None:
                continue

            pressure = self._friendly_pressure_penalty(c, pos, tile_pos, friendly_positions)
            if pressure >= FRIENDLY_BREAK_HARD_PENALTY:
                continue

            score = pos.distance_squared(tile_pos)
            if core_target is not None:
                score += tile_pos.distance_squared(core_target)

            score += self._break_priority_score_adjust(c, bid, etype)
            if self._tile_is_live_break_feed_tile(c, tile_pos, bid, etype, team):
                score -= BREAK_FEED_PRIORITY_BONUS

            score += pressure
            score += self._break_lane_penalty(pos, tile_pos, core_target)

            if self._recently_broken_tile(c, tile_pos):
                score += 5000

            if (
                self.break_target is not None
                and tile_pos.x == self.break_target.x
                and tile_pos.y == self.break_target.y
            ):
                score -= BREAK_STICKY_BONUS

            if score < best_score:
                best_score = score
                best_pos = tile_pos
                best_kind = kind

        if best_pos is not None:
            self.break_target = Position(best_pos.x, best_pos.y)
            self.break_target_kind = best_kind
            self.break_target_stale_turns = 0
            return self.break_target

        if self.break_target is not None and not self._recently_broken_tile(c, self.break_target):
            self.break_target_stale_turns += 1
            if self.break_target_stale_turns <= BREAK_TARGET_STALE_LIMIT:
                return self.break_target

        self._clear_break_target()
        return None

    # -- Scout -----------------------------------------------------------
    def _do_scout(self, c, pos):
        cx = c.get_map_width() // 2
        cy = c.get_map_height() // 2
        centre = Position(cx, cy)

        if pos.distance_squared(centre) <= SCOUT_ARRIVE_DIST2:
            if not hasattr(self, '_scout_arrive_round'):
                self._scout_arrive_round = c.get_current_round()
            if c.get_current_round() - self._scout_arrive_round >= SCOUT_LOITER_ROUNDS:
                self.path = []
                self.path_index_by_xy = {}
                self.route_idx = 0
                self.ore_target = None
                self.last_dir = None
                self.stuck_turns = 0
                self._clear_path_plan()
                self.scout_done = True
            return

        if c.get_move_cooldown() > 0:
            return

        if self._move_with_path_plan(c, pos, centre, 'scout'):
            self.stuck_turns = 0
            return
        moved = self._local_step_toward_target(
            c,
            pos,
            centre,
            back=_OPP.get(self.last_dir) if self.last_dir else None,
        )
        if moved is not None:
            self.last_dir = moved
            self.stuck_turns = 0
            return
        self.stuck_turns += 1
    # -- Miner outward ---------------------------------------------------
    def _do_outward(self, c, pos):
        if self.ore_target is not None and self._ore_invalid(c, self.ore_target):
            self.ore_target = None

        if c.get_action_cooldown() == 0:
            for d in CARDINALS:
                check = pos.add(d)
                try:
                    if c.can_build_harvester(check):
                        c.build_harvester(check)
                        self.harvest_count += 1
                        self.ore_target = None
                        self.phase = 'route'
                        self.route_idx = max(0, len(self.path) - 1)
                        self.stuck_turns = 0
                        self._clear_path_plan()
                        return
                except Exception:
                    pass

        if c.get_move_cooldown() > 0:
            return

        if self.ore_target is None:
            self.ore_target = self._nearest_ore(c, pos)

        if self.ore_target is not None:
            moved = self._step_toward_target(c, pos, self.ore_target)
            if moved:
                self.last_dir = moved
                self.stuck_turns = 0
                return
            self.stuck_turns += 1
            if self.stuck_turns >= 2:
                self.ore_target = None
                self._start_escape(pos, 'outward')
            return

        moved = self._outward_step(c, pos)
        if moved:
            self.last_dir = moved
            self.stuck_turns = 0
            return

        self.stuck_turns += 1
        if self.stuck_turns >= 2:
            self._start_escape(pos, 'outward')

    def _outward_step(self, c, pos):
        back = _OPP.get(self.last_dir) if self.last_dir else None
        for d in (self.spoke_dir, _RIGHT.get(self.spoke_dir), _LEFT.get(self.spoke_dir)):
            if d is None or d == back:
                continue
            if self._do_step(c, pos, d):
                return d

        frontier = self._spoke_frontier_target(c, pos)
        best_dir = None
        best_score = 10**18
        for d in self._spoke_diagonal_dirs():
            if d is None or d == back:
                continue
            nxt = pos.add(d)
            if not self._in_bounds(c, nxt):
                continue
            if self._is_wall(c, nxt):
                continue
            if self._has_other_bot(c, nxt):
                continue

            score = nxt.distance_squared(frontier)
            score += self._lane_penalty(pos.x, pos.y, nxt.x, nxt.y)
            if d == self.last_dir:
                score -= 2
            if d == back:
                score += 4

            if score < best_score:
                best_score = score
                best_dir = d

        if best_dir is not None and self._do_step(c, pos, best_dir):
            return best_dir
        return None

    def _step_toward_target(self, c, pos, target):
        if self._move_with_path_plan(c, pos, target, 'outward'):
            return self.last_dir

        target = self._resolve_move_target(c, pos, target)
        if target is None:
            return None

        back = _OPP.get(self.last_dir) if self.last_dir else None
        return self._local_step_toward_target(c, pos, target, back)

    def _spoke_diagonal_dirs(self):
        return {
            Direction.NORTH: (Direction.NORTHEAST, Direction.NORTHWEST),
            Direction.SOUTH: (Direction.SOUTHEAST, Direction.SOUTHWEST),
            Direction.EAST: (Direction.NORTHEAST, Direction.SOUTHEAST),
            Direction.WEST: (Direction.NORTHWEST, Direction.SOUTHWEST),
        }.get(self.spoke_dir, ())

    def _spoke_frontier_target(self, c, pos):
        if self.spoke_dir is None or self.spoke_dir == Direction.CENTRE:
            return pos
        vx, vy = _DIR_VECTORS.get(self.spoke_dir, (0, 0))
        return Position(
            max(0, min(c.get_map_width() - 1, pos.x + vx * 4)),
            max(0, min(c.get_map_height() - 1, pos.y + vy * 4)),
        )

    def _ordered_target_dirs(self, pos, target):
        dx = target.x - pos.x
        dy = target.y - pos.y

        ordered = []
        if dx != 0 and dy != 0:
            try:
                diag = pos.direction_to(target)
            except Exception:
                diag = None
            if diag not in (None, Direction.CENTRE):
                ordered.append(diag)

        if abs(dx) >= abs(dy):
            primary = Direction.EAST if dx > 0 else Direction.WEST if dx < 0 else None
            secondary = Direction.SOUTH if dy > 0 else Direction.NORTH if dy < 0 else None
        else:
            primary = Direction.SOUTH if dy > 0 else Direction.NORTH if dy < 0 else None
            secondary = Direction.EAST if dx > 0 else Direction.WEST if dx < 0 else None

        ordered.extend((
            primary,
            secondary,
            _RIGHT.get(primary) if primary else None,
            _LEFT.get(primary) if primary else None,
        ))
        return ordered

    def _local_step_toward_target(self, c, pos, target, back=None):
        seen = set()
        for d in self._ordered_target_dirs(pos, target):
            if d is None or d == back or d in seen:
                continue
            seen.add(d)
            if self._do_step(c, pos, d):
                return d
        return None

    def _escape_diagonal_dirs(self):
        return {
            Direction.NORTH: (Direction.NORTHEAST, Direction.NORTHWEST),
            Direction.SOUTH: (Direction.SOUTHEAST, Direction.SOUTHWEST),
            Direction.EAST: (Direction.NORTHEAST, Direction.SOUTHEAST),
            Direction.WEST: (Direction.NORTHWEST, Direction.SOUTHWEST),
        }.get(self.escape_dir, ())

    # -- Escape ----------------------------------------------------------
    def _start_escape(self, pos, return_phase):
        self.phase = 'escape'
        self.escape_return_phase = return_phase

        if self.spoke_dir in (Direction.NORTH, Direction.SOUTH):
            self.escape_dir = Direction.EAST if (pos.x + pos.y) % 2 == 0 else Direction.WEST
        else:
            self.escape_dir = Direction.NORTH if (pos.x + pos.y) % 2 == 0 else Direction.SOUTH

        self.escape_steps = 2
        self.stuck_turns = 0

    def _do_escape(self, c, pos):
        if c.get_move_cooldown() > 0:
            return

        if self.escape_steps <= 0:
            self.phase = self.escape_return_phase
            return

        for d in self._escape_diagonal_dirs():
            if self._do_step(c, pos, d):
                self.last_dir = d
                self.escape_steps -= 1
                if self.escape_steps <= 0:
                    self.phase = self.escape_return_phase
                return

        if self._do_step(c, pos, self.escape_dir):
            self.last_dir = self.escape_dir
            self.escape_steps -= 1
            if self.escape_steps <= 0:
                self.phase = self.escape_return_phase
            return

        alt = _OPP.get(self.escape_dir)
        if alt and self._do_step(c, pos, alt):
            self.last_dir = alt
            self.escape_dir = alt
            self.escape_steps -= 1
            if self.escape_steps <= 0:
                self.phase = self.escape_return_phase
            return

        self.spoke_dir = _RIGHT.get(self.spoke_dir, Direction.NORTH)
        self.phase = self.escape_return_phase

    # -- Sabotage --------------------------------------------------------
    def _activate_saboteur(self, c):
        self.role = 'saboteur'
        self.phase = 'sabotage'
        self.ore_target = None
        self.path = []
        self.path_index_by_xy = {}
        self.route_idx = 0
        self.route_pivot = None
        self.route_skip_until_idx = None
        self.pending_bridge_from = None
        self.pending_bridge_to = None
        self.pending_bridge_anchor = None
        self.pending_bridge_stage = None
        self.last_dir = None
        self.stuck_turns = 0
        self.escape_dir = None
        self.escape_steps = 0
        self._clear_path_plan()

        cur = c.get_position()
        cur_xy = (cur.x, cur.y)
        self.sabotage_path = [cur_xy]
        self.sabotage_visited = {cur_xy}
        self.sabotage_phase = 'travel'
        self.sabotage_dir = self._cardinal_dir(cur, self._current_enemy_target(c))
        self.break_coord_slot = c.get_id() % 3
        self._clear_break_target()

    def _maybe_activate_saboteur(self, c):
        if self.role != 'miner' or self.stay_miner:
            return
        if self.harvest_count < MIN_HARVESTS_BEFORE_SABOTAGE:
            return

        round_num = c.get_current_round()
        slot = c.get_id() % 4
        if round_num >= SABOTAGE_START_1 and slot == 0:
            self._activate_saboteur(c)
        elif round_num >= SABOTAGE_START_2 and slot == 1:
            self._activate_saboteur(c)

    def _is_in_enemy_territory(self, c, pos):
        target = self._current_enemy_target(c)
        return pos.distance_squared(target) < pos.distance_squared(self.core_pos)

    def _conv_dir_for(self, tile_pos):
        return self._cardinal_dir(tile_pos, self.core_pos)

    def _tile_is_our_team(self, c, bid):
        try:
            return c.get_team(bid) == c.get_team()
        except Exception:
            return False

    def _sab_passable(self, c, nxt):
        if not self._in_bounds(c, nxt):
            return False
        if self._is_wall(c, nxt):
            return False
        if self._has_other_bot(c, nxt):
            return False

        bid = c.get_tile_building_id(nxt)
        if bid is None:
            return True

        try:
            etype = c.get_entity_type(bid)
            if etype in WALKABLE_TYPES:
                return True
            if etype == EntityType.CORE and c.get_team(bid) == c.get_team():
                return True
            return False
        except Exception:
            return False

    def _sab_can_fill(self, c, nxt):
        if not self._sab_passable(c, nxt):
            return False
        if nxt.distance_squared(self.core_pos) < SABOTAGE_SAFE_CORE_DIST2:
            return False
        if not self._is_in_enemy_territory(c, nxt):
            return False

        bid = c.get_tile_building_id(nxt)
        if bid is not None:
            try:
                etype = c.get_entity_type(bid)
                if etype in WALKABLE_TYPES:
                    return self._tile_is_our_team(c, bid)
                return False
            except Exception:
                return False
        return True

    def _sab_stamp(self, c, tile_pos):
        if not self._is_in_enemy_territory(c, tile_pos):
            return
        if self._is_enemy_walkable_tile(c, tile_pos):
            return

        conv_dir = self._conv_dir_for(tile_pos)
        if conv_dir is None:
            return

        bid = c.get_tile_building_id(tile_pos)
        if bid is not None:
            try:
                etype = c.get_entity_type(bid)
                if etype == EntityType.CONVEYOR:
                    try:
                        if c.get_direction(bid) == conv_dir:
                            return
                    except Exception:
                        return
                    try:
                        if c.can_destroy(tile_pos):
                            c.destroy(tile_pos)
                    except Exception:
                        pass
                    return
                elif etype == EntityType.ROAD:
                    try:
                        if c.can_destroy(tile_pos):
                            c.destroy(tile_pos)
                    except Exception:
                        pass
                    return
                else:
                    return
            except Exception:
                return

        try:
            if c.can_build_conveyor(tile_pos, conv_dir):
                c.build_conveyor(tile_pos, conv_dir)
        except Exception:
            pass

    def _tile_has_correct_conveyor(self, c, tile_pos):
        conv_dir = self._conv_dir_for(tile_pos)
        if conv_dir is None:
            return True
        try:
            bid = c.get_tile_building_id(tile_pos)
            if bid is None:
                return False
            if c.get_entity_type(bid) != EntityType.CONVEYOR:
                return False
            return c.get_direction(bid) == conv_dir
        except Exception:
            return False

    def _sab_candidate_dirs(self):
        base = self.sabotage_dir or Direction.EAST
        return [base, _RIGHT[base], _LEFT[base], _OPP[base]]

    def _do_sabotage(self, c, pos):
        if self._attack_enemy_tile_if_profitable(c, pos):
            return

        if self.enemy_core_pos is None:
            self._try_read_enemy_core_marker(c)
        self._maybe_advance_enemy_candidate(c, pos)

        cur_xy = (pos.x, pos.y)
        target = self._current_enemy_target(c)

        if self.sabotage_phase == 'travel':
            self.sabotage_dir = self._cardinal_dir(pos, target)
            if c.get_move_cooldown() == 0:
                if not self._is_in_enemy_territory(c, pos):
                    self._move_best_of_all_dirs(c, pos, target)
                    return
            if not self._is_in_enemy_territory(c, pos):
                return
            self.sabotage_phase = 'fill'
            self.sabotage_path = [cur_xy]
            self.sabotage_visited = {cur_xy}

        if self._sab_can_fill(c, pos):
            if c.get_action_cooldown() == 0:
                self._sab_stamp(c, pos)
            if not self._tile_has_correct_conveyor(c, pos):
                return

        if c.get_move_cooldown() > 0:
            return

        if cur_xy not in self.sabotage_visited:
            self.sabotage_visited.add(cur_xy)
        if not self.sabotage_path or self.sabotage_path[-1] != cur_xy:
            self.sabotage_path.append(cur_xy)

        if self._move_best_of_all_dirs(c, pos, target):
            return

        for d in self._sab_candidate_dirs():
            nxt = pos.add(d)
            nxt_xy = (nxt.x, nxt.y)
            if nxt_xy in self.sabotage_visited:
                continue
            if not self._sab_can_fill(c, nxt):
                continue
            try:
                if c.can_move(d):
                    c.move(d)
                    self.sabotage_dir = d
                    self.last_dir = d
                    return
            except Exception:
                pass

    # -- Miner route-back with conveyors + bridging ----------------------
    def _clear_pending_bridge_job(self):
        self.pending_bridge_from = None
        self.pending_bridge_to = None
        self.pending_bridge_anchor = None
        self.pending_bridge_stage = None

    def _clear_path_plan(self):
        self.path_plan_target = None
        self.path_plan_tiles = []
        self.path_plan_index = 0
        self.path_plan_kind = None

    def _finish_pending_bridge_job(self, c, pos):
        if self.pending_bridge_from is None or self.pending_bridge_to is None:
            return True

        bridge_from = self.pending_bridge_from
        bridge_to = self.pending_bridge_to
        bridge_anchor = self.pending_bridge_anchor
        if bridge_anchor is None:
            self._clear_pending_bridge_job()
            return True
        if pos.x != bridge_anchor.x or pos.y != bridge_anchor.y:
            self._clear_pending_bridge_job()
            return True

        if self._has_other_bot(c, bridge_from):
            return False

        if self._tile_has_bridge_target(c, bridge_from, bridge_to):
            self._clear_pending_bridge_job()
            return True

        try:
            bid = c.get_tile_building_id(bridge_from)
        except Exception:
            bid = None

        if bid is not None:
            try:
                etype = c.get_entity_type(bid)
            except Exception:
                return False

            if etype == EntityType.BRIDGE:
                try:
                    target = c.get_bridge_target(bid)
                    if target.x == bridge_to.x and target.y == bridge_to.y:
                        self._clear_pending_bridge_job()
                        return True
                except Exception:
                    return False
                self.pending_bridge_stage = 'clear'
                try:
                    if c.can_destroy(bridge_from):
                        c.destroy(bridge_from)
                except Exception:
                    pass
                return False

            if etype in (EntityType.ROAD, EntityType.CONVEYOR):
                self.pending_bridge_stage = 'clear'
                try:
                    if c.can_destroy(bridge_from):
                        c.destroy(bridge_from)
                except Exception:
                    pass
                return False

            return False

        self.pending_bridge_stage = 'build'
        if c.get_action_cooldown() > 0:
            return False

        try:
            if c.can_build_bridge(bridge_from, bridge_to):
                c.build_bridge(bridge_from, bridge_to)
                self._clear_pending_bridge_job()
                return True
        except Exception:
            pass
        return False

    def _do_route(self, c, pos):
        if not self.path:
            self._reset_for_next_trip(c)
            return

        if not self._finish_pending_bridge_job(c, pos):
            return

        try:
            bid = c.get_tile_building_id(pos)
            if (
                self.pending_bridge_from is None
                and bid is not None
                and c.get_entity_type(bid) == EntityType.CORE
                and c.get_team(bid) == c.get_team()
            ):
                self._reset_for_next_trip(c)
                return
        except Exception:
            pass

        xy = (pos.x, pos.y)
        self.route_idx = self.path_index_by_xy.get(xy, self.route_idx)
        self.route_idx = max(0, min(self.route_idx, len(self.path) - 1))

        if self.route_skip_until_idx is not None:
            if self.route_skip_until_idx < 0:
                if self.route_idx > 0:
                    next_walk = Position(self.path[self.route_idx - 1][0], self.path[self.route_idx - 1][1])
                    walk_dir = self._dir_between_adjacent(pos, next_walk)
                    if walk_dir is None:
                        self.route_skip_until_idx = None
                    elif c.get_move_cooldown() == 0 and self._do_step(c, pos, walk_dir):
                        self.route_idx -= 1
                    return

                core_step = self._best_route_core_tile(pos)
                try:
                    bid = c.get_tile_building_id(pos)
                    if (
                        bid is not None
                        and c.get_entity_type(bid) == EntityType.CORE
                        and c.get_team(bid) == c.get_team()
                    ):
                        self.route_skip_until_idx = None
                        self._reset_for_next_trip(c)
                        return
                except Exception:
                    pass
                if core_step is not None and c.get_move_cooldown() == 0:
                    step_dir = self._dir_between_adjacent(pos, core_step)
                    if step_dir is not None:
                        self._do_step(c, pos, step_dir)
                return

            if self.route_idx > self.route_skip_until_idx:
                next_walk = Position(self.path[self.route_idx - 1][0], self.path[self.route_idx - 1][1])
                walk_dir = self._dir_between_adjacent(pos, next_walk)
                if walk_dir is None:
                    self.route_skip_until_idx = None
                elif c.get_move_cooldown() == 0 and self._do_step(c, pos, walk_dir):
                    self.route_idx -= 1
                    if self.route_idx <= self.route_skip_until_idx:
                        self.route_skip_until_idx = None
                return
            self.route_skip_until_idx = None

        if self.route_pivot is not None:
            pivot_x, pivot_y, dest_x, dest_y = self.route_pivot
            pivot_pos = Position(pivot_x, pivot_y)
            dest_pos = Position(dest_x, dest_y)

            if pos.x == dest_pos.x and pos.y == dest_pos.y:
                self.route_pivot = None
                if self.route_idx > 0:
                    self.route_idx -= 1
            elif pos.x == pivot_pos.x and pos.y == pivot_pos.y:
                conv_dir = self._cardinal_dir(pos, dest_pos)
                if conv_dir is None:
                    self.route_pivot = None
                else:
                    if not self._tile_has_conveyor(c, pos, conv_dir):
                        if c.get_action_cooldown() == 0:
                            self._place_conveyor(c, pos, conv_dir)
                        return
                    if c.get_move_cooldown() == 0 and self._do_step(c, pos, conv_dir):
                        self.route_pivot = None
                        if self.route_idx > 0:
                            self.route_idx -= 1
                    return
            else:
                self.route_pivot = None

        reaching_core = self.route_idx <= 0
        if reaching_core:
            next_pos = self._best_route_core_tile(pos)
        else:
            next_pos = Position(self.path[self.route_idx - 1][0], self.path[self.route_idx - 1][1])

        if next_pos is None:
            self._reset_for_next_trip(c)
            return

        step_dir = self._dir_between_adjacent(pos, next_pos)
        if step_dir in CARDINALS:
            if not self._tile_has_conveyor(c, pos, step_dir):
                if c.get_action_cooldown() == 0:
                    self._place_conveyor(c, pos, step_dir)
                return
            if c.get_move_cooldown() == 0 and self._do_step(c, pos, step_dir):
                if reaching_core:
                    self._reset_for_next_trip(c)
                else:
                    self.route_idx -= 1
            return

        if step_dir is None:
            self._reset_for_next_trip(c)
            return

        if self._tile_has_bridge_target(c, pos, next_pos):
            if c.get_move_cooldown() == 0 and self._do_step(c, pos, step_dir):
                if reaching_core:
                    self._reset_for_next_trip(c)
                else:
                    self.route_idx -= 1
            return

        pivot = self._pick_route_pivot(c, pos, next_pos)
        if pivot is not None:
            conv_dir = self._cardinal_dir(pos, pivot)
            if conv_dir is None:
                self._reset_for_next_trip(c)
                return
            if not self._tile_has_conveyor(c, pos, conv_dir):
                if c.get_action_cooldown() == 0:
                    self._place_conveyor(c, pos, conv_dir)
                return
            if c.get_move_cooldown() == 0 and self._do_step(c, pos, conv_dir):
                self.route_pivot = (pivot.x, pivot.y, next_pos.x, next_pos.y)
            return

        bridge_target, bridge_idx = self._best_route_bridge_target(c, pos, next_pos, reaching_core)
        if bridge_target is None:
            if c.get_move_cooldown() == 0 and self._do_step(c, pos, step_dir):
                new_pos = c.get_position()
                if new_pos.x == next_pos.x and new_pos.y == next_pos.y:
                    self.pending_bridge_from = Position(pos.x, pos.y)
                    self.pending_bridge_to = Position(next_pos.x, next_pos.y)
                    self.pending_bridge_anchor = Position(new_pos.x, new_pos.y)
                    self.pending_bridge_stage = 'clear'
                    self.route_skip_until_idx = self.route_idx - 1 if not reaching_core else -1
                    if not reaching_core:
                        self.route_idx -= 1
            return

        if self._tile_has_bridge_target(c, pos, bridge_target):
            if c.get_move_cooldown() == 0 and self._do_step(c, pos, step_dir):
                if not reaching_core:
                    self.route_idx -= 1
                self.route_skip_until_idx = bridge_idx
            return

        if c.get_move_cooldown() == 0 and self._do_step(c, pos, step_dir):
            new_pos = c.get_position()
            if new_pos.x == next_pos.x and new_pos.y == next_pos.y:
                self.pending_bridge_from = Position(pos.x, pos.y)
                self.pending_bridge_to = Position(bridge_target.x, bridge_target.y)
                self.pending_bridge_anchor = Position(new_pos.x, new_pos.y)
                self.pending_bridge_stage = 'clear'
                self.route_skip_until_idx = bridge_idx
                if not reaching_core:
                    self.route_idx -= 1

    def _reset_for_next_trip(self, c):
        self.phase = 'outward'
        self.path = []
        self.path_index_by_xy = {}
        self.route_idx = 0
        self.route_pivot = None
        self.route_skip_until_idx = None
        self.pending_bridge_from = None
        self.pending_bridge_to = None
        self.pending_bridge_anchor = None
        self.pending_bridge_stage = None
        self.last_dir = None
        self.ore_target = None
        self.stuck_turns = 0
        self.escape_dir = None
        self.escape_steps = 0
        self.spoke_dir = _RIGHT.get(self.spoke_dir, CARDINALS[0])
        self._clear_path_plan()

        if c.get_current_round() >= RUSH_START and not self.stay_miner:
            self._activate_assaulter(c)
            return
        self._maybe_activate_saboteur(c)

    # -- Building helpers ------------------------------------------------
    def _dir_between_adjacent(self, pos, target):
        dx = target.x - pos.x
        dy = target.y - pos.y
        return {
            (0, -1): Direction.NORTH,
            (1, -1): Direction.NORTHEAST,
            (1, 0): Direction.EAST,
            (1, 1): Direction.SOUTHEAST,
            (0, 1): Direction.SOUTH,
            (-1, 1): Direction.SOUTHWEST,
            (-1, 0): Direction.WEST,
            (-1, -1): Direction.NORTHWEST,
        }.get((dx, dy))

    def _best_route_core_tile(self, pos):
        if self.core_pos is None:
            return None
        best = None
        best_score = None
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                tp = Position(self.core_pos.x + dx, self.core_pos.y + dy)
                direction = self._dir_between_adjacent(pos, tp)
                if direction is None:
                    continue
                score = (0 if direction in CARDINALS else 1, pos.distance_squared(tp), tp.x, tp.y)
                if best_score is None or score < best_score:
                    best_score = score
                    best = tp
        return best

    def _best_route_bridge_target(self, c, pos, next_pos, reaching_core):
        allow_direct_diagonal = False
        if next_pos is not None:
            step_dir = self._dir_between_adjacent(pos, next_pos)
            allow_direct_diagonal = step_dir is not None and step_dir not in CARDINALS

        best = None
        best_idx = None
        best_score = None

        window_start = max(0, self.route_idx - ROUTE_BRIDGE_LOOKBACK)
        for idx in range(self.route_idx - 1, window_start - 1, -1):
            tp = Position(self.path[idx][0], self.path[idx][1])
            if pos.distance_squared(tp) > 9:
                continue

            skip = self.route_idx - idx
            if skip < ROUTE_BRIDGE_MIN_SKIP and not (allow_direct_diagonal and idx == self.route_idx - 1):
                continue

            score = (-skip, pos.distance_squared(tp), tp.distance_squared(self.core_pos), tp.x, tp.y)
            if best_score is None or score < best_score:
                best_score = score
                best = tp
                best_idx = idx

        if self.core_pos is not None:
            for dx in (-1, 0, 1):
                for dy in (-1, 0, 1):
                    tp = Position(self.core_pos.x + dx, self.core_pos.y + dy)
                    if pos.distance_squared(tp) > 9:
                        continue
                    skip = self.route_idx + 1
                    if skip < ROUTE_BRIDGE_MIN_SKIP and not reaching_core:
                        continue
                    score = (-skip, pos.distance_squared(tp), tp.x, tp.y)
                    if best_score is None or score < best_score:
                        best_score = score
                        best = tp
                        best_idx = -1

        return best, best_idx

    def _route_pivot_candidates(self, pos, target):
        return (
            Position(target.x, pos.y),
            Position(pos.x, target.y),
        )

    def _route_pivot_usable(self, c, pivot):
        if not self._in_bounds(c, pivot):
            return False
        if self._is_wall(c, pivot):
            return False
        if self._has_other_bot(c, pivot):
            return False

        try:
            bid = c.get_tile_building_id(pivot)
        except Exception:
            bid = None

        if bid is None:
            try:
                env = c.get_tile_env(pivot)
            except Exception:
                return False
            return env not in (Environment.ORE_TITANIUM, Environment.ORE_AXIONITE)

        try:
            etype = c.get_entity_type(bid)
            if etype in WALKABLE_TYPES:
                return True
            return etype == EntityType.CORE and c.get_team(bid) == c.get_team()
        except Exception:
            return False

    def _pick_route_pivot(self, c, pos, target):
        best = None
        best_score = None
        for pivot in self._route_pivot_candidates(pos, target):
            if not self._route_pivot_usable(c, pivot):
                continue
            score = (
                pivot.distance_squared(target),
                0 if self._cardinal_dir(pivot, target) in CARDINALS else 1,
                pivot.x,
                pivot.y,
            )
            if best_score is None or score < best_score:
                best_score = score
                best = pivot
        return best

    def _tile_has_conveyor(self, c, tile_pos, direction):
        try:
            bid = c.get_tile_building_id(tile_pos)
            if bid is None:
                return False
            if c.get_entity_type(bid) != EntityType.CONVEYOR:
                return False
            return c.get_direction(bid) == direction
        except Exception:
            return False

    def _place_conveyor(self, c, tile_pos, direction):
        bid = c.get_tile_building_id(tile_pos)
        if bid is not None:
            try:
                etype = c.get_entity_type(bid)
                if etype == EntityType.CONVEYOR:
                    try:
                        if c.get_direction(bid) == direction:
                            return
                    except Exception:
                        return
                    try:
                        if c.can_destroy(tile_pos):
                            c.destroy(tile_pos)
                    except Exception:
                        pass
                    return
                elif etype in (EntityType.ROAD, EntityType.BRIDGE):
                    try:
                        if c.can_destroy(tile_pos):
                            c.destroy(tile_pos)
                    except Exception:
                        pass
                    return
                else:
                    return
            except Exception:
                return

        try:
            if c.can_build_conveyor(tile_pos, direction):
                c.build_conveyor(tile_pos, direction)
        except Exception:
            pass

    def _tile_has_bridge_target(self, c, tile_pos, target_pos):
        try:
            bid = c.get_tile_building_id(tile_pos)
            if bid is None:
                return False
            if c.get_entity_type(bid) != EntityType.BRIDGE:
                return False
            target = c.get_bridge_target(bid)
            return target.x == target_pos.x and target.y == target_pos.y
        except Exception:
            return False

    # -- Path recording --------------------------------------------------
    def _record_stack_path(self, path_list, pos):
        xy = (pos.x, pos.y)
        if not path_list:
            path_list.append(xy)
            self.path_index_by_xy[xy] = 0
            return
        if path_list[-1] == xy:
            return
        if len(path_list) >= 2 and path_list[-2] == xy:
            removed = path_list.pop()
            self.path_index_by_xy.pop(removed, None)
            return

        start = max(0, len(path_list) - 8)
        for i in range(len(path_list) - 2, start - 1, -1):
            if path_list[i] == xy:
                for removed in path_list[i + 1:]:
                    self.path_index_by_xy.pop(removed, None)
                del path_list[i + 1:]
                return
        path_list.append(xy)
        self.path_index_by_xy[xy] = len(path_list) - 1

    # -- Shared movement / map helpers ----------------------------------
    def _tile_can_be_approach_target(self, c, tp):
        if not self._in_bounds(c, tp):
            return False
        if self._is_wall(c, tp):
            return False
        if self._has_other_bot(c, tp):
            return False

        try:
            bid = c.get_tile_building_id(tp)
        except Exception:
            return False

        if bid is None:
            try:
                env = c.get_tile_env(tp)
            except Exception:
                return False
            return env not in (Environment.ORE_TITANIUM, Environment.ORE_AXIONITE)

        try:
            etype = c.get_entity_type(bid)
            team = c.get_team(bid)
        except Exception:
            return False

        if etype in WALKABLE_TYPES:
            return True
        if etype == EntityType.CORE and team == c.get_team():
            return True
        return False

    def _best_adjacent_ore_stand_target(self, c, pos, ore_pos):
        best = None
        best_score = 10**18

        for d in CARDINALS:
            vx, vy = _DIR_VECTORS[d]
            stand = Position(ore_pos.x - vx, ore_pos.y - vy)

            if not self._tile_can_be_approach_target(c, stand):
                continue

            score = pos.distance_squared(stand)
            score += self._lane_penalty(pos.x, pos.y, stand.x, stand.y)

            if score < best_score:
                best_score = score
                best = stand

        return best

    def _best_core_approach_target(self, c, pos, core_pos):
        best = None
        best_score = 10**18

        for tp in self._core_approach_tiles(c, core_pos):
            score = pos.distance_squared(tp)
            if tp.x == pos.x or tp.y == pos.y:
                score -= 2
            if score < best_score:
                best_score = score
                best = tp

        return best

    def _resolve_move_target(self, c, pos, target):
        if target is None:
            return None

        core_target = None
        if self.enemy_core_pos is not None:
            core_target = self.enemy_core_pos
        elif self.enemy_core_guess is not None:
            core_target = self.enemy_core_guess

        if core_target is not None and target.x == core_target.x and target.y == core_target.y:
            approach = self._best_core_approach_target(c, pos, core_target)
            if approach is not None:
                return approach

        try:
            env = c.get_tile_env(target)
            if env in (Environment.ORE_TITANIUM, Environment.ORE_AXIONITE):
                stand = self._best_adjacent_ore_stand_target(c, pos, target)
                if stand is not None:
                    return stand
        except Exception:
            pass

        return target

    def _best_assault_detour_target(self, c, pos, target):
        target = self._resolve_move_target(c, pos, target)
        if target is None:
            return None

        primary = self._cardinal_dir(pos, target)
        if primary is None:
            return None

        prefer_right_first = ((c.get_id() + self.assault_detour_flip) % 2 == 0)
        side_dirs = []
        first_side = _RIGHT.get(primary) if prefer_right_first else _LEFT.get(primary)
        second_side = _LEFT.get(primary) if prefer_right_first else _RIGHT.get(primary)
        if first_side is not None:
            side_dirs.append(first_side)
        if second_side is not None and second_side != first_side:
            side_dirs.append(second_side)

        fvx, fvy = _DIR_VECTORS[primary]
        best = None
        best_score = None

        for side in side_dirs:
            svx, svy = _DIR_VECTORS[side]
            for side_len in (ASSAULT_DETOUR_SIDE_LEN, 4, 8):
                for fwd_len in (ASSAULT_DETOUR_FWD_LEN, 0, 4):
                    cand = Position(
                        max(0, min(c.get_map_width() - 1, pos.x + svx * side_len + fvx * fwd_len)),
                        max(0, min(c.get_map_height() - 1, pos.y + svy * side_len + fvy * fwd_len)),
                    )
                    if not self._tile_can_be_approach_target(c, cand):
                        continue
                    score = (cand.distance_squared(target), pos.distance_squared(cand), cand.x, cand.y)
                    if best_score is None or score < best_score:
                        best_score = score
                        best = cand

        return best

    def _note_assault_progress(self, c, old_pos, new_pos, moved, active_target):
        cur = new_pos if moved else old_pos
        cur_xy = (cur.x, cur.y)
        stalled = not moved

        if active_target is not None and cur.distance_squared(active_target) >= old_pos.distance_squared(active_target):
            stalled = True
        if cur_xy in self.assault_recent_positions[-2:]:
            stalled = True

        self.assault_recent_positions.append(cur_xy)
        if len(self.assault_recent_positions) > ASSAULT_RECENT_MEMORY:
            self.assault_recent_positions = self.assault_recent_positions[-ASSAULT_RECENT_MEMORY:]

        if self.assault_detour_steps > 0 and self.assault_detour_target is not None:
            if cur.distance_squared(self.assault_detour_target) <= 2:
                self.assault_detour_target = None
                self.assault_detour_steps = 0
            elif moved:
                self.assault_detour_steps -= 1
                if self.assault_detour_steps <= 0:
                    self.assault_detour_target = None
                    self.assault_detour_steps = 0

        if stalled:
            self.assault_stall_turns += 1
        else:
            self.assault_stall_turns = 0

    # Generic map / stepping primitives shared by assault, scout, and economy.
    def _in_bounds(self, c, pos):
        return 0 <= pos.x < c.get_map_width() and 0 <= pos.y < c.get_map_height()

    def _is_wall(self, c, pos):
        try:
            return c.get_tile_env(pos) == Environment.WALL
        except Exception:
            return True

    def _has_other_bot(self, c, pos):
        try:
            bid = c.get_tile_builder_bot_id(pos)
            return bid is not None and bid != c.get_id()
        except Exception:
            return False

    def _tile_passable_for_move(self, c, pos):
        if not self._in_bounds(c, pos):
            return False
        if self._is_wall(c, pos):
            return False
        if self._has_other_bot(c, pos):
            return False

        bid = c.get_tile_building_id(pos)
        if bid is None:
            return True
        try:
            etype = c.get_entity_type(bid)
            if etype in WALKABLE_TYPES:
                return True
            if etype == EntityType.CORE and c.get_team(bid) == c.get_team():
                return True
            return False
        except Exception:
            return False

    def _move_best_of_all_dirs(self, c, pos, target):
        kind = 'scout' if self.is_scout and not self.scout_done else 'travel'
        if self._move_with_path_plan(c, pos, target, kind):
            return True

        tx, ty = target.x, target.y
        back = _OPP.get(self.last_dir) if self.last_dir else None
        best_d = None
        best_score = 10**18

        for d in ALL_DIRS:
            nxt = pos.add(d)
            if not self._tile_passable_for_move(c, nxt):
                continue

            nx, ny = nxt.x, nxt.y
            score = (nx - tx) * (nx - tx) + (ny - ty) * (ny - ty)

            if self._is_enemy_walkable_tile(c, nxt):
                score -= 10000
            if d == self.last_dir:
                score -= 3
            if d == back:
                score += 5

            if score < best_score:
                best_score = score
                best_d = d

        if best_d is not None and self._do_step(c, pos, best_d):
            self.last_dir = best_d
            return True
        return False

    def _do_step(self, c, pos, d):
        if d is None or d == Direction.CENTRE:
            return False

        nxt = pos.add(d)
        if not self._in_bounds(c, nxt):
            return False
        if self._is_wall(c, nxt):
            return False
        if self._has_other_bot(c, nxt):
            return False

        if c.get_action_cooldown() == 0:
            try:
                env = c.get_tile_env(nxt)
                if env not in (Environment.ORE_TITANIUM, Environment.ORE_AXIONITE):
                    if c.can_build_road(nxt):
                        c.build_road(nxt)
            except Exception:
                pass

        try:
            if c.can_move(d):
                c.move(d)
                return True
        except Exception:
            pass
        return False

    # Main assault movement routine: try the cached path planner first, then
    # use the local detour-aware step scorer for the final push.
    def _assault_move(self, c, pos, target, prefer_safe_tiles=True):
        base_target = self._resolve_move_target(c, pos, target)
        if base_target is None:
            return False

        active_target = base_target
        if self._move_with_path_plan(c, pos, active_target, 'assault'):
            new_pos = c.get_position()
            self._note_assault_progress(c, pos, new_pos, True, active_target)
            return True
        if self.enemy_core_pos is None:
            refreshed = self._current_enemy_target(c)
            if refreshed is not None and (refreshed.x != target.x or refreshed.y != target.y):
                base_target = self._resolve_move_target(c, pos, refreshed)
                if base_target is None:
                    self._note_assault_progress(c, pos, pos, False, active_target)
                    return False
                active_target = base_target

        if self.assault_detour_steps > 0 and self.assault_detour_target is not None:
            active_target = self.assault_detour_target
        elif self.assault_stall_turns >= ASSAULT_STALL_TRIGGER:
            detour = self._best_assault_detour_target(c, pos, base_target)
            if detour is not None:
                self.assault_detour_target = detour
                self.assault_detour_steps = ASSAULT_DETOUR_STEPS
                self.assault_detour_flip += 1
                self.assault_stall_turns = 0
                active_target = detour

        tx, ty = active_target.x, active_target.y
        back = _OPP.get(self.last_dir) if self.last_dir else None
        can_build = c.get_action_cooldown() == 0

        best_d = None
        best_score = 10**18
        best_needs_pave = False

        for d in ALL_DIRS:
            nxt = pos.add(d)
            if not self._in_bounds(c, nxt):
                continue
            if self._is_wall(c, nxt):
                continue
            if self._has_other_bot(c, nxt):
                continue

            needs_pave = False
            already_walkable = False
            try:
                bid = c.get_tile_building_id(nxt)
            except Exception:
                bid = None

            if bid is None:
                if can_build:
                    needs_pave = True
                else:
                    continue
            else:
                try:
                    etype = c.get_entity_type(bid)
                    team = c.get_team(bid)
                except Exception:
                    continue
                if etype in WALKABLE_TYPES:
                    already_walkable = True
                elif etype == EntityType.CORE and team == c.get_team():
                    already_walkable = True
                else:
                    continue

            nx, ny = nxt.x, nxt.y
            score = (nx - tx) * (nx - tx) + (ny - ty) * (ny - ty)

            if already_walkable and self._is_enemy_walkable_tile(c, nxt):
                if prefer_safe_tiles:
                    score += 7000
                else:
                    score -= 7000
            elif already_walkable:
                score -= 10

            recent = list(reversed(self.assault_recent_positions[-ASSAULT_RECENT_MEMORY:]))
            for idx, (rx, ry) in enumerate(recent, start=1):
                if nx == rx and ny == ry:
                    score += ASSAULT_REPEAT_PENALTY * (ASSAULT_RECENT_MEMORY - min(idx, ASSAULT_RECENT_MEMORY) + 1)
                    break

            if d == self.last_dir:
                score -= 3
            if d == back:
                score += 6

            try:
                env = c.get_tile_env(nxt)
                if env in (Environment.ORE_TITANIUM, Environment.ORE_AXIONITE):
                    score += 300
            except Exception:
                pass

            if score < best_score:
                best_score = score
                best_d = d
                best_needs_pave = needs_pave

        if best_d is None:
            self._note_assault_progress(c, pos, pos, False, active_target)
            return False

        nxt = pos.add(best_d)
        if best_needs_pave:
            try:
                if c.can_build_road(nxt):
                    c.build_road(nxt)
                else:
                    self._note_assault_progress(c, pos, pos, False, active_target)
                    return False
            except Exception:
                self._note_assault_progress(c, pos, pos, False, active_target)
                return False

        try:
            if c.can_move(best_d):
                c.move(best_d)
                self.last_dir = best_d
                new_pos = c.get_position()
                self._note_assault_progress(c, pos, new_pos, True, active_target)
                return True
        except Exception:
            pass

        self._note_assault_progress(c, pos, pos, False, active_target)
        return False

    # Shared path planner. Attackers use it for long pushes and for breaking out
    # of stalls, while scouts / outward miners can borrow it when they get stuck.
    def _should_use_path_plan(self, c, pos, target, kind):
        resolved = self._resolve_move_target(c, pos, target)
        if resolved is None:
            return False

        d2 = pos.distance_squared(resolved)
        if kind == 'assault':
            return d2 >= PATH_PLAN_ATTACK_DIST2 or self.assault_stall_turns >= PATH_PLAN_STALL_TRIGGER
        if kind == 'outward':
            return d2 >= PATH_PLAN_OUTWARD_DIST2 or self.stuck_turns >= PATH_PLAN_STALL_TRIGGER
        if kind == 'scout':
            return d2 >= PATH_PLAN_SCOUT_DIST2 or self.stuck_turns >= PATH_PLAN_STALL_TRIGGER
        return d2 >= PATH_PLAN_OUTWARD_DIST2 or self.stuck_turns >= PATH_PLAN_STALL_TRIGGER

    def _core_approach_tiles(self, c, core_pos):
        tiles = []
        for dx in (-2, -1, 0, 1, 2):
            for dy in (-2, -1, 0, 1, 2):
                if max(abs(dx), abs(dy)) != 2:
                    continue
                tp = Position(core_pos.x + dx, core_pos.y + dy)
                if self._tile_can_be_approach_target(c, tp):
                    tiles.append(tp)
        return tiles

    def _path_goal_positions(self, c, pos, target, kind):
        if target is None:
            return []

        goal_tiles = []
        is_core_like = False
        if self.enemy_core_pos is not None:
            is_core_like = target.x == self.enemy_core_pos.x and target.y == self.enemy_core_pos.y
        if not is_core_like and self.enemy_core_guess is not None:
            is_core_like = target.x == self.enemy_core_guess.x and target.y == self.enemy_core_guess.y

        if kind == 'assault' and is_core_like:
            goal_tiles = self._core_approach_tiles(c, target)
        else:
            try:
                env = c.get_tile_env(target)
            except Exception:
                env = None
            if env in (Environment.ORE_TITANIUM, Environment.ORE_AXIONITE):
                for d in CARDINALS:
                    vx, vy = _DIR_VECTORS[d]
                    stand = Position(target.x - vx, target.y - vy)
                    if self._tile_can_be_approach_target(c, stand):
                        goal_tiles.append(stand)

        if goal_tiles:
            return goal_tiles

        resolved = self._resolve_move_target(c, pos, target)
        if resolved is None or not self._tile_can_be_approach_target(c, resolved):
            return []
        return [resolved]

    def _path_plan_key(self, kind, goals):
        return (kind, tuple(sorted((p.x, p.y) for p in goals)))

    def _path_heuristic(self, xy, goals_xy):
        x, y = xy
        best = 10**18
        for gx, gy in goals_xy:
            h = max(abs(gx - x), abs(gy - y))
            if h < best:
                best = h
        return best if best < 10**18 else 0

    def _build_path_plan(self, c, pos, target, kind):
        goals = self._path_goal_positions(c, pos, target, kind)
        if not goals:
            self._clear_path_plan()
            return False

        key = self._path_plan_key(kind, goals)
        start = (pos.x, pos.y)
        goals_xy = {(p.x, p.y) for p in goals}

        frontier = [(self._path_heuristic(start, goals_xy), 0, start)]
        came_from = {}
        best_cost = {start: 0}
        visited = set()

        while frontier:
            _, cost, cur = heapq.heappop(frontier)
            if cur in visited:
                continue
            visited.add(cur)

            if cur in goals_xy:
                rev = [cur]
                while cur in came_from:
                    cur = came_from[cur]
                    rev.append(cur)
                rev.reverse()
                self.path_plan_target = key
                self.path_plan_tiles = rev
                self.path_plan_index = 1
                self.path_plan_kind = kind
                return True

            cx, cy = cur
            for d in ALL_DIRS:
                nx = cx + _DIR_VECTORS[d][0]
                ny = cy + _DIR_VECTORS[d][1]
                nxt = (nx, ny)
                tp = Position(nx, ny)
                if nxt != start and not self._tile_can_be_approach_target(c, tp):
                    continue

                new_cost = cost + 1
                if new_cost >= best_cost.get(nxt, 10**18):
                    continue

                best_cost[nxt] = new_cost
                came_from[nxt] = cur
                priority = new_cost + self._path_heuristic(nxt, goals_xy)
                heapq.heappush(frontier, (priority, new_cost, nxt))

        self._clear_path_plan()
        if kind == 'assault' and self.enemy_core_pos is None:
            current = self._current_enemy_target(c)
            if current is not None and target.x == current.x and target.y == current.y:
                self.enemy_bad_targets.add((target.x, target.y))
                if self.enemy_candidates:
                    self.enemy_candidate_idx = (self.enemy_candidate_idx + 1) % len(self.enemy_candidates)
        return False

    def _move_with_path_plan(self, c, pos, target, kind):
        if not self._should_use_path_plan(c, pos, target, kind):
            if self.path_plan_kind == kind:
                self._clear_path_plan()
            return False

        goals = self._path_goal_positions(c, pos, target, kind)
        if not goals:
            return False

        key = self._path_plan_key(kind, goals)
        need_replan = (
            self.path_plan_kind != kind
            or self.path_plan_target != key
            or not self.path_plan_tiles
        )

        if not need_replan:
            prev_idx = max(0, self.path_plan_index - 1)
            if prev_idx >= len(self.path_plan_tiles) or self.path_plan_tiles[prev_idx] != (pos.x, pos.y):
                need_replan = True

        if need_replan and not self._build_path_plan(c, pos, target, kind):
            return False

        if self.path_plan_index >= len(self.path_plan_tiles):
            self._clear_path_plan()
            return False

        nx, ny = self.path_plan_tiles[self.path_plan_index]
        nxt = Position(nx, ny)
        step_dir = self._dir_between_adjacent(pos, nxt)
        if step_dir is None:
            self._clear_path_plan()
            return False

        if not self._tile_can_be_approach_target(c, nxt):
            self._clear_path_plan()
            return False

        if self._do_step(c, pos, step_dir):
            self.last_dir = step_dir
            self.path_plan_index += 1
            return True

        self._clear_path_plan()
        return False

    def _cardinal_dir(self, pos, target):
        dx = target.x - pos.x
        dy = target.y - pos.y
        if dx == 0 and dy == 0:
            return None
        if dx == 0:
            return Direction.SOUTH if dy > 0 else Direction.NORTH
        if dy == 0:
            return Direction.EAST if dx > 0 else Direction.WEST
        if abs(dx) >= abs(dy):
            return Direction.EAST if dx > 0 else Direction.WEST
        return Direction.SOUTH if dy > 0 else Direction.NORTH

    def _nearest_ore(self, c, pos):
        px, py = pos.x, pos.y
        best_ti = None
        best_ti_d = 10**18
        best_ax = None
        best_ax_d = 10**18

        for tp in c.get_nearby_tiles():
            env = c.get_tile_env(tp)
            if env not in (Environment.ORE_TITANIUM, Environment.ORE_AXIONITE):
                continue
            if c.get_tile_building_id(tp) is not None:
                continue

            tx, ty = tp.x, tp.y
            score = (tx - px) * (tx - px) + (ty - py) * (ty - py) + self._lane_penalty(px, py, tx, ty)

            if env == Environment.ORE_TITANIUM:
                if score < best_ti_d:
                    best_ti_d = score
                    best_ti = tp
            else:
                if score < best_ax_d:
                    best_ax_d = score
                    best_ax = tp

        return best_ti if best_ti is not None else best_ax

    def _lane_penalty(self, px, py, tx, ty):
        sd = self.spoke_dir
        if sd == Direction.NORTH and ty > py:
            return 12
        if sd == Direction.SOUTH and ty < py:
            return 12
        if sd == Direction.EAST and tx < px:
            return 12
        if sd == Direction.WEST and tx > px:
            return 12
        return 0

    def _ore_invalid(self, c, ore_pos):
        try:
            env = c.get_tile_env(ore_pos)
            if env not in (Environment.ORE_TITANIUM, Environment.ORE_AXIONITE):
                return True
            return c.get_tile_building_id(ore_pos) is not None
        except Exception:
            return True
