"""Raider role for the ultimate bot — sabotage and gunner hijacking.

The raider is a mid/late-game offensive unit that:
    1. Travels toward the enemy core using greedy 8-directional movement.
    2. On arrival, attempts to hijack live enemy feed tiles (conveyors
       carrying resources) by building gunners on them.
    3. Sabotages enemy conveyors by replacing them with friendly ones
       pointing toward our core.
    4. Attacks enemy buildings within action range.
    5. Shares enemy core position via markers for other raiders.

Inspired by version4's assault logic but with strict CPU budgeting.

CPU budget per round: ~100-200µs.
    - Marker scan: O(V) where V = nearby entities (~20 max).
    - Movement: O(8) greedy best-first.
    - Hijack/sabotage: O(4) cardinal neighbours.
    - Total well under 2ms.
"""

from cambc import Controller, Direction, EntityType, GameError, Position

from constants import (
    CARDINAL_DELTAS,
    DIRECTION_BY_DELTA,
    MAP_ENEMY,
    MAP_FREE,
    MAP_OBSTACLE,
    MAP_UNKNOWN,
    PASSABLE_TILES,
    WALKABLE_TILES,
)
from logger import log_event


# ---------------------------------------------------------------------------
# Raider tuning constants
# ---------------------------------------------------------------------------

# Squared distance to enemy core at which raider enters assault mode.
_ASSAULT_DIST_SQ = 25  # ~5 tiles

# Maximum rounds stuck before rotating approach direction.
_MAX_STUCK = 4

# Marker encoding for enemy core position sharing.
# Uses bit 30 as flag (from version4 pattern).
_MARKER_ENEMY_CORE_FLAG = 1 << 30

# Maximum markers to place per raider (prevent marker spam).
_MAX_MARKERS_PLACED = 3
_FORWARD_TURRET_MIN_ROUND = 160
_SIEGE_TURRET_MIN_ROUND = 220
_FORWARD_TURRET_TI_BUFFER = 180
_FORWARD_TURRET_MAX_NEARBY = 3


# ---------------------------------------------------------------------------
# State
# ---------------------------------------------------------------------------

class RaiderState:
    """Per-unit persistent state for the raider role."""
    __slots__ = (
        "core_xy",            # our core position
        "map_w",
        "map_h",
        "enemy_core_pos",     # confirmed enemy core (x, y) or None
        "enemy_candidates",   # list of symmetry-based enemy positions
        "enemy_idx",          # current candidate index
        "bad_targets",        # set of disproven enemy positions
        "phase",              # "travel" | "assault"
        "stuck_turns",
        "last_pos",
        "last_move_dir",      # (dx, dy) of last successful move
        "markers_placed",     # count of markers placed by this raider
        "orbit_angle",        # rotation step for orbiting enemy core
    )

    def __init__(self, core_pos: Position, map_w: int, map_h: int):
        cx, cy = core_pos.x, core_pos.y
        self.core_xy = (cx, cy)
        self.map_w = map_w
        self.map_h = map_h

        # Generate enemy core candidates via 3 symmetry types.
        self.enemy_candidates = [
            (map_w - 1 - cx, map_h - 1 - cy),  # rotational (180°)
            (map_w - 1 - cx, cy),               # vertical mirror
            (cx, map_h - 1 - cy),               # horizontal mirror
        ]
        self.enemy_idx = 0
        self.bad_targets = set()
        self.enemy_core_pos = None

        self.phase = "travel"
        self.stuck_turns = 0
        self.last_pos = None
        self.last_move_dir = (0, 0)
        self.markers_placed = 0
        self.orbit_angle = 0


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def run_raider(c: Controller, state: RaiderState, local_map):
    """Execute one tick of the raider behaviour.

    Total cost: ≤ 200µs per round.
    """
    pos = c.get_position()
    cur_xy = (pos.x, pos.y)
    rnd = c.get_current_round()
    uid = c.get_id()

    # --- Stuck detection (O(1)) ---
    if state.last_pos == cur_xy:
        state.stuck_turns += 1
    else:
        state.stuck_turns = 0
    state.last_pos = cur_xy

    # --- Always: scan for enemy core in vision (O(V), V≈20) ---
    _raider_scan_enemy_core(c, state, pos, rnd, uid)

    # --- Always: read enemy core markers from other raiders (O(V)) ---
    _raider_read_enemy_markers(c, state, rnd, uid)

    # --- Phase dispatch ---
    target = _raider_current_target(state)

    if state.phase == "travel":
        # Check if close enough to switch to assault
        dist_sq = (pos.x - target[0]) ** 2 + (pos.y - target[1]) ** 2
        if dist_sq <= _ASSAULT_DIST_SQ:
            state.phase = "assault"
            state.stuck_turns = 0
            state.orbit_angle = 0
            log_event(rnd, uid, "raider", f"({pos.x},{pos.y})",
                      "enter_assault", tx=target[0], ty=target[1])
            _run_raider_assault(c, state, pos, cur_xy, target, rnd, uid)
            return

        # Travel toward target
        _raider_travel(c, state, pos, cur_xy, target, rnd, uid)

    elif state.phase == "assault":
        _run_raider_assault(c, state, pos, cur_xy, target, rnd, uid)


# ---------------------------------------------------------------------------
# Travel phase — greedy 8-directional movement with road paving
# ---------------------------------------------------------------------------

def _raider_travel(c, state, pos, cur_xy, target, rnd, uid):
    """Greedy movement toward target with road-paving-ahead. O(8) per tick.

    Empty tiles are NOT passable.  Must build road on NEXT tile before
    moving, just like the defender.
    """

    # --- Can we do anything this tick? ---
    can_move_now = c.get_move_cooldown() == 0
    can_act_now = c.get_action_cooldown() == 0
    if not can_move_now and not can_act_now:
        return

    # --- Score all 8 directions ---
    best_score = -9999
    best_delta = None  # (dx, dy)

    tx, ty = target
    cur_dist = abs(pos.x - tx) + abs(pos.y - ty)

    for dx, dy in _ALL_8_DELTAS:
        nx, ny = pos.x + dx, pos.y + dy

        # Score: prefer tiles closer to target (Manhattan distance)
        new_dist = abs(nx - tx) + abs(ny - ty)
        score = (cur_dist - new_dist) * 100  # positive = getting closer

        # Bonus: continue in same direction (momentum)
        if (dx, dy) == state.last_move_dir:
            score += 5

        # Penalty: avoid backtracking
        if state.last_pos is not None:
            if (nx, ny) == state.last_pos:
                score -= 50

        if score > best_score:
            best_score = score
            best_delta = (dx, dy)

    if best_delta is None:
        state.stuck_turns += 1
        return

    nx, ny = pos.x + best_delta[0], pos.y + best_delta[1]
    move_dir = DIRECTION_BY_DELTA.get(best_delta)
    if move_dir is None:
        state.stuck_turns += 1
        return

    # --- Try to move if tile is already passable ---
    if can_move_now:
        try:
            if c.can_move(move_dir):
                c.move(move_dir)
                state.last_move_dir = best_delta
                state.stuck_turns = 0
                return
        except GameError:
            pass

    # --- Tile not passable: build road on NEXT tile first ---
    if can_act_now:
        next_pos = Position(nx, ny)
        try:
            if c.can_build_road(next_pos):
                c.build_road(next_pos)
                # Try to move onto freshly-built road this same tick
                if can_move_now:
                    try:
                        if c.can_move(move_dir):
                            c.move(move_dir)
                            state.last_move_dir = best_delta
                            state.stuck_turns = 0
                            return
                    except GameError:
                        pass
                state.stuck_turns = 0
                return
        except GameError:
            pass

    state.stuck_turns += 1

    # If stuck, cycle to next enemy candidate
    if state.stuck_turns >= _MAX_STUCK:
        state.bad_targets.add(target)
        state.enemy_idx = (state.enemy_idx + 1) % len(state.enemy_candidates)
        state.stuck_turns = 0
        new_target = _raider_current_target(state)
        log_event(rnd, uid, "raider", f"({pos.x},{pos.y})",
                  "target_cycled", old_tx=target[0], old_ty=target[1],
                  new_tx=new_target[0], new_ty=new_target[1])


# ---------------------------------------------------------------------------
# Assault phase — hijack, sabotage, attack
# ---------------------------------------------------------------------------

def _run_raider_assault(c, state, pos, cur_xy, target, rnd, uid):
    """Multi-mode assault near enemy base. One action per round max.

    Priority:
    1. Hijack: build gunner on live enemy feed tile (O(4))
    2. Sabotage: replace enemy conveyor with ours (O(1) current tile)
    3. Attack: fire at enemy building on our tile (O(1))
    4. Orbit: move around enemy core looking for opportunities (O(8))
    """

    if c.get_action_cooldown() == 0:
        # Mode 1: Hijack — find adjacent enemy conveyor carrying resources.
        # Try sentinel first to mirror the #1 replay's forward static pressure;
        # fall back to the cheaper gunner when sentinel placement is illegal.
        hijack = _find_hijack_opportunity(c, pos, state)
        if hijack is not None:
            gun_pos, gun_dir = hijack
            cleared_feed_tile = False
            try:
                bid = c.get_tile_building_id(gun_pos)
                if bid is not None and bid != 0 and c.get_team(bid) != c.get_team():
                    if c.can_destroy(gun_pos):
                        c.destroy(gun_pos)
                        cleared_feed_tile = True
            except GameError:
                pass
            try:
                if c.can_build_sentinel(gun_pos, gun_dir):
                    c.build_sentinel(gun_pos, gun_dir)
                    log_event(rnd, uid, "raider", f"({pos.x},{pos.y})",
                              "hijack_sentinel_built",
                              gx=gun_pos.x, gy=gun_pos.y,
                              cleared=1 if cleared_feed_tile else 0)
                    return
            except GameError:
                pass
            try:
                if c.can_build_gunner(gun_pos, gun_dir):
                    c.build_gunner(gun_pos, gun_dir)
                    log_event(rnd, uid, "raider", f"({pos.x},{pos.y})",
                              "hijack_gunner_built",
                              gx=gun_pos.x, gy=gun_pos.y,
                              cleared=1 if cleared_feed_tile else 0)
                    return
            except GameError:
                pass

        # Mode 2: Forward turret pressure — copy the #1 replay's habit of
        # leaving sentinels/gunners in enemy territory. Preflight the shot
        # geometry before spending scale.
        forward_turret = _find_forward_turret_opportunity(c, pos, state, target, rnd)
        if forward_turret is not None:
            turret_type, turret_pos, turret_dir, shot_target = forward_turret
            try:
                if turret_type == EntityType.SENTINEL:
                    c.build_sentinel(turret_pos, turret_dir)
                    log_event(rnd, uid, "raider", f"({pos.x},{pos.y})",
                              "forward_sentinel_built",
                              tx=turret_pos.x, ty=turret_pos.y,
                              sx=shot_target.x, sy=shot_target.y)
                    return
                c.build_gunner(turret_pos, turret_dir)
                log_event(rnd, uid, "raider", f"({pos.x},{pos.y})",
                          "forward_gunner_built",
                          tx=turret_pos.x, ty=turret_pos.y,
                          sx=shot_target.x, sy=shot_target.y)
                return
            except GameError:
                pass

        siege_turret = _find_siege_turret_opportunity(c, pos, state, target, rnd)
        if siege_turret is not None:
            turret_type, turret_pos, turret_dir, shot_target, clear_first = siege_turret
            try:
                if clear_first and c.can_destroy(turret_pos):
                    c.destroy(turret_pos)
                if turret_type == EntityType.SENTINEL:
                    c.build_sentinel(turret_pos, turret_dir)
                    log_event(rnd, uid, "raider", f"({pos.x},{pos.y})",
                              "siege_sentinel_built",
                              tx=turret_pos.x, ty=turret_pos.y,
                              sx=shot_target.x, sy=shot_target.y)
                    return
                c.build_gunner(turret_pos, turret_dir)
                log_event(rnd, uid, "raider", f"({pos.x},{pos.y})",
                          "siege_gunner_built",
                          tx=turret_pos.x, ty=turret_pos.y,
                          sx=shot_target.x, sy=shot_target.y)
                return
            except GameError:
                pass

        # Mode 3: Sabotage — stamp conveyor on enemy conveyor tile
        sabotaged = _sabotage_current_tile(c, state, pos, rnd, uid)
        if sabotaged:
            return

        # Mode 4: Attack — fire at enemy building on our tile
        cur_p = Position(pos.x, pos.y)
        try:
            bid = c.get_tile_building_id(cur_p)
            if bid is not None and bid != 0:
                if c.get_team(bid) != c.get_team():
                    if c.can_fire(cur_p):
                        c.fire(cur_p)
                        log_event(rnd, uid, "raider", f"({pos.x},{pos.y})",
                                  "attacked_enemy_building")
                        return
        except GameError:
            pass

    # Mode 5: Orbit — move around enemy core
    if c.get_move_cooldown() == 0:
        _raider_orbit(c, state, pos, target, rnd, uid)


# ---------------------------------------------------------------------------
# Hijack helper — find live enemy feed tile
# ---------------------------------------------------------------------------

def _find_hijack_opportunity(c, pos, state):
    """Check 4 cardinal neighbours for enemy conveyors carrying resources.

    O(4). Returns (Position, Direction) for gunner placement, or None.
    """
    my_team = c.get_team()

    for dx, dy in CARDINAL_DELTAS:
        nx, ny = pos.x + dx, pos.y + dy
        tp = Position(nx, ny)
        try:
            bid = c.get_tile_building_id(tp)
            if bid is None or bid == 0:
                continue
            if c.get_team(bid) == my_team:
                continue
            etype = c.get_entity_type(bid)
            if etype not in (EntityType.CONVEYOR, EntityType.ARMOURED_CONVEYOR):
                continue

            # Check if conveyor is carrying resources (live feed).
            # get_stored_resource returns ResourceType | None.
            try:
                stored = c.get_stored_resource(bid)
                if stored is not None:
                    # Build gunner facing toward enemy core
                    face_dir = _direction_toward_target(
                        (nx, ny),
                        state.enemy_core_pos or state.core_xy)
                    return tp, face_dir
            except GameError:
                continue
        except GameError:
            continue

    return None


def _find_siege_turret_opportunity(c, pos, state, target, rnd):
    if rnd < _SIEGE_TURRET_MIN_ROUND:
        return None
    if (pos.x - target[0]) ** 2 + (pos.y - target[1]) ** 2 > 36:
        return None
    if _count_nearby_friendly_forward_turrets(c, pos) >= _FORWARD_TURRET_MAX_NEARBY:
        return None
    try:
        unit_count = c.get_unit_count()
        if unit_count >= 48:
            return None
        ti, _ = c.get_global_resources()
        gun_cost, _ = c.get_gunner_cost()
        sent_cost, _ = c.get_sentinel_cost()
    except GameError:
        return None

    shot_target = Position(target[0], target[1])
    candidates = []
    for dx, dy in _ALL_8_DELTAS:
        bx = pos.x + dx
        by = pos.y + dy
        bpos = Position(bx, by)
        dist_to_target = abs(bx - target[0]) + abs(by - target[1])
        clear_first = False
        try:
            bid = c.get_tile_building_id(bpos)
            if bid is not None and bid != 0:
                if c.get_team(bid) == c.get_team():
                    continue
                etype = c.get_entity_type(bid)
                if etype not in (EntityType.ROAD, EntityType.MARKER, EntityType.BARRIER):
                    continue
                clear_first = True
        except GameError:
            continue
        candidates.append((dist_to_target, bpos, clear_first))

    candidates.sort(key=lambda item: (item[0], item[1].x, item[1].y))

    for _dist, bpos, clear_first in candidates:
        for direction in _direction_options_toward((bpos.x, bpos.y), target):
            if ti >= sent_cost + 260 and unit_count <= 44:
                try:
                    buildable = True
                    if not clear_first:
                        buildable = c.can_build_sentinel(bpos, direction)
                    if buildable:
                        return (EntityType.SENTINEL, bpos, direction, shot_target, clear_first)
                except GameError:
                    pass
            if ti >= gun_cost + 80:
                try:
                    buildable = True
                    if not clear_first:
                        buildable = c.can_build_gunner(bpos, direction)
                    if buildable:
                        return (EntityType.GUNNER, bpos, direction, shot_target, clear_first)
                except GameError:
                    pass

    return None


def _find_forward_turret_opportunity(c, pos, state, target, rnd):
    """Find a buildable forward turret with a useful shot or ammo feed.

    This is deliberately local and cheap: scan visible enemy buildings, then
    test only the 8 tiles around the raider.
    """
    if rnd < _FORWARD_TURRET_MIN_ROUND:
        return None
    if _count_nearby_friendly_forward_turrets(c, pos) >= _FORWARD_TURRET_MAX_NEARBY:
        return None
    try:
        if c.get_unit_count() >= 48:
            return None
        ti, _ = c.get_global_resources()
    except GameError:
        return None

    enemy_targets = _visible_enemy_building_targets(c)
    if not enemy_targets:
        return None

    feed_tiles = _enemy_conveyor_output_tiles(c)
    build_tiles = []
    for dx, dy in _ALL_8_DELTAS:
        bx = pos.x + dx
        by = pos.y + dy
        bpos = Position(bx, by)
        fed_from = feed_tiles.get((bx, by))
        feed_bonus = 0 if fed_from is not None else 1
        target_dist = min(
            abs(bx - tpos.x) + abs(by - tpos.y)
            for _prio, tpos in enemy_targets
        )
        build_tiles.append((feed_bonus, target_dist, bpos, fed_from))

    build_tiles.sort(key=lambda item: (item[0], item[1], item[2].x, item[2].y))

    try:
        sent_cost, _ = c.get_sentinel_cost()
    except GameError:
        sent_cost = 30
    try:
        gun_cost, _ = c.get_gunner_cost()
    except GameError:
        gun_cost = 10

    for _feed_bonus, _dist, bpos, fed_from in build_tiles:
        incoming_dir = None
        if fed_from is not None:
            incoming_dir = _direction_8_from_delta(fed_from[0] - bpos.x, fed_from[1] - bpos.y)

        for priority, shot_pos in enemy_targets:
            for direction in _direction_options_toward((bpos.x, bpos.y), (shot_pos.x, shot_pos.y)):
                if incoming_dir is not None and direction == incoming_dir:
                    continue
                if ti >= sent_cost + _FORWARD_TURRET_TI_BUFFER:
                    try:
                        if (
                            c.can_build_sentinel(bpos, direction)
                            and c.can_fire_from(bpos, direction, EntityType.SENTINEL, shot_pos)
                        ):
                            return (EntityType.SENTINEL, bpos, direction, shot_pos)
                    except GameError:
                        pass
                # Use gunners as the cheap fallback, especially when a visible
                # line shot exists but the sentinel scale hit is too expensive.
                if priority <= 3 and ti >= gun_cost + (_FORWARD_TURRET_TI_BUFFER // 2):
                    try:
                        if (
                            c.can_build_gunner(bpos, direction)
                            and c.can_fire_from(bpos, direction, EntityType.GUNNER, shot_pos)
                        ):
                            return (EntityType.GUNNER, bpos, direction, shot_pos)
                    except GameError:
                        pass

    # If an enemy conveyor can feed a new turret but no visible shot preflights,
    # still place a cheap gunner pointed at the likely enemy core. This mirrors
    # the replay's resource-line hijack pressure without spending sentinel scale.
    if ti < gun_cost + (_FORWARD_TURRET_TI_BUFFER // 2):
        return None
    for _feed_bonus, _dist, bpos, fed_from in build_tiles:
        if fed_from is None:
            continue
        incoming_dir = _direction_8_from_delta(fed_from[0] - bpos.x, fed_from[1] - bpos.y)
        for direction in _direction_options_toward((bpos.x, bpos.y), target):
            if direction == incoming_dir:
                continue
            try:
                if c.can_build_gunner(bpos, direction):
                    return (EntityType.GUNNER, bpos, direction, Position(target[0], target[1]))
            except GameError:
                pass

    return None


def _visible_enemy_building_targets(c):
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
        priority = _enemy_target_priority(etype)
        if priority is None:
            continue
        targets.append((priority, pos))

    targets.sort(key=lambda item: (item[0], item[1].x, item[1].y))
    return targets[:8]


def _enemy_target_priority(etype):
    if etype == EntityType.CORE:
        return 0
    if etype == EntityType.HARVESTER:
        return 1
    if etype in (EntityType.CONVEYOR, EntityType.SPLITTER, EntityType.ARMOURED_CONVEYOR, EntityType.BRIDGE):
        return 2
    if etype in (EntityType.GUNNER, EntityType.SENTINEL, EntityType.BREACH, EntityType.LAUNCHER, EntityType.FOUNDRY):
        return 3
    if etype in (EntityType.ROAD, EntityType.BARRIER):
        return 4
    return None


def _enemy_conveyor_output_tiles(c):
    my_team = c.get_team()
    outputs = {}
    try:
        nearby = c.get_nearby_buildings()
    except GameError:
        return outputs

    for bid in nearby:
        try:
            if c.get_team(bid) == my_team:
                continue
            etype = c.get_entity_type(bid)
            if etype not in (EntityType.CONVEYOR, EntityType.SPLITTER, EntityType.ARMOURED_CONVEYOR):
                continue
            direction = c.get_direction(bid)
            delta = _DELTA_BY_DIRECTION.get(direction)
            if delta is None:
                continue
            p = c.get_position(bid)
            outputs[(p.x + delta[0], p.y + delta[1])] = (p.x, p.y)
        except GameError:
            continue

    return outputs


def _count_nearby_friendly_forward_turrets(c, pos):
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
            p = c.get_position(bid)
            if (p.x - pos.x) ** 2 + (p.y - pos.y) ** 2 <= 20:
                count += 1
        except GameError:
            continue
    return count


# ---------------------------------------------------------------------------
# Sabotage helper — replace enemy conveyor with ours
# ---------------------------------------------------------------------------

def _sabotage_current_tile(c, state, pos, rnd, uid):
    """If standing on an enemy conveyor, destroy it and build ours pointing home.

    O(1). Returns True if action was taken.
    """
    cur_p = Position(pos.x, pos.y)
    my_team = c.get_team()

    try:
        bid = c.get_tile_building_id(cur_p)
        if bid is None or bid == 0:
            return False
        if c.get_team(bid) == my_team:
            return False
        etype = c.get_entity_type(bid)
        if etype not in (EntityType.CONVEYOR, EntityType.ROAD):
            return False

        # Destroy enemy building
        if c.can_destroy(cur_p):
            c.destroy(cur_p)

        # Build our conveyor pointing toward our core
        out_dir = _direction_toward_target(
            (pos.x, pos.y), state.core_xy)
        if c.can_build_conveyor(cur_p, out_dir):
            c.build_conveyor(cur_p, out_dir)
            log_event(rnd, uid, "raider", f"({pos.x},{pos.y})",
                      "sabotaged_conveyor")
            return True
    except GameError:
        pass

    return False


# ---------------------------------------------------------------------------
# Orbit movement — circle around enemy core looking for opportunities
# ---------------------------------------------------------------------------

# Orbit offsets: 8 positions around the target at distance 2-3.
_ORBIT_OFFSETS = [
    (0, -3), (2, -2), (3, 0), (2, 2),
    (0, 3), (-2, 2), (-3, 0), (-2, -2),
]

def _raider_orbit(c, state, pos, target, rnd, uid):
    """Move around enemy core in a circular pattern. O(8).

    Builds roads ahead if needed (enemy territory has no friendly roads).
    """
    tx, ty = target
    state.orbit_angle = (state.orbit_angle + 1) % len(_ORBIT_OFFSETS)
    can_act = c.get_action_cooldown() == 0

    # Try orbit positions, starting from current angle
    for i in range(len(_ORBIT_OFFSETS)):
        idx = (state.orbit_angle + i) % len(_ORBIT_OFFSETS)
        ox, oy = _ORBIT_OFFSETS[idx]
        goal_x, goal_y = tx + ox, ty + oy

        # Greedy move toward this orbit position
        best_delta = None
        best_dist = 9999

        for dx, dy in _ALL_8_DELTAS:
            nx, ny = pos.x + dx, pos.y + dy
            d = abs(nx - goal_x) + abs(ny - goal_y)
            if d < best_dist:
                best_dist = d
                best_delta = (dx, dy)

        if best_delta is not None:
            move_dir = DIRECTION_BY_DELTA.get(best_delta)
            if move_dir is None:
                continue
            try:
                if c.can_move(move_dir):
                    c.move(move_dir)
                    state.orbit_angle = idx
                    return
            except GameError:
                pass
            # Tile not passable: build road ahead
            if can_act:
                nx, ny = pos.x + best_delta[0], pos.y + best_delta[1]
                try:
                    if c.can_build_road(Position(nx, ny)):
                        c.build_road(Position(nx, ny))
                        if c.can_move(move_dir):
                            c.move(move_dir)
                            state.orbit_angle = idx
                            return
                        return  # Road built, move next tick
                except GameError:
                    pass

    # Fallback: any valid move (with road-ahead)
    for dx, dy in _ALL_8_DELTAS:
        move_dir = DIRECTION_BY_DELTA.get((dx, dy))
        if move_dir is None:
            continue
        try:
            if c.can_move(move_dir):
                c.move(move_dir)
                return
        except GameError:
            pass
        if can_act:
            nx, ny = pos.x + dx, pos.y + dy
            try:
                if c.can_build_road(Position(nx, ny)):
                    c.build_road(Position(nx, ny))
                    return  # Move next tick
            except GameError:
                pass


# ---------------------------------------------------------------------------
# Enemy core detection — scan vision and read markers
# ---------------------------------------------------------------------------

def _raider_scan_enemy_core(c, state, pos, rnd, uid):
    """Scan nearby buildings for enemy core. O(V).

    If found, lock enemy_core_pos and share via marker.
    """
    if state.enemy_core_pos is not None:
        return  # Already found

    my_team = c.get_team()
    try:
        for bid in c.get_nearby_buildings():
            try:
                if c.get_entity_type(bid) != EntityType.CORE:
                    continue
                if c.get_team(bid) == my_team:
                    continue
                epos = c.get_position(bid)
                state.enemy_core_pos = (epos.x, epos.y)
                log_event(rnd, uid, "raider", f"({pos.x},{pos.y})",
                          "enemy_core_found",
                          ex=epos.x, ey=epos.y)

                # Share via marker (if we haven't spammed too many)
                _raider_share_enemy_core(c, state, pos, rnd, uid)
                return
            except GameError:
                continue
    except GameError:
        pass


def _raider_read_enemy_markers(c, state, rnd, uid):
    """Read enemy core position from markers placed by other raiders. O(V).

    Looks for markers with the enemy core flag set.
    """
    if state.enemy_core_pos is not None:
        return  # Already know

    my_team = c.get_team()
    try:
        for bid in c.get_nearby_buildings():
            try:
                if c.get_entity_type(bid) != EntityType.MARKER:
                    continue
                if c.get_team(bid) != my_team:
                    continue
                val = c.get_marker_value(bid)
                if val & _MARKER_ENEMY_CORE_FLAG:
                    ex = (val >> 12) & 0xFFF
                    ey = val & 0xFFF
                    if 0 < ex < state.map_w and 0 < ey < state.map_h:
                        state.enemy_core_pos = (ex, ey)
                        log_event(rnd, uid, "raider", "(?,?)",
                                  "enemy_core_from_marker",
                                  ex=ex, ey=ey)
                        return
            except GameError:
                continue
    except GameError:
        pass


def _raider_share_enemy_core(c, state, pos, rnd, uid):
    """Place a marker encoding the enemy core position. O(1).

    Limited to _MAX_MARKERS_PLACED per raider to prevent spam.
    """
    if state.markers_placed >= _MAX_MARKERS_PLACED:
        return
    if state.enemy_core_pos is None:
        return

    ex, ey = state.enemy_core_pos
    marker_val = _MARKER_ENEMY_CORE_FLAG | ((ex & 0xFFF) << 12) | (ey & 0xFFF)

    # Place marker adjacent to current position.
    # API: can_place_marker(position) takes a single Position arg.
    for dx, dy in CARDINAL_DELTAS:
        mx, my = pos.x + dx, pos.y + dy
        mp = Position(mx, my)
        try:
            if c.can_place_marker(mp):
                c.place_marker(mp, marker_val)
                state.markers_placed += 1
                log_event(rnd, uid, "raider", f"({pos.x},{pos.y})",
                          "shared_enemy_core_marker",
                          ex=ex, ey=ey, count=state.markers_placed)
                return
        except GameError:
            continue


# ---------------------------------------------------------------------------
# Utility helpers
# ---------------------------------------------------------------------------

# All 8 movement deltas (cardinal + diagonal).
_ALL_8_DELTAS = (
    (0, -1), (1, -1), (1, 0), (1, 1),
    (0, 1), (-1, 1), (-1, 0), (-1, -1),
)

_DIRECTION_8_BY_DELTA = {
    (0, -1): Direction.NORTH,
    (1, -1): Direction.NORTHEAST,
    (1, 0): Direction.EAST,
    (1, 1): Direction.SOUTHEAST,
    (0, 1): Direction.SOUTH,
    (-1, 1): Direction.SOUTHWEST,
    (-1, 0): Direction.WEST,
    (-1, -1): Direction.NORTHWEST,
}

_DELTA_BY_DIRECTION = {direction: delta for delta, direction in _DIRECTION_8_BY_DELTA.items()}


def _raider_current_target(state):
    """Return the current enemy core target position.

    Uses confirmed enemy_core_pos if known, otherwise cycles through
    symmetry-based candidates.
    """
    if state.enemy_core_pos is not None:
        return state.enemy_core_pos

    # Filter out bad targets
    for _ in range(len(state.enemy_candidates)):
        candidate = state.enemy_candidates[state.enemy_idx]
        if candidate not in state.bad_targets:
            return candidate
        state.enemy_idx = (state.enemy_idx + 1) % len(state.enemy_candidates)

    # All candidates exhausted — just go to first one
    return state.enemy_candidates[0]


def _direction_toward_target(from_xy, to_xy):
    """Return the cardinal Direction closest to the vector from→to.

    O(1). Returns the dominant axis direction.
    """
    dx = to_xy[0] - from_xy[0]
    dy = to_xy[1] - from_xy[1]

    # Pick dominant axis
    if abs(dx) >= abs(dy):
        return Direction.EAST if dx > 0 else Direction.WEST
    else:
        return Direction.SOUTH if dy > 0 else Direction.NORTH


def _direction_8_from_delta(dx, dy):
    sx = 0 if dx == 0 else (1 if dx > 0 else -1)
    sy = 0 if dy == 0 else (1 if dy > 0 else -1)
    return _DIRECTION_8_BY_DELTA.get((sx, sy))


def _direction_options_toward(from_xy, to_xy):
    dx = to_xy[0] - from_xy[0]
    dy = to_xy[1] - from_xy[1]
    primary = _direction_8_from_delta(dx, dy)
    cardinal = _direction_toward_target(from_xy, to_xy)
    options = []
    for direction in (primary, cardinal):
        if direction is not None and direction not in options:
            options.append(direction)

    ranked = []
    for delta, direction in _DIRECTION_8_BY_DELTA.items():
        if direction in options:
            continue
        dot = dx * delta[0] + dy * delta[1]
        ranked.append((-dot, direction.name, direction))
    ranked.sort()
    options.extend(direction for _neg_dot, _name, direction in ranked)
    return tuple(options)
