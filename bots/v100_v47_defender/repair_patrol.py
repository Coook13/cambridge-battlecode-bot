"""Repair patrol bot for the ultimate bot — walks conveyor chains and fixes breaks.

The repair patrol is a mid-game support role that:
    1. Builds a patrol route from known friendly conveyors (via local_map).
    2. Walks the route sequentially, checking each conveyor for damage.
    3. Rebuilds destroyed conveyors with the correct direction.
    4. Heals damaged conveyors and harvesters.
    5. Loops forever, periodically refreshing the route.

CPU budget per round: ~100-200µs.
    - Movement: O(1) toward next waypoint.
    - Integrity check: O(1) single tile lookup.
    - Route rebuild: O(C) where C = known conveyors, but only every full loop.
"""

from cambc import Controller, Direction, EntityType, GameError, Position

from constants import (
    CARDINAL_DELTAS,
    CONVEYOR_MAX_HP,
    DIRECTION_BY_DELTA,
    HARVESTER_MAX_HP,
    BRIDGE_MAX_HP,
    MAP_CONVEYOR,
    MAP_ROAD,
    PASSABLE_TILES,
    WALKABLE_TILES,
)
from logger import log_event


# ---------------------------------------------------------------------------
# Tuning
# ---------------------------------------------------------------------------

# Rounds between full route rebuilds (even without completing a loop).
_ROUTE_REFRESH_INTERVAL = 100

# Maximum stuck rounds before skipping current waypoint.
_MAX_STUCK = 5

# All 8 movement deltas for greedy direction scoring.
_ALL_8_DELTAS = (
    (0, -1), (1, -1), (1, 0), (1, 1),
    (0, 1), (-1, 1), (-1, 0), (-1, -1),
)


# ---------------------------------------------------------------------------
# State
# ---------------------------------------------------------------------------

class RepairPatrolState:
    """Per-unit persistent state for the repair patrol role."""
    __slots__ = (
        "core_xy",
        "patrol_route",       # list[(x,y)] — conveyor positions to check
        "route_idx",          # current position in patrol_route
        "phase",              # "build_route" | "patrol" | "repair"
        "repair_target",      # (x, y) tile being repaired
        "repair_type",        # "missing" | "damaged" | "heal_harvester"
        "repair_direction",   # Direction for conveyor rebuild
        "stuck_turns",
        "last_pos",
        "last_route_round",   # round when route was last built
    )

    def __init__(self, core_pos: Position):
        self.core_xy = (core_pos.x, core_pos.y)
        self.patrol_route = []
        self.route_idx = 0
        self.phase = "build_route"
        self.repair_target = None
        self.repair_type = None
        self.repair_direction = None
        self.stuck_turns = 0
        self.last_pos = None
        self.last_route_round = 0


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def run_repair_patrol(c: Controller, state: RepairPatrolState, local_map):
    """Execute one tick of the repair patrol behaviour."""
    pos = c.get_position()
    cur_xy = (pos.x, pos.y)
    rnd = c.get_current_round()
    uid = c.get_id()

    # --- Stuck detection ---
    if state.last_pos == cur_xy:
        state.stuck_turns += 1
    else:
        state.stuck_turns = 0
    state.last_pos = cur_xy

    # --- Phase dispatch ---
    if state.phase == "build_route":
        _build_patrol_route(c, state, local_map, cur_xy, rnd, uid)
        return

    if state.phase == "repair":
        _handle_repair(c, state, pos, cur_xy, rnd, uid)
        return

    if state.phase == "patrol":
        _patrol_tick(c, state, pos, cur_xy, rnd, uid, local_map)


# ---------------------------------------------------------------------------
# Route building — scan local_map for known friendly conveyors
# ---------------------------------------------------------------------------

def _build_patrol_route(c, state, local_map, cur_xy, rnd, uid):
    """Build the patrol route from known friendly infrastructure.

    O(C) where C = number of conveyor tiles in local_map.
    Only runs when entering build_route phase (not every tick).

    Route includes:
    - All friendly conveyors
    - All friendly harvesters (to heal if damaged)
    - All friendly bridges (to heal if damaged)
    Sorted by distance from core (patrol inner chain first).
    """
    cx, cy = state.core_xy
    my_team = c.get_team()
    route = []

    if local_map is not None:
        # Scan all known entities for friendly infrastructure
        for eid, rec in local_map.entities.items():
            if not rec.get("alive", False):
                continue
            if rec.get("team") != my_team:
                continue

            etype = rec.get("entity_type")
            pos_data = rec.get("position")
            if pos_data is None:
                continue

            px, py = pos_data

            # Include conveyors, harvesters, and bridges in patrol
            if etype in (EntityType.CONVEYOR, EntityType.ARMOURED_CONVEYOR,
                         EntityType.HARVESTER, EntityType.BRIDGE):
                # Sort key: Manhattan distance from core
                dist = abs(px - cx) + abs(py - cy)
                route.append((dist, px, py))

    # Sort by distance from core (inner chain first, then outward)
    route.sort()
    state.patrol_route = [(x, y) for _, x, y in route]
    state.route_idx = 0
    state.phase = "patrol"
    state.last_route_round = rnd

    log_event(rnd, uid, "repair_patrol", f"({cur_xy[0]},{cur_xy[1]})",
              "route_built", waypoints=len(state.patrol_route))


# ---------------------------------------------------------------------------
# Patrol tick — walk route and check integrity
# ---------------------------------------------------------------------------

def _patrol_tick(c, state, pos, cur_xy, rnd, uid, local_map):
    """Walk the patrol route. At each waypoint, check infrastructure integrity."""

    # Periodically refresh route
    if rnd - state.last_route_round >= _ROUTE_REFRESH_INTERVAL:
        state.phase = "build_route"
        return

    if not state.patrol_route:
        state.phase = "build_route"
        return

    # Current patrol target
    target = state.patrol_route[state.route_idx]

    # --- Are we adjacent/on the target? Check integrity ---
    dist = abs(cur_xy[0] - target[0]) + abs(cur_xy[1] - target[1])
    if dist <= 1:
        issue = _check_infrastructure_integrity(c, target)
        if issue is not None:
            itype, direction = issue
            state.repair_target = target
            state.repair_type = itype
            state.repair_direction = direction
            state.phase = "repair"
            log_event(rnd, uid, "repair_patrol", f"({pos.x},{pos.y})",
                      "issue_found", tx=target[0], ty=target[1],
                      itype=itype)
            return

        # Tile is fine — advance to next waypoint
        state.route_idx += 1
        if state.route_idx >= len(state.patrol_route):
            # Completed full loop — rebuild route
            state.phase = "build_route"
            log_event(rnd, uid, "repair_patrol", f"({pos.x},{pos.y})",
                      "loop_complete")
        state.stuck_turns = 0
        return

    # --- Move toward current waypoint ---
    if c.get_move_cooldown() > 0:
        return

    _move_toward_target(c, pos, target)

    # Skip waypoint if stuck too long
    if state.stuck_turns >= _MAX_STUCK:
        state.route_idx += 1
        if state.route_idx >= len(state.patrol_route):
            state.phase = "build_route"
        state.stuck_turns = 0
        log_event(rnd, uid, "repair_patrol", f"({pos.x},{pos.y})",
                  "skipped_stuck_waypoint", tx=target[0], ty=target[1])


# ---------------------------------------------------------------------------
# Infrastructure integrity check
# ---------------------------------------------------------------------------

def _check_infrastructure_integrity(c, tile_xy):
    """Check if infrastructure at tile_xy is intact.

    O(1) — single tile building lookup.

    Returns:
        None if OK.
        ("missing", direction_or_None) if conveyor destroyed.
        ("damaged", None) if building damaged (needs heal).
    """
    x, y = tile_xy
    tp = Position(x, y)
    my_team = c.get_team()

    try:
        if not c.is_in_vision(tp):
            return None  # Can't check out-of-vision tiles

        bid = c.get_tile_building_id(tp)

        # Nothing on tile — infrastructure was destroyed
        if bid is None or bid == 0:
            return ("missing", None)

        etype = c.get_entity_type(bid)
        team = c.get_team(bid)

        # Enemy replaced our infrastructure
        if team != my_team:
            return ("missing", None)

        # Check HP for damage
        hp = c.get_hp(bid)

        if etype == EntityType.CONVEYOR:
            if hp < CONVEYOR_MAX_HP:
                return ("damaged", None)
        elif etype == EntityType.ARMOURED_CONVEYOR:
            # Armoured conveyors have higher HP, same logic
            if hp < 50:  # ARMOURED_CONVEYOR_MAX_HP
                return ("damaged", None)
        elif etype == EntityType.HARVESTER:
            if hp < HARVESTER_MAX_HP:
                return ("damaged", None)
        elif etype == EntityType.BRIDGE:
            if hp < BRIDGE_MAX_HP:
                return ("damaged", None)

    except GameError:
        return None  # Can't check = assume OK

    return None


# ---------------------------------------------------------------------------
# Repair handler
# ---------------------------------------------------------------------------

def _handle_repair(c, state, pos, cur_xy, rnd, uid):
    """Perform repair on the target tile.

    Must be adjacent to (or on) the repair target.
    """
    if state.repair_target is None:
        state.phase = "patrol"
        return

    rx, ry = state.repair_target
    dist = abs(cur_xy[0] - rx) + abs(cur_xy[1] - ry)

    # Move closer if not adjacent
    if dist > 1:
        if c.get_move_cooldown() == 0:
            _move_toward_target(c, pos, state.repair_target)
        if state.stuck_turns >= _MAX_STUCK:
            _clear_repair(state)
            log_event(rnd, uid, "repair_patrol", f"({pos.x},{pos.y})",
                      "repair_abandoned_stuck",
                      rx=rx, ry=ry)
        return

    # We're adjacent — perform repair
    if c.get_action_cooldown() > 0:
        return

    target_p = Position(rx, ry)

    if state.repair_type == "missing":
        # Try to rebuild conveyor (need to determine direction)
        # First try to destroy any enemy building on the tile
        try:
            if c.can_destroy(target_p):
                c.destroy(target_p)
        except GameError:
            pass

        # Determine direction: toward core
        cx, cy = state.core_xy
        dx = 1 if cx > rx else (-1 if cx < rx else 0)
        dy = 1 if cy > ry else (-1 if cy < ry else 0)
        # Prefer dominant axis
        if abs(cx - rx) >= abs(cy - ry):
            direction = DIRECTION_BY_DELTA.get((dx, 0))
        else:
            direction = DIRECTION_BY_DELTA.get((0, dy))

        if direction is None:
            direction = Direction.NORTH  # fallback

        try:
            if c.can_build_conveyor(target_p, direction):
                c.build_conveyor(target_p, direction)
                log_event(rnd, uid, "repair_patrol", f"({pos.x},{pos.y})",
                          "rebuilt_conveyor", rx=rx, ry=ry)
                _clear_repair(state)
                return
        except GameError:
            pass

        # Try road as fallback
        try:
            if c.can_build_road(target_p):
                c.build_road(target_p)
                log_event(rnd, uid, "repair_patrol", f"({pos.x},{pos.y})",
                          "rebuilt_road", rx=rx, ry=ry)
                _clear_repair(state)
                return
        except GameError:
            pass

        # Can't rebuild — move on
        _clear_repair(state)

    elif state.repair_type == "damaged":
        try:
            if c.can_heal(target_p):
                c.heal(target_p)
                log_event(rnd, uid, "repair_patrol", f"({pos.x},{pos.y})",
                          "healed_infrastructure", rx=rx, ry=ry)

                # Check if fully healed
                try:
                    bid = c.get_tile_building_id(target_p)
                    if bid is not None:
                        hp = c.get_hp(bid)
                        etype = c.get_entity_type(bid)
                        max_hp = _max_hp_for_type(etype)
                        if hp >= max_hp:
                            _clear_repair(state)
                            return
                except GameError:
                    pass
                # Not fully healed — continue next round
                return
            else:
                _clear_repair(state)
        except GameError:
            _clear_repair(state)

    else:
        _clear_repair(state)


def _clear_repair(state):
    """Reset repair state and return to patrol."""
    state.repair_target = None
    state.repair_type = None
    state.repair_direction = None
    state.phase = "patrol"


def _max_hp_for_type(etype):
    """Return max HP for a building entity type."""
    if etype == EntityType.CONVEYOR:
        return CONVEYOR_MAX_HP
    if etype == EntityType.HARVESTER:
        return HARVESTER_MAX_HP
    if etype == EntityType.BRIDGE:
        return BRIDGE_MAX_HP
    if etype == EntityType.ARMOURED_CONVEYOR:
        return 50
    return 20  # default


# ---------------------------------------------------------------------------
# Movement helper — uses Direction API correctly
# ---------------------------------------------------------------------------

def _move_toward_target(c, pos, target):
    """Greedy 8-directional move toward target with road-ahead. O(8).

    Empty tiles are NOT passable.  Must build road on the NEXT tile
    before stepping onto it (matches defender and miner patterns).
    Uses DIRECTION_BY_DELTA for reliable Direction lookup.
    """
    tx, ty = target
    best_delta = None
    best_dist = abs(pos.x - tx) + abs(pos.y - ty)

    for dx, dy in _ALL_8_DELTAS:
        nx, ny = pos.x + dx, pos.y + dy
        d = abs(nx - tx) + abs(ny - ty)
        if d < best_dist:
            best_dist = d
            best_delta = (dx, dy)

    if best_delta is None:
        return

    move_dir = DIRECTION_BY_DELTA.get(best_delta)
    if move_dir is None:
        return

    nx, ny = pos.x + best_delta[0], pos.y + best_delta[1]

    # Try to move if tile is already passable
    if c.get_move_cooldown() == 0:
        try:
            if c.can_move(move_dir):
                c.move(move_dir)
                return
        except GameError:
            pass

    # Tile not passable: build road on NEXT tile first
    if c.get_action_cooldown() == 0:
        next_pos = Position(nx, ny)
        try:
            if c.can_build_road(next_pos):
                c.build_road(next_pos)
                # Try to move onto freshly-built road this same tick
                if c.get_move_cooldown() == 0:
                    try:
                        if c.can_move(move_dir):
                            c.move(move_dir)
                            return
                    except GameError:
                        pass
                return  # Road built, move next tick
        except GameError:
            pass
