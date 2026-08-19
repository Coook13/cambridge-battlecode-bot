"""Quick miner role for the ultimate bot.

Early-game economy is intentionally simple:
- visible-local titanium targeting
- low-cost spoke exploration
- reverse-path conveyor stamping
- bridge-anchor return into the core

This version fixes:
1. Early miner crowding by relying on only 2 pre-defense miners in main.py
2. Two-tile oscillation / back-and-forth behaviour
3. Broken return-chain gaps when enemy transport gets cleared
4. Empty-tile detection bugs where tile_building_id() == 0 was treated as occupied
"""

from cambc import Controller, Direction, EntityType, Environment, GameError, Position

from constants import CARDINAL_DELTAS
from logger import log_event


_BRIDGE_SWITCH_CHEBYSHEV = 4
_MAX_STUCK_TURNS = 3
_MAX_BRIDGE_RETRIES = 3
_MAX_BOUNCE_TURNS = 2


_OPP_DELTA = {
    (0, -1): (0, 1),
    (0, 1): (0, -1),
    (1, 0): (-1, 0),
    (-1, 0): (1, 0),
}

_CARDINAL_DIRECTION_BY_DELTA = {
    (0, -1): Direction.NORTH,
    (1, 0): Direction.EAST,
    (0, 1): Direction.SOUTH,
    (-1, 0): Direction.WEST,
}


class MinerState:
    """Per-unit persistent state for the miner."""

    __slots__ = (
        "core_xy",
        "spoke_dx",
        "spoke_dy",
        "phase",
        "stuck_turns",
        "bounce_turns",
        "prev_pos",
        "last_pos",
        "last_move_delta",
        "ore_target",
        "path",
        "route_idx",
        "bridge_build_pos",
        "bridge_build_target",
        "bridge_anchor_xy",
        "bridge_retry_turns",
        "known_ore_blacklist",
        "blocked_anchors",
    )

    def __init__(self, core_pos: Position, bot_id: int):
        self.core_xy = (core_pos.x, core_pos.y)
        spoke_idx = bot_id % 4
        self.spoke_dx = CARDINAL_DELTAS[spoke_idx][0]
        self.spoke_dy = CARDINAL_DELTAS[spoke_idx][1]
        self.phase = "outward"
        self.stuck_turns = 0
        self.bounce_turns = 0
        self.prev_pos = None
        self.last_pos = None
        self.last_move_delta = None
        self.ore_target = None
        self.path = []
        self.route_idx = 0
        self.bridge_build_pos = None
        self.bridge_build_target = None
        self.bridge_anchor_xy = None
        self.bridge_retry_turns = 0
        self.known_ore_blacklist = set()
        self.blocked_anchors = set()


def run_miner(c: Controller, state: MinerState, local_map):
    """Execute one tick of the miner."""
    pos = c.get_position()
    cur_xy = (pos.x, pos.y)
    rnd = c.get_current_round()
    uid = c.get_id()

    if state.last_pos is None:
        _seed_outward_spoke_from_pos(state, cur_xy)

    if state.last_pos == cur_xy:
        state.stuck_turns += 1
    elif state.prev_pos == cur_xy:
        state.bounce_turns += 1
        state.stuck_turns = max(state.stuck_turns, state.bounce_turns)
    else:
        state.stuck_turns = 0
        state.bounce_turns = 0

    state.prev_pos = state.last_pos
    state.last_pos = cur_xy

    if state.phase == "outward":
        _miner_phase_outward(c, state, local_map, pos, cur_xy, rnd, uid)
    elif state.phase == "return":
        _miner_phase_return(c, state, pos, cur_xy, rnd, uid)


def _seed_outward_spoke_from_pos(state, cur_xy):
    """On the miner's first real tick, point its spoke away from the core."""
    cx, cy = state.core_xy
    dx = cur_xy[0] - cx
    dy = cur_xy[1] - cy

    if dx == 0 and dy == 0:
        return

    if abs(dx) >= abs(dy):
        state.spoke_dx = 1 if dx > 0 else -1
        state.spoke_dy = 0
    else:
        state.spoke_dx = 0
        state.spoke_dy = 1 if dy > 0 else -1


def _miner_phase_outward(c, state, local_map, pos, cur_xy, rnd, uid):
    """Walk outward, scan for nearby titanium, and build a harvester when adjacent."""
    _record_path(state.path, cur_xy)

    # Priority 1: if adjacent titanium is buildable, build immediately.
    for dx, dy in CARDINAL_DELTAS:
        hx, hy = pos.x + dx, pos.y + dy
        hp = Position(hx, hy)
        try:
            env = c.get_tile_env(hp)
            if not _is_income_ore(env):
                continue
            bid = c.get_tile_building_id(hp)
            if bid is not None and bid != 0:
                continue
            if c.get_action_cooldown() > 0:
                state.stuck_turns = 0
                state.bounce_turns = 0
                return
            if c.can_build_harvester(hp):
                c.build_harvester(hp)
                log_event(
                    rnd,
                    uid,
                    "miner",
                    f"({pos.x},{pos.y})",
                    "built_harvester",
                    hx=hx,
                    hy=hy,
                )
                state.known_ore_blacklist.discard((hx, hy))
                state.phase = "return"
                state.route_idx = max(0, len(state.path) - 1)
                state.stuck_turns = 0
                state.bounce_turns = 0
                state.bridge_anchor_xy = None
                state.bridge_retry_turns = 0
                return
        except GameError:
            continue

    if state.bounce_turns >= _MAX_BOUNCE_TURNS and state.ore_target is not None:
        state.known_ore_blacklist.add(state.ore_target)
        state.ore_target = None

    if (
        state.ore_target is None
        or state.stuck_turns >= _MAX_STUCK_TURNS
        or state.bounce_turns >= _MAX_BOUNCE_TURNS
    ):
        state.ore_target = _pick_best_ore_target(c, local_map, pos, state)

    if state.ore_target is not None:
        try:
            tp = Position(state.ore_target[0], state.ore_target[1])
            env = c.get_tile_env(tp)
            if not _is_income_ore(env):
                state.ore_target = None
            else:
                bid = c.get_tile_building_id(tp)
                if bid is not None and bid != 0:
                    state.ore_target = None
        except GameError:
            state.ore_target = None

    if c.get_move_cooldown() > 0 and c.get_action_cooldown() > 0:
        return

    if state.ore_target is not None:
        target_xy = state.ore_target
    else:
        target_xy = (
            cur_xy[0] + state.spoke_dx * 20,
            cur_xy[1] + state.spoke_dy * 20,
        )

    best_delta = _best_progress_step(c, cur_xy, target_xy, state)
    if best_delta is None:
        state.stuck_turns += 1

        if state.ore_target is not None and state.stuck_turns >= _MAX_STUCK_TURNS:
            state.known_ore_blacklist.add(state.ore_target)
            state.ore_target = None

        if state.ore_target is None and len(state.path) >= 8:
            try:
                map_w = c.get_map_width()
                map_h = c.get_map_height()
                on_edge = (
                    cur_xy[0] == 0
                    or cur_xy[0] == map_w - 1
                    or cur_xy[1] == 0
                    or cur_xy[1] == map_h - 1
                )
            except GameError:
                on_edge = False

            if on_edge:
                state.phase = "return"
                state.route_idx = max(0, len(state.path) - 1)
                state.bridge_anchor_xy = None
                state.bridge_retry_turns = 0
                return

        _handle_stuck_rotation(
            state,
            pos,
            rnd,
            uid,
            c.get_map_width(),
            c.get_map_height(),
        )
        return

    _try_move_cardinal(c, state, cur_xy, best_delta)


def _miner_phase_return(c, state, pos, cur_xy, rnd, uid):
    """Retrace outward path with conveyors, then anchor a bridge into the core."""
    cx, cy = state.core_xy

    if state.bridge_build_pos is not None:
        if c.get_action_cooldown() > 0:
            return
        if _build_pending_bridge(c, state, rnd, uid):
            _reset_for_next_trip(state, pos, rnd, uid)
            return

        state.bridge_retry_turns += 1
        if state.bridge_retry_turns >= _MAX_BRIDGE_RETRIES:
            if state.bridge_build_pos is not None:
                state.blocked_anchors.add(state.bridge_build_pos)
            log_event(
                rnd,
                uid,
                "miner",
                f"({cur_xy[0]},{cur_xy[1]})",
                "bridge_build_abandoned",
                retries=state.bridge_retry_turns,
            )
            state.bridge_build_pos = None
            state.bridge_build_target = None
            state.bridge_anchor_xy = None
            state.bridge_retry_turns = 0
        return

    if not state.path:
        _reset_for_next_trip(state, pos, rnd, uid)
        return

    chebyshev = max(abs(cur_xy[0] - cx), abs(cur_xy[1] - cy))

    if chebyshev <= _BRIDGE_SWITCH_CHEBYSHEV:
        _run_bridge_anchor_mode(c, state, pos, cur_xy, rnd, uid)
        return

    state.route_idx = max(0, min(state.route_idx, len(state.path) - 1))
    while state.route_idx > 0 and state.path[state.route_idx] != cur_xy:
        state.route_idx -= 1

    if state.route_idx > 0:
        next_xy = state.path[state.route_idx - 1]
    else:
        next_xy = (cx, cy)

    step_delta = _cardinal_step_delta(cur_xy, next_xy)
    if step_delta is None:
        if state.route_idx > 0:
            state.route_idx -= 1
        else:
            _reset_for_next_trip(state, pos, rnd, uid)
        return

    if _advance_with_stamp(c, cur_xy, next_xy, rnd, uid):
        state.stuck_turns = 0
        state.bounce_turns = 0
        if state.route_idx > 0:
            state.route_idx -= 1
        return

    state.stuck_turns += 1
    if state.stuck_turns >= _MAX_STUCK_TURNS * 3:
        _reset_for_next_trip(state, pos, rnd, uid)


def _run_bridge_anchor_mode(c, state, pos, cur_xy, rnd, uid):
    """Route to a valid bridge anchor, stamp conveyors into it, then bridge."""
    cx, cy = state.core_xy

    anchor_xy = state.bridge_anchor_xy
    if anchor_xy is None or not _anchor_matches_ring(anchor_xy, cx, cy):
        anchor_xy = _pick_nearest_bridge_anchor(
            c,
            state,
            cur_xy,
            cx,
            cy,
            c.get_map_width(),
            c.get_map_height(),
        )
        state.bridge_anchor_xy = anchor_xy

    if anchor_xy is None:
        _reset_for_next_trip(state, pos, rnd, uid)
        return

    if cur_xy != anchor_xy:
        next_xy = _next_bridge_anchor_step(cur_xy, anchor_xy, cx, cy)
        if next_xy is None:
            state.stuck_turns += 1
            if state.stuck_turns >= _MAX_STUCK_TURNS * 3:
                _reset_for_next_trip(state, pos, rnd, uid)
            return

        if _advance_with_stamp(c, cur_xy, next_xy, rnd, uid):
            state.stuck_turns = 0
            state.bounce_turns = 0
            return

        state.stuck_turns += 1
        if state.stuck_turns >= _MAX_STUCK_TURNS * 3:
            _reset_for_next_trip(state, pos, rnd, uid)
        return

    bridge_target = _find_bridge_target(cur_xy, cx, cy)
    if bridge_target is None:
        state.bridge_anchor_xy = None
        state.stuck_turns += 1
        return

    if c.get_action_cooldown() > 0 or c.get_move_cooldown() > 0:
        return

    cur_p = Position(cur_xy[0], cur_xy[1])

    try:
        bid = c.get_tile_building_id(cur_p)
    except GameError:
        bid = None

    if bid is not None and bid != 0:
        try:
            etype = c.get_entity_type(bid)
            team = c.get_team(bid)

            if team == c.get_team() and etype == EntityType.BRIDGE:
                try:
                    ex_target = c.get_bridge_target(bid)
                    if (ex_target.x, ex_target.y) == bridge_target:
                        _reset_for_next_trip(state, pos, rnd, uid)
                        return
                except GameError:
                    _reset_for_next_trip(state, pos, rnd, uid)
                    return

                state.blocked_anchors.add(cur_xy)
                state.bridge_anchor_xy = None
                state.stuck_turns += 1
                return

            if team != c.get_team() and etype in (
                EntityType.ROAD,
                EntityType.CONVEYOR,
                EntityType.SPLITTER,
                EntityType.BRIDGE,
                EntityType.ARMOURED_CONVEYOR,
            ):
                try:
                    if c.get_action_cooldown() == 0 and c.can_destroy(cur_p):
                        c.destroy(cur_p)
                except GameError:
                    pass
                return

        except GameError:
            pass

    if c.get_action_cooldown() > 0 or c.get_move_cooldown() > 0:
        return

    state.bridge_build_pos = cur_xy
    state.bridge_build_target = bridge_target
    state.bridge_retry_turns = 0

    if _step_off_for_bridge(c, state, cur_xy, cx, cy):
        log_event(
            rnd,
            uid,
            "miner",
            f"({cur_xy[0]},{cur_xy[1]})",
            "bridge_step_off",
            tx=bridge_target[0],
            ty=bridge_target[1],
        )
        return

    state.blocked_anchors.add(cur_xy)
    state.bridge_build_pos = None
    state.bridge_build_target = None
    state.bridge_anchor_xy = None
    state.stuck_turns += 1


def _lane_penalty(state, px, py, tx, ty):
    if state.spoke_dx == 0 and state.spoke_dy == -1 and ty > py:
        return 12
    if state.spoke_dx == 0 and state.spoke_dy == 1 and ty < py:
        return 12
    if state.spoke_dx == 1 and state.spoke_dy == 0 and tx < px:
        return 12
    if state.spoke_dx == -1 and state.spoke_dy == 0 and tx > px:
        return 12
    return 0


def _is_income_ore(env):
    return env == Environment.ORE_TITANIUM


def _pick_best_ore_target(c, local_map, pos, state):
    """Visible-local titanium targeting, inspired by version4.

    This deliberately avoids long global commits into bad map zones.
    """
    px, py = pos.x, pos.y
    best = None
    best_score = 10**18

    try:
        for tp in c.get_nearby_tiles():
            try:
                env = c.get_tile_env(tp)
                if env != Environment.ORE_TITANIUM:
                    continue

                bid = c.get_tile_building_id(tp)
                if bid is not None and bid != 0:
                    continue

                ore_xy = (tp.x, tp.y)
                if ore_xy in state.known_ore_blacklist:
                    continue

                score = (
                    (tp.x - px) * (tp.x - px)
                    + (tp.y - py) * (tp.y - py)
                    + _lane_penalty(state, px, py, tp.x, tp.y)
                )

                if score < best_score:
                    best_score = score
                    best = ore_xy
            except GameError:
                continue
    except GameError:
        pass

    return best


def _ordered_unique_deltas(items):
    out = []
    seen = set()
    for item in items:
        if item is None:
            continue
        if item in seen:
            continue
        seen.add(item)
        out.append(item)
    return out


def _best_progress_step(c, cur_xy, target_xy, state):
    """Choose a cheap progress step while avoiding direct oscillation."""
    back = _OPP_DELTA.get(state.last_move_delta)

    target_candidates = []
    if target_xy is not None:
        dx = target_xy[0] - cur_xy[0]
        dy = target_xy[1] - cur_xy[1]

        if abs(dx) >= abs(dy):
            primary = (1, 0) if dx > 0 else (-1, 0) if dx < 0 else None
            secondary = (0, 1) if dy > 0 else (0, -1) if dy < 0 else None
        else:
            primary = (0, 1) if dy > 0 else (0, -1) if dy < 0 else None
            secondary = (1, 0) if dx > 0 else (-1, 0) if dx < 0 else None

        target_candidates.extend([primary, secondary])

    spoke = (state.spoke_dx, state.spoke_dy)
    right = (-state.spoke_dy, state.spoke_dx)
    left = (state.spoke_dy, -state.spoke_dx)

    ordered = _ordered_unique_deltas(target_candidates + [spoke, right, left, back])

    for allow_back in (False, True):
        best = None
        best_key = None

        for cand in ordered:
            if cand is None:
                continue
            if not allow_back and cand == back:
                continue
            if not _can_progress_delta(c, cur_xy, cand, state):
                continue

            nx = cur_xy[0] + cand[0]
            ny = cur_xy[1] + cand[1]
            dist = abs(nx - target_xy[0]) + abs(ny - target_xy[1]) if target_xy is not None else 0

            key = (
                dist,
                1 if cand == back else 0,
                0 if cand == spoke else 1,
                nx,
                ny,
            )

            if best_key is None or key < best_key:
                best_key = key
                best = cand

        if best is not None:
            return best

    return None


def _handle_stuck_rotation(state, pos, rnd, uid, map_w=None, map_h=None):
    if state.stuck_turns < _MAX_STUCK_TURNS:
        return

    if map_w is not None and map_h is not None:
        cxm = map_w // 2
        cym = map_h // 2
        dx = cxm - pos.x
        dy = cym - pos.y

        if abs(dx) >= abs(dy):
            state.spoke_dx = 1 if dx > 0 else -1
            state.spoke_dy = 0
        else:
            state.spoke_dx = 0
            state.spoke_dy = 1 if dy > 0 else -1
    else:
        old_dx, old_dy = state.spoke_dx, state.spoke_dy
        state.spoke_dx, state.spoke_dy = -old_dy, old_dx

    state.stuck_turns = 0
    state.bounce_turns = 0
    state.ore_target = None
    state.last_move_delta = None

    log_event(
        rnd,
        uid,
        "miner",
        f"({pos.x},{pos.y})",
        "spoke_rotated",
        dx=state.spoke_dx,
        dy=state.spoke_dy,
    )


def _try_move_cardinal(c, state, cur_xy, best_delta):
    """Move one cardinal step, building a road if required."""
    move_dir = _CARDINAL_DIRECTION_BY_DELTA.get(best_delta)
    if move_dir is None:
        state.stuck_turns += 1
        return

    if c.get_move_cooldown() == 0:
        try:
            if c.can_move(move_dir):
                c.move(move_dir)
                state.last_move_delta = best_delta
                state.stuck_turns = 0
                return
        except GameError:
            pass

    if c.get_action_cooldown() == 0:
        nx = cur_xy[0] + best_delta[0]
        ny = cur_xy[1] + best_delta[1]
        next_pos = Position(nx, ny)
        try:
            if c.can_build_road(next_pos):
                c.build_road(next_pos)
                if c.get_move_cooldown() == 0:
                    try:
                        if c.can_move(move_dir):
                            c.move(move_dir)
                            state.last_move_delta = best_delta
                            state.stuck_turns = 0
                            return
                    except GameError:
                        pass
                state.stuck_turns = 0
                return
        except GameError:
            pass

    state.stuck_turns += 1


def _can_progress_delta(c, cur_xy, delta, state):
    move_dir = _CARDINAL_DIRECTION_BY_DELTA.get(delta)
    if move_dir is None:
        return False

    nx = cur_xy[0] + delta[0]
    ny = cur_xy[1] + delta[1]
    next_pos = Position(nx, ny)

    cx, cy = state.core_xy
    cur_cheb = max(abs(cur_xy[0] - cx), abs(cur_xy[1] - cy))
    next_cheb = max(abs(nx - cx), abs(ny - cy))

    if state.phase == "outward":
        if cur_cheb > 3 and next_cheb <= 3:
            return False
        if cur_cheb <= 3 and next_cheb < cur_cheb:
            return False

    try:
        if c.get_move_cooldown() == 0 and c.can_move(move_dir):
            return True
    except GameError:
        pass

    try:
        if c.get_action_cooldown() == 0 and c.can_build_road(next_pos):
            return True
    except GameError:
        pass

    return False


def _cardinal_step_delta(a_xy, b_xy):
    dx = b_xy[0] - a_xy[0]
    dy = b_xy[1] - a_xy[1]
    if dx != 0 and dy != 0:
        return None
    if dx > 0:
        return (1, 0)
    if dx < 0:
        return (-1, 0)
    if dy > 0:
        return (0, 1)
    if dy < 0:
        return (0, -1)
    return None


def _tile_has_conveyor(c, tile_xy, direction):
    tile_pos = Position(tile_xy[0], tile_xy[1])
    try:
        bid = c.get_tile_building_id(tile_pos)
        if bid is None or bid == 0:
            return False
        if c.get_entity_type(bid) != EntityType.CONVEYOR:
            return False
        return c.get_direction(bid) == direction
    except GameError:
        return False


def _place_conveyor_stamp(c, tile_xy, direction, rnd, uid):
    """Destroy replaceable transport/road on this tile and stamp the conveyor."""
    tile_pos = Position(tile_xy[0], tile_xy[1])

    try:
        bid = c.get_tile_building_id(tile_pos)
    except GameError:
        bid = None

    if bid is not None and bid != 0:
        try:
            etype = c.get_entity_type(bid)
            team = c.get_team(bid)

            if team != c.get_team():
                if etype in (
                    EntityType.ROAD,
                    EntityType.CONVEYOR,
                    EntityType.SPLITTER,
                    EntityType.BRIDGE,
                    EntityType.ARMOURED_CONVEYOR,
                ):
                    try:
                        if c.get_action_cooldown() == 0 and c.can_destroy(tile_pos):
                            c.destroy(tile_pos)
                    except GameError:
                        pass
                return False

            if etype == EntityType.CONVEYOR:
                try:
                    if c.get_direction(bid) == direction:
                        return True
                except GameError:
                    return False
                try:
                    if c.can_destroy(tile_pos):
                        c.destroy(tile_pos)
                except GameError:
                    pass
                return False

            elif etype == EntityType.ROAD:
                try:
                    if c.can_destroy(tile_pos):
                        c.destroy(tile_pos)
                except GameError:
                    pass
                return False

            elif etype == EntityType.BRIDGE:
                return True

            else:
                return False

        except GameError:
            return False

    try:
        if c.can_build_conveyor(tile_pos, direction):
            c.build_conveyor(tile_pos, direction)
            log_event(
                rnd,
                uid,
                "miner",
                f"({tile_xy[0]},{tile_xy[1]})",
                "stamp_conveyor",
                d=direction.name,
            )
            return True
    except GameError:
        pass

    return _tile_has_conveyor(c, tile_xy, direction)


def _advance_with_stamp(c, cur_xy, next_xy, rnd, uid):
    """Stamp current tile toward next_xy, then move. Returns True on move.

    Important fix:
    If the next tile is enemy transport, do NOT skip over it anymore.
    Instead, clear it first so the chain remains contiguous.
    """
    step_delta = _cardinal_step_delta(cur_xy, next_xy)
    if step_delta is None:
        return False

    move_dir = _CARDINAL_DIRECTION_BY_DELTA.get(step_delta)
    if move_dir is None:
        return False

    next_pos = Position(next_xy[0], next_xy[1])

    try:
        nbid = c.get_tile_building_id(next_pos)
    except GameError:
        nbid = None

    if nbid is not None and nbid != 0:
        try:
            if c.get_team(nbid) != c.get_team() and c.get_entity_type(nbid) in (
                EntityType.ROAD,
                EntityType.CONVEYOR,
                EntityType.SPLITTER,
                EntityType.BRIDGE,
                EntityType.ARMOURED_CONVEYOR,
            ):
                if c.get_action_cooldown() == 0:
                    try:
                        if c.can_destroy(next_pos):
                            c.destroy(next_pos)
                            log_event(
                                rnd,
                                uid,
                                "miner",
                                f"({cur_xy[0]},{cur_xy[1]})",
                                "cleared_enemy_transport_ahead",
                                tx=next_pos.x,
                                ty=next_pos.y,
                            )
                    except GameError:
                        pass
                return False
        except GameError:
            pass

    if not _tile_has_conveyor(c, cur_xy, move_dir):
        if c.get_action_cooldown() == 0:
            _place_conveyor_stamp(c, cur_xy, move_dir, rnd, uid)
        return False

    if c.get_move_cooldown() > 0:
        return False

    try:
        if c.can_move(move_dir):
            c.move(move_dir)
            return True
    except GameError:
        pass

    if c.get_action_cooldown() == 0:
        try:
            if c.can_build_road(next_pos):
                c.build_road(next_pos)
        except GameError:
            pass

    return False


def _record_path(path_list, xy):
    if not path_list:
        path_list.append(xy)
        return
    if path_list[-1] == xy:
        return
    if len(path_list) >= 2 and path_list[-2] == xy:
        path_list.pop()
        return

    start = max(0, len(path_list) - 8)
    for i in range(len(path_list) - 2, start - 1, -1):
        if path_list[i] == xy:
            del path_list[i + 1:]
            return

    path_list.append(xy)


def _find_bridge_target(cur_xy, cx, cy):
    """Closest core footprint tile reachable from cur_xy by a bridge."""
    best = None
    best_d2 = 10
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            tx = cx + dx
            ty = cy + dy
            d2 = (tx - cur_xy[0]) * (tx - cur_xy[0]) + (ty - cur_xy[1]) * (ty - cur_xy[1])
            if 0 < d2 <= 9 and (best is None or d2 < best_d2):
                best = (tx, ty)
                best_d2 = d2
    return best


def _anchor_matches_ring(anchor_xy, cx, cy):
    return max(abs(anchor_xy[0] - cx), abs(anchor_xy[1] - cy)) == _BRIDGE_SWITCH_CHEBYSHEV


def _bridge_anchor_candidates(cx, cy, map_w, map_h):
    out = []
    seen = set()
    for dy in (-1, 0, 1):
        for xy in ((cx - 4, cy + dy), (cx + 4, cy + dy)):
            if 0 <= xy[0] < map_w and 0 <= xy[1] < map_h and xy not in seen:
                seen.add(xy)
                out.append(xy)
    for dx in (-1, 0, 1):
        for xy in ((cx + dx, cy - 4), (cx + dx, cy + 4)):
            if 0 <= xy[0] < map_w and 0 <= xy[1] < map_h and xy not in seen:
                seen.add(xy)
                out.append(xy)
    return tuple(out)


def _pick_nearest_bridge_anchor(c, state, cur_xy, cx, cy, map_w, map_h):
    candidates = []
    for anchor_xy in _bridge_anchor_candidates(cx, cy, map_w, map_h):
        if _find_bridge_target(anchor_xy, cx, cy) is None:
            continue
        if anchor_xy in state.blocked_anchors:
            continue
        candidates.append(anchor_xy)

    if not candidates:
        return None

    return min(
        candidates,
        key=lambda t: (
            abs(t[0] - cur_xy[0]) + abs(t[1] - cur_xy[1]),
            abs(t[0] - cx) + abs(t[1] - cy),
            t[0],
            t[1],
        ),
    )


def _next_bridge_anchor_step(cur_xy, anchor_xy, cx, cy):
    """One cheap cardinal step toward the selected anchor without entering the barrier."""
    options = []
    dx = anchor_xy[0] - cur_xy[0]
    dy = anchor_xy[1] - cur_xy[1]

    if dx > 0:
        options.append((1, 0))
    elif dx < 0:
        options.append((-1, 0))

    if dy > 0:
        options.append((0, 1))
    elif dy < 0:
        options.append((0, -1))

    if len(options) == 2 and abs(dy) > abs(dx):
        options[0], options[1] = options[1], options[0]

    for step_dx, step_dy in options:
        nx = cur_xy[0] + step_dx
        ny = cur_xy[1] + step_dy
        if max(abs(nx - cx), abs(ny - cy)) <= 3:
            continue
        return (nx, ny)

    return None


def _step_off_for_bridge(c, state, cur_xy, cx, cy):
    """Step off the anchor so the bridge can be built on the saved tile next tick."""
    chebyshev = max(abs(cur_xy[0] - cx), abs(cur_xy[1] - cy))
    candidates = []

    for dx, dy in CARDINAL_DELTAS:
        nx = cur_xy[0] + dx
        ny = cur_xy[1] + dy
        new_cheb = max(abs(nx - cx), abs(ny - cy))
        if new_cheb <= 3:
            continue
        priority = 0 if new_cheb == chebyshev else 1
        candidates.append((priority, dx, dy))

    candidates.sort()

    for _, dx, dy in candidates:
        move_dir = _CARDINAL_DIRECTION_BY_DELTA.get((dx, dy))
        if move_dir is None:
            continue
        nx = cur_xy[0] + dx
        ny = cur_xy[1] + dy
        next_pos = Position(nx, ny)

        if c.get_move_cooldown() > 0:
            return False

        try:
            if c.can_move(move_dir):
                c.move(move_dir)
                state.stuck_turns = 0
                return True
        except GameError:
            pass

        if c.get_action_cooldown() == 0:
            try:
                if c.can_build_road(next_pos):
                    c.build_road(next_pos)
                    try:
                        if c.can_move(move_dir):
                            c.move(move_dir)
                            state.stuck_turns = 0
                            return True
                    except GameError:
                        pass
            except GameError:
                pass

    return False


def _build_pending_bridge(c, state, rnd, uid):
    """Attempt to build the pending bridge from the saved anchor tile."""
    bx, by = state.bridge_build_pos
    tx, ty = state.bridge_build_target
    bridge_pos = Position(bx, by)
    target_pos = Position(tx, ty)

    try:
        my_pos = c.get_position()
    except GameError:
        return False

    if my_pos.distance_squared(bridge_pos) > 2:
        return False

    try:
        bid = c.get_tile_building_id(bridge_pos)
    except GameError:
        bid = None

    if bid is not None and bid != 0:
        try:
            etype = c.get_entity_type(bid)
            team = c.get_team(bid)

            if team == c.get_team() and etype == EntityType.BRIDGE:
                try:
                    ex_target = c.get_bridge_target(bid)
                    if (ex_target.x, ex_target.y) == (tx, ty):
                        state.bridge_build_pos = None
                        state.bridge_build_target = None
                        state.bridge_anchor_xy = None
                        state.bridge_retry_turns = 0
                        return True
                except GameError:
                    pass
                return False

            if team == c.get_team() and etype in (
                EntityType.ROAD,
                EntityType.CONVEYOR,
                EntityType.SPLITTER,
            ):
                try:
                    if c.can_destroy(bridge_pos):
                        c.destroy(bridge_pos)
                except GameError:
                    pass
                return False
            else:
                return False

        except GameError:
            return False

    try:
        if c.can_build_bridge(bridge_pos, target_pos):
            c.build_bridge(bridge_pos, target_pos)
            log_event(
                rnd,
                uid,
                "miner",
                f"({bx},{by})",
                "built_bridge",
                tx=tx,
                ty=ty,
            )
            state.bridge_build_pos = None
            state.bridge_build_target = None
            state.bridge_anchor_xy = None
            state.bridge_retry_turns = 0
            return True
    except GameError:
        pass

    return False


def _reset_for_next_trip(state, pos, rnd, uid):
    old_dx, old_dy = state.spoke_dx, state.spoke_dy
    state.spoke_dx, state.spoke_dy = -old_dy, old_dx
    state.phase = "outward"
    state.stuck_turns = 0
    state.bounce_turns = 0
    state.ore_target = None
    state.path = []
    state.route_idx = 0
    state.bridge_build_pos = None
    state.bridge_build_target = None
    state.bridge_anchor_xy = None
    state.bridge_retry_turns = 0
    state.last_move_delta = None
    state.prev_pos = None
    state.last_pos = (pos.x, pos.y)
    if len(state.known_ore_blacklist) > 24:
        state.known_ore_blacklist.clear()

    log_event(
        rnd,
        uid,
        "miner",
        f"({pos.x},{pos.y})",
        "new_trip",
        dx=state.spoke_dx,
        dy=state.spoke_dy,
    )