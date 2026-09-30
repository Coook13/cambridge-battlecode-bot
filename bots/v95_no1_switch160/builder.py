from cambc import Controller, EntityType, Environment, GameError, Position
from constants import (ACTION_RADIUS_SQ, BARRIER_MAX_HP, CORE_MAX_HP,
                       LAUNCHER_MAX_HP, MSG_DEFENSE_COMPLETE,
                       encode_marker, marker_tile_candidates)
from logger import log_event


def _defender_get_nearby_buildings(c: Controller, state, rnd: int):
    """Cache nearby building ids once per defender tick."""
    if getattr(state, "_nearby_buildings_round", -1) != rnd:
        try:
            state._nearby_buildings_cache = list(c.get_nearby_buildings())
        except GameError:
            state._nearby_buildings_cache = []
        state._nearby_buildings_round = rnd
    return state._nearby_buildings_cache


def _state_local_map(state):
    return getattr(state, "local_map", None)


def _is_non_ore_passable(c: Controller, pos: Position) -> bool:
    try:
        if not c.is_tile_passable(pos):
            return False
    except GameError:
        return False

    try:
        env = c.get_tile_env(pos)
    except GameError:
        return False

    return env not in (Environment.ORE_TITANIUM, Environment.ORE_AXIONITE)


def _defense_complete_marker_value(state: "DefenderState") -> int:
    return encode_marker(MSG_DEFENSE_COMPLETE, state.cx, state.cy)


def _defender_launcher_wait_tile(state: "DefenderState"):
    """Match launcher queue tile selection used by launcher runtime logic."""
    cx, cy = state.cx, state.cy
    lx, ly = state.launcher_pos

    candidates = []
    for dx in (-1, 0, 1):
        for dy in (-1, 0, 1):
            wx, wy = cx + dx, cy + dy
            if not (0 <= wx < state.map_w and 0 <= wy < state.map_h):
                continue

            ddx = wx - lx
            ddy = wy - ly
            if ddx * ddx + ddy * ddy <= ACTION_RADIUS_SQ:
                candidates.append((wx, wy))

    if not candidates:
        return None

    return min(candidates, key=lambda t: abs(t[0] - cx) + abs(t[1] - cy))


def _defender_launcher_build_stand_tile(state: "DefenderState"):
    """Pick a safe adjacent stand tile for building/repairing launcher."""
    cx, cy = state.cx, state.cy
    lx, ly = state.launcher_pos
    wait_tile = _defender_launcher_wait_tile(state)

    def _collect(allow_wait: bool):
        out = []
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                sx, sy = cx + dx, cy + dy
                if not (0 <= sx < state.map_w and 0 <= sy < state.map_h):
                    continue
                if (sx, sy) == (lx, ly):
                    continue
                if _is_reserved_marker_tile(state, sx, sy):
                    continue
                if (not allow_wait) and wait_tile is not None and (sx, sy) == wait_tile:
                    continue

                ddx = sx - lx
                ddy = sy - ly
                if ddx * ddx + ddy * ddy <= ACTION_RADIUS_SQ:
                    out.append((sx, sy))
        return out

    candidates = _collect(allow_wait=False)
    if not candidates:
        candidates = _collect(allow_wait=True)
    if not candidates:
        return cx, cy

    return min(candidates, key=lambda t: abs(t[0] - cx) + abs(t[1] - cy))


def _defender_launcher_maintenance_stand_tile(state: "DefenderState"):
    """Pick a launcher-adjacent stand tile, preferring not to occupy queue tile."""
    return _defender_launcher_build_stand_tile(state)


def _defender_reassign_marker_walk_tasks(state: "DefenderState", tasks):
    """Keep marker tile clear by reassigning its barrier jobs to adjacent walk tiles."""
    kept = []
    deferred_barriers = []

    for task in tasks:
        wx, wy = task["walk"]
        if _is_reserved_marker_tile(state, wx, wy):
            deferred_barriers.extend(task["barriers"])
            continue
        kept.append({"walk": (wx, wy), "barriers": list(task["barriers"])})

    for bx, by in deferred_barriers:
        best_idx = None
        best_key = None
        for idx, task in enumerate(kept):
            wx, wy = task["walk"]
            ddx = wx - bx
            ddy = wy - by
            if ddx * ddx + ddy * ddy > ACTION_RADIUS_SQ:
                continue
            key = (abs(wx - bx) + abs(wy - by), idx)
            if best_key is None or key < best_key:
                best_key = key
                best_idx = idx

        if best_idx is not None and (bx, by) not in kept[best_idx]["barriers"]:
            kept[best_idx]["barriers"].append((bx, by))

    return kept


def _defender_pick_next_step(c, state: "DefenderState", pos, tx: int, ty: int, avoid_tiles=None):
    """Pick next step while avoiding reserved or forbidden tiles when possible."""
    avoid = set(avoid_tiles or ())

    def _forbidden(nx: int, ny: int) -> bool:
        if _is_reserved_marker_tile(state, nx, ny):
            return True
        if (nx, ny) in avoid and (nx, ny) != (tx, ty):
            return True
        return False

    primary_dir = pos.direction_to(Position(tx, ty))
    primary_step = pos.add(primary_dir)
    px, py = primary_step.x, primary_step.y

    if (tx, ty) == (px, py) or not _forbidden(px, py):
        return primary_dir, primary_step

    candidates = []
    for nx in range(pos.x - 1, pos.x + 2):
        for ny in range(pos.y - 1, pos.y + 2):
            if nx == pos.x and ny == pos.y:
                continue
            if not (0 <= nx < state.map_w and 0 <= ny < state.map_h):
                continue
            if _forbidden(nx, ny):
                continue

            score = abs(nx - tx) + abs(ny - ty)
            candidates.append(
                (score, abs(nx - state.cx) + abs(ny - state.cy), nx, ny))

    candidates.sort()
    for _, _, nx, ny in candidates:
        step_pos = Position(nx, ny)
        move_dir = pos.direction_to(step_pos)
        moved = pos.add(move_dir)
        if moved.x != nx or moved.y != ny:
            continue
        try:
            if c.can_move(move_dir):
                return move_dir, step_pos
        except GameError:
            continue

    # No safe immediate detour exists.
    if (tx, ty) == (px, py):
        return primary_dir, primary_step
    return None, None


# ---------------------------------------------------------------------------
# Defender state machine
# ---------------------------------------------------------------------------

def _compute_ring_tasks(cx, cy, map_w, map_h):
    """
    Generate the 16 walk tiles (Chebyshev-2 ring) and their assigned barrier
    tiles (Chebyshev-3 ring) in clockwise order starting from (cx, cy-2).
    Filters out-of-bounds walk/barrier tiles.
    """
    raw = [
        # Top edge
        ((cx,     cy - 2), [(cx,     cy - 3)]),
        ((cx + 1, cy - 2), [(cx + 1, cy - 3)]),
        # Top-right corner
        ((cx + 2, cy - 2), [(cx + 2, cy - 3),
         (cx + 3, cy - 3), (cx + 3, cy - 2)]),
        # Right edge
        ((cx + 2, cy - 1), [(cx + 3, cy - 1)]),
        ((cx + 2, cy),     [(cx + 3, cy)]),
        ((cx + 2, cy + 1), [(cx + 3, cy + 1)]),
        # Bottom-right corner
        ((cx + 2, cy + 2), [(cx + 3, cy + 2),
         (cx + 3, cy + 3), (cx + 2, cy + 3)]),
        # Bottom edge
        ((cx + 1, cy + 2), [(cx + 1, cy + 3)]),
        ((cx,     cy + 2), [(cx,     cy + 3)]),
        ((cx - 1, cy + 2), [(cx - 1, cy + 3)]),
        # Bottom-left corner
        ((cx - 2, cy + 2), [(cx - 2, cy + 3),
         (cx - 3, cy + 3), (cx - 3, cy + 2)]),
        # Left edge
        ((cx - 2, cy + 1), [(cx - 3, cy + 1)]),
        ((cx - 2, cy),     [(cx - 3, cy)]),
        ((cx - 2, cy - 1), [(cx - 3, cy - 1)]),
        # Top-left corner
        ((cx - 2, cy - 2), [(cx - 3, cy - 2),
         (cx - 3, cy - 3), (cx - 2, cy - 3)]),
        # Last top edge tile
        ((cx - 1, cy - 2), [(cx - 1, cy - 3)]),
    ]

    tasks = []
    for (wx, wy), barriers in raw:
        # Skip walk tiles that are out of bounds
        if not (0 <= wx < map_w and 0 <= wy < map_h):
            continue
        # Filter barrier tiles that are out of bounds
        valid_barriers = [(bx, by) for bx, by in barriers
                          if 0 <= bx < map_w and 0 <= by < map_h]
        tasks.append({"walk": (wx, wy), "barriers": valid_barriers})
    return tasks


class DefenderState:
    def __init__(self, core_pos, map_w: int, map_h: int, local_map=None):
        self.cx = core_pos.x
        self.cy = core_pos.y
        self.map_w = map_w
        self.map_h = map_h
        self.local_map = local_map
        self.phase = "init"
        self.ring_tasks = []
        self.ring_index = 0
        self.barriers_built = set()
        self.current_path = []  # list of (x, y) tuples for navigation
        self.built_roads = set()  # track roads we built on the ring
        self.actually_built_barriers = set()  # barriers successfully constructed
        # (x, y) of landing pad road (set during traverse)
        self.landing_pad = None
        # Guard phase fields
        self.all_barrier_positions = set()  # set of (bx, by)
        self.barrier_to_walk = {}  # {(bx,by): (wx,wy)}
        self.launcher_pos = (core_pos.x + 2, core_pos.y)
        self.repair_target = None  # (x, y) of thing to repair
        self.repair_stand = None   # (x, y) where to stand for repair
        self.repair_type = None    # "rebuild"|"heal"|"rebuild_launcher"|"heal_launcher"
        self.marker_tiles = marker_tile_candidates(
            self.cx,
            self.cy,
            self.map_w,
            self.map_h,
        )
        if self.marker_tiles:
            self.marker_tile = self.marker_tiles[0]
        else:
            self.marker_tiles = ((self.cx, self.cy),)
            self.marker_tile = self.marker_tiles[0]
        self._comm_marker = None
        self._comm_stand = None
        self._comm_failed_markers = set()
        self._nearby_buildings_round = -1
        self._nearby_buildings_cache = []


def _is_reserved_marker_tile(state: DefenderState, x: int, y: int) -> bool:
    return (x, y) == state.marker_tile


def run_defender(c: Controller, state: DefenderState):
    """Main tick function for the defender bot."""
    pos = c.get_position()
    rnd = c.get_current_round()
    uid = c.get_id()

    log_event(rnd, uid, "defender", f"({pos.x},{pos.y})",
              "defender_tick", phase=state.phase, ring_idx=state.ring_index)

    if state.phase == "init":
        _defender_phase_init(c, state, pos, rnd, uid)
    elif state.phase == "move_to_ring":
        _defender_phase_move_to_ring(c, state, pos, rnd, uid)
    elif state.phase == "traverse":
        _defender_phase_traverse(c, state, pos, rnd, uid)
    elif state.phase == "build_landing_pad":
        _defender_phase_build_landing_pad(c, state, pos, rnd, uid)
    elif state.phase == "move_to_launcher":
        _defender_phase_move_to_launcher(c, state, pos, rnd, uid)
    elif state.phase == "build_launcher":
        _defender_phase_build_launcher(c, state, pos, rnd, uid)
    elif state.phase == "return_to_core":
        _defender_phase_return_to_core(c, state, pos, rnd, uid)
    elif state.phase == "communicate":
        _defender_phase_communicate(c, state, pos, rnd, uid)
    elif state.phase == "communicate_return":
        _defender_phase_communicate_return(c, state, pos, rnd, uid)
    elif state.phase == "guard":
        _defender_phase_guard(c, state, pos, rnd, uid)


def _defender_phase_init(c, state, pos, rnd, uid):
    """Compute ring tasks and transition to move_to_ring."""
    w = state.map_w
    h = state.map_h
    state.ring_tasks = _compute_ring_tasks(state.cx, state.cy, w, h)
    state.ring_tasks = _defender_reassign_marker_walk_tasks(
        state,
        state.ring_tasks,
    )

    # Build barrier lookup tables for guard phase
    for task in state.ring_tasks:
        wx, wy = task["walk"]
        for bx, by in task["barriers"]:
            state.all_barrier_positions.add((bx, by))
            state.barrier_to_walk[(bx, by)] = (wx, wy)

    log_event(rnd, uid, "defender", f"({pos.x},{pos.y})",
              "ring_tasks_computed", count=len(state.ring_tasks),
              barriers=len(state.all_barrier_positions))

    # Build landing pad FIRST (before barriers) so the launched bot has
    # a clear outward tile. Then come back and build the barrier wall.
    state.phase = "build_landing_pad"
    log_event(rnd, uid, "defender", f"({pos.x},{pos.y})",
              "transition", to="build_landing_pad")


def _defender_try_destroy_marker_at(c, state, pos, tx, ty, rnd, uid):
    """Try to destroy a friendly marker at (tx, ty). Destroy is free, no cd."""
    if _is_reserved_marker_tile(state, tx, ty):
        return False

    target = Position(tx, ty)
    try:
        bid = c.get_tile_building_id(target)
        if bid is not None and bid != 0:
            if c.get_entity_type(bid) == EntityType.MARKER and c.get_team(bid) == c.get_team():
                c.destroy(target)
                log_event(rnd, uid, "defender", f"({pos.x},{pos.y})",
                          "destroyed_marker", mx=tx, my=ty)
                return True
    except GameError:
        pass
    return False


def _defender_move_toward(c, state, pos, tx, ty, rnd, uid, avoid_tiles=None):
    """
    Move one step toward (tx, ty). Returns True if moved.
    Computes the NEXT STEP tile via direction_to and handles building
    roads on that next tile if needed.
    """
    if c.get_move_cooldown() > 0:
        return False

    move_dir, next_step = _defender_pick_next_step(
        c,
        state,
        pos,
        tx,
        ty,
        avoid_tiles=avoid_tiles,
    )
    if move_dir is None or next_step is None:
        return False
    nx, ny = next_step.x, next_step.y

    if _is_reserved_marker_tile(state, nx, ny) and (tx, ty) != (nx, ny):
        log_event(rnd, uid, "defender", f"({pos.x},{pos.y})",
                  "marker_tile_avoidance_block", mx=nx, my=ny)
        return False

    # Check if next step tile is passable (core tiles, roads, conveyors)
    step_passable = _is_non_ore_passable(c, next_step)

    if not step_passable:
        # Need to build a road on the next step tile first
        if c.get_action_cooldown() == 0:
            if _is_reserved_marker_tile(state, nx, ny):
                return False
            try:
                if c.can_build_road(next_step):
                    c.build_road(next_step)
                    state.built_roads.add((nx, ny))
                    log_event(rnd, uid, "defender", f"({pos.x},{pos.y})",
                              "built_road", rx=nx, ry=ny)
                    # Now try to move
                    if c.can_move(move_dir):
                        c.move(move_dir)
                        log_event(rnd, uid, "defender", f"({pos.x},{pos.y})",
                                  "moved", d=move_dir.name)
                        return True
                else:
                    log_event(rnd, uid, "defender", f"({pos.x},{pos.y})",
                              "cant_build_road", rx=nx, ry=ny)
            except GameError as e:
                log_event(rnd, uid, "defender", f"({pos.x},{pos.y})",
                          "road_build_error", err=repr(e))
        return False

    # Next step is passable, just move
    try:
        if c.can_move(move_dir):
            c.move(move_dir)
            log_event(rnd, uid, "defender", f"({pos.x},{pos.y})",
                      "moved", d=move_dir.name)
            return True
        else:
            log_event(rnd, uid, "defender", f"({pos.x},{pos.y})",
                      "move_blocked", d=move_dir.name, tx=nx, ty=ny)
    except GameError as e:
        log_event(rnd, uid, "defender", f"({pos.x},{pos.y})",
                  "move_error", err=repr(e))
    return False


def _defender_phase_move_to_ring(c, state, pos, rnd, uid):
    """Navigate from core centre to the first walk tile."""
    if not state.current_path:
        state.phase = "traverse"
        log_event(rnd, uid, "defender", f"({pos.x},{pos.y})",
                  "transition", to="traverse")
        # Run traverse immediately this tick
        _defender_phase_traverse(c, state, pos, rnd, uid)
        return

    tx, ty = state.current_path[0]

    if pos.x == tx and pos.y == ty:
        state.current_path.pop(0)
        if not state.current_path:
            state.phase = "traverse"
            log_event(rnd, uid, "defender", f"({pos.x},{pos.y})",
                      "transition", to="traverse")
            _defender_phase_traverse(c, state, pos, rnd, uid)
        return

    _defender_move_toward(c, state, pos, tx, ty, rnd, uid)


def _defender_phase_traverse(c, state, pos, rnd, uid):
    """
    Main barrier-building loop. For each ring task:
    1. Move to the walk tile (building road if needed).
    2. Build all assigned barriers from that walk tile.
    3. Advance to next task.
    """
    if state.ring_index >= len(state.ring_tasks):
        state.phase = "move_to_launcher"
        log_event(rnd, uid, "defender", f"({pos.x},{pos.y})",
                  "transition", to="move_to_launcher")
        _defender_phase_move_to_launcher(c, state, pos, rnd, uid)
        return

    task = state.ring_tasks[state.ring_index]
    wx, wy = task["walk"]
    barriers = task["barriers"]

    if pos.x != wx or pos.y != wy:
        # Need to move to walk tile
        _defender_move_toward(c, state, pos, wx, wy, rnd, uid)
        return

    # We are at the walk tile. Build barriers.
    remaining = [b for b in barriers if b not in state.barriers_built]

    if remaining and c.get_action_cooldown() == 0:
        bx, by = remaining[0]
        target = Position(bx, by)
        try:
            if c.can_build_barrier(target):
                c.build_barrier(target)
                state.barriers_built.add((bx, by))
                state.actually_built_barriers.add((bx, by))
                log_event(rnd, uid, "defender", f"({pos.x},{pos.y})",
                          "built_barrier", bx=bx, by=by)
            else:
                # If a removable structure blocks this tile (often road/conveyor),
                # clear it now and retry barrier placement in a later round.
                cleared = False
                try:
                    if c.can_destroy(target):
                        c.destroy(target)
                        cleared = True
                        log_event(rnd, uid, "defender", f"({pos.x},{pos.y})",
                                  "cleared_for_barrier", bx=bx, by=by)
                except GameError:
                    pass

                if not cleared:
                    # Can't build here (wall, ore, occupied, etc.) — skip it
                    state.barriers_built.add((bx, by))
                    log_event(rnd, uid, "defender", f"({pos.x},{pos.y})",
                              "skip_barrier", bx=bx, by=by)
        except GameError as e:
            # Skip on error too
            state.barriers_built.add((bx, by))
            log_event(rnd, uid, "defender", f"({pos.x},{pos.y})",
                      "barrier_error", bx=bx, by=by, err=repr(e))
        return  # one action per tick

    if not remaining:
        # All barriers for this task done. Try to build road on next walk tile
        # to save a round (if action cd is 0 and next tile is adjacent).
        state.ring_index += 1
        state.barriers_built.clear()

        if state.ring_index >= len(state.ring_tasks):
            state.phase = "move_to_launcher"
            log_event(rnd, uid, "defender", f"({pos.x},{pos.y})",
                      "transition", to="move_to_launcher")
            return

        next_task = state.ring_tasks[state.ring_index]
        nwx, nwy = next_task["walk"]

        # Try to pre-build road on next tile if action cd allows and tile is adjacent
        if c.get_action_cooldown() == 0:
            next_target = Position(nwx, nwy)
            tile_passable = _is_non_ore_passable(c, next_target)
            if not tile_passable and not _is_reserved_marker_tile(state, nwx, nwy):
                try:
                    if c.can_build_road(next_target):
                        c.build_road(next_target)
                        state.built_roads.add((nwx, nwy))
                        log_event(rnd, uid, "defender", f"({pos.x},{pos.y})",
                                  "pre_built_road", rx=nwx, ry=nwy)
                except GameError:
                    pass

        # Move toward next walk tile if possible
        if c.get_move_cooldown() == 0:
            _defender_move_toward(c, state, pos, nwx, nwy, rnd, uid)


def _defender_phase_build_landing_pad(c, state, pos, rnd, uid):
    """Build a road at Chebyshev-4 BEFORE building barriers.

    Since no barriers exist yet, the defender can walk freely outward:
    core tiles → Chebyshev-2 (build road) → Chebyshev-3 (build road) →
    build road at Chebyshev-4 (the landing pad). Then return to core
    and start the barrier traverse.
    """
    cx, cy = state.cx, state.cy

    if state.landing_pad is not None:
        # Already built — return to core and start barrier traverse.
        if pos.x == cx and pos.y == cy:
            # Set up the traverse path
            state.current_path = []
            if state.ring_tasks:
                first_wx, first_wy = state.ring_tasks[0]["walk"]
                state.current_path = [(first_wx, first_wy)]
            state.phase = "move_to_ring"
            log_event(rnd, uid, "defender", f"({pos.x},{pos.y})",
                      "transition", to="move_to_ring")
            return
        _defender_move_toward(c, state, pos, cx, cy, rnd, uid)
        return

    w = state.map_w
    h = state.map_h

    # Candidates: (Chebyshev-4 pad tile, Chebyshev-3 road, Chebyshev-2 road)
    # The defender walks outward building roads, then builds pad at Chebyshev-4.
    candidates = [
        ((cx + 4, cy), (cx + 3, cy), (cx + 2, cy)),
        ((cx - 4, cy), (cx - 3, cy), (cx - 2, cy)),
        ((cx, cy + 4), (cx, cy + 3), (cx, cy + 2)),
        ((cx, cy - 4), (cx, cy - 3), (cx, cy - 2)),
        ((cx + 4, cy + 1), (cx + 3, cy + 1), (cx + 2, cy + 1)),
        ((cx + 4, cy - 1), (cx + 3, cy - 1), (cx + 2, cy - 1)),
        ((cx - 4, cy + 1), (cx - 3, cy + 1), (cx - 2, cy + 1)),
        ((cx - 4, cy - 1), (cx - 3, cy - 1), (cx - 2, cy - 1)),
        ((cx + 1, cy + 4), (cx + 1, cy + 3), (cx + 1, cy + 2)),
        ((cx - 1, cy + 4), (cx - 1, cy + 3), (cx - 1, cy + 2)),
        ((cx + 1, cy - 4), (cx + 1, cy - 3), (cx + 1, cy - 2)),
        ((cx - 1, cy - 4), (cx - 1, cy - 3), (cx - 1, cy - 2)),
    ]

    # Filter: all 3 tiles must be in bounds
    if not hasattr(state, '_lp_step'):
        valid = []
        for (px, py), (mx, my), (rx, ry) in candidates:
            if not (0 <= px < w and 0 <= py < h):
                continue
            if not (0 <= mx < w and 0 <= my < h):
                continue
            if not (0 <= rx < w and 0 <= ry < h):
                continue
            if _is_reserved_marker_tile(state, px, py):
                continue
            if _is_reserved_marker_tile(state, mx, my):
                continue
            if _is_reserved_marker_tile(state, rx, ry):
                continue
            valid.append(((px, py), (mx, my), (rx, ry)))
        if not valid:
            log_event(rnd, uid, "defender", f"({pos.x},{pos.y})",
                      "no_landing_pad_candidates")
            state.landing_pad = (cx + 3, cy)  # dummy
            return
        state._lp_valid = valid
        state._lp_step = 0
        state._lp_idx = 0

    valid = state._lp_valid
    if state._lp_idx >= len(valid):
        log_event(rnd, uid, "defender", f"({pos.x},{pos.y})",
                  "all_landing_pad_candidates_failed")
        state.landing_pad = (cx + 3, cy)  # dummy
        return

    (px, py), (mx, my), (rx, ry) = valid[state._lp_idx]

    # Step 0: Walk to Chebyshev-2 tile (building road via _defender_move_toward)
    if state._lp_step == 0:
        if pos.x != rx or pos.y != ry:
            _defender_move_toward(c, state, pos, rx, ry, rnd, uid)
            return
        state._lp_step = 1

    # Step 1: Build road at Chebyshev-3 tile
    if state._lp_step == 1:
        target = Position(mx, my)
        if c.get_action_cooldown() == 0:
            try:
                if c.can_build_road(target):
                    c.build_road(target)
                    state.built_roads.add((mx, my))
                    log_event(rnd, uid, "defender", f"({pos.x},{pos.y})",
                              "built_road_cheb3", rx=mx, ry=my)
                    state._lp_step = 2
                else:
                    # Try next candidate
                    state._lp_idx += 1
                    state._lp_step = 0
            except GameError:
                state._lp_idx += 1
                state._lp_step = 0
        return

    # Step 2: Walk to Chebyshev-3 road
    if state._lp_step == 2:
        if pos.x != mx or pos.y != my:
            _defender_move_toward(c, state, pos, mx, my, rnd, uid)
            return
        state._lp_step = 3

    # Step 3: Build road at Chebyshev-4 tile — the landing pad
    if state._lp_step == 3:
        target = Position(px, py)
        if c.get_action_cooldown() == 0:
            try:
                if c.can_build_road(target):
                    c.build_road(target)
                    state.landing_pad = (px, py)
                    state.launcher_pos = (rx, ry)
                    state.built_roads.add((px, py))
                    log_event(rnd, uid, "defender", f"({pos.x},{pos.y})",
                              "built_landing_pad", px=px, py=py,
                              lx=rx, ly=ry)
                else:
                    # Try next candidate
                    state._lp_idx += 1
                    state._lp_step = 0
            except GameError:
                state._lp_idx += 1
                state._lp_step = 0
        return


def _defender_phase_move_to_launcher(c, state, pos, rnd, uid):
    """Navigate to a tile adjacent to the chosen launcher tile to build it."""
    lx, ly = state.launcher_pos
    sx, sy = _defender_launcher_build_stand_tile(state)

    if pos.x == sx and pos.y == sy:
        state.phase = "build_launcher"
        log_event(rnd, uid, "defender", f"({pos.x},{pos.y})",
                  "transition", to="build_launcher")
        _defender_phase_build_launcher(c, state, pos, rnd, uid)
        return

    if pos.x == lx and pos.y == ly:
        # Never stand on the launcher tile for construction.
        _defender_move_toward(c, state, pos, sx, sy, rnd, uid)
        return

    _defender_move_toward(c, state, pos, sx, sy, rnd, uid)


def _defender_phase_build_launcher(c, state, pos, rnd, uid):
    """Destroy road at chosen launcher tile and build launcher from adjacent tile."""
    lx, ly = state.launcher_pos
    sx, sy = _defender_launcher_build_stand_tile(state)
    target = Position(lx, ly)

    if pos.x == lx and pos.y == ly:
        _defender_move_toward(c, state, pos, sx, sy, rnd, uid)
        return

    ddx = pos.x - lx
    ddy = pos.y - ly
    if ddx * ddx + ddy * ddy > ACTION_RADIUS_SQ:
        _defender_move_toward(c, state, pos, sx, sy, rnd, uid)
        return

    if c.get_action_cooldown() > 0:
        return

    # Destroy road at launcher position (free, no cd cost)
    try:
        if c.can_destroy(target):
            c.destroy(target)
            log_event(rnd, uid, "defender", f"({pos.x},{pos.y})",
                      "destroyed_road_for_launcher", lx=lx, ly=ly)
    except GameError:
        pass

    # Build launcher
    try:
        if c.can_build_launcher(target):
            c.build_launcher(target)
            log_event(rnd, uid, "defender", f"({pos.x},{pos.y})",
                      "built_launcher", lx=lx, ly=ly)
            state.phase = "return_to_core"
            log_event(rnd, uid, "defender", f"({pos.x},{pos.y})",
                      "transition", to="return_to_core")
        else:
            log_event(rnd, uid, "defender", f"({pos.x},{pos.y})",
                      "cant_build_launcher", lx=lx, ly=ly)
    except GameError as e:
        log_event(rnd, uid, "defender", f"({pos.x},{pos.y})",
                  "launcher_error", err=repr(e))


def _defender_phase_return_to_core(c, state, pos, rnd, uid):
    """Navigate back to core centre before entering guard."""
    cx, cy = state.cx, state.cy
    if abs(pos.x - cx) <= 1 and abs(pos.y - cy) <= 1:
        # Enter explicit communication phase before guarding.
        state.phase = "communicate"
        state._comm_marker = None
        state._comm_stand = None
        state._comm_failed_markers = set()
        log_event(rnd, uid, "defender", f"({pos.x},{pos.y})",
                  "transition", to="communicate")
        _defender_phase_communicate(c, state, pos, rnd, uid)
        return
    _defender_move_toward(c, state, pos, cx, cy, rnd, uid)


def _defender_launcher_launch_tile(state: DefenderState):
    """Queue/launch tile for launcher pickup, one step from launcher toward core."""
    lx, ly = state.launcher_pos
    dx = 0 if state.cx == lx else (1 if state.cx > lx else -1)
    dy = 0 if state.cy == ly else (1 if state.cy > ly else -1)
    return (lx + dx, ly + dy)


def _defender_pick_communicate_target(state: DefenderState, pos, launch_tile):
    """Pick marker tile and stand tile adjacent to it while avoiding launch tile."""
    failed = getattr(state, "_comm_failed_markers", set())
    best = None

    for mx, my in state.marker_tiles:
        if (mx, my) in failed:
            continue

        for sx in range(state.cx - 1, state.cx + 2):
            for sy in range(state.cy - 1, state.cy + 2):
                if not (0 <= sx < state.map_w and 0 <= sy < state.map_h):
                    continue
                if (sx, sy) == launch_tile:
                    continue
                if (sx, sy) == state.launcher_pos:
                    continue
                if max(abs(sx - mx), abs(sy - my)) > 1:
                    continue

                key = (
                    abs(sx - pos.x) + abs(sy - pos.y),
                    abs(mx - state.cx) + abs(my - state.cy),
                    abs(sx - state.cx) + abs(sy - state.cy),
                    mx,
                    my,
                    sx,
                    sy,
                )
                if best is None or key < best[0]:
                    best = (key, (mx, my), (sx, sy))

    if best is None:
        return None
    return best[1], best[2]


def _defender_phase_communicate(c, state, pos, rnd, uid):
    """Publish defense completion marker while avoiding launcher launch tile."""
    cx, cy = state.cx, state.cy
    launch_tile = _defender_launcher_launch_tile(state)

    if abs(pos.x - cx) > 1 or abs(pos.y - cy) > 1:
        _defender_move_toward(
            c,
            state,
            pos,
            cx,
            cy,
            rnd,
            uid,
            avoid_tiles={launch_tile},
        )
        return

    if state._comm_marker is None or state._comm_stand is None:
        target = _defender_pick_communicate_target(state, pos, launch_tile)
        if target is None:
            return
        state._comm_marker, state._comm_stand = target

    stand_x, stand_y = state._comm_stand
    if (pos.x, pos.y) != (stand_x, stand_y):
        _defender_move_toward(
            c,
            state,
            pos,
            stand_x,
            stand_y,
            rnd,
            uid,
            avoid_tiles={launch_tile},
        )
        return

    marker_val = _defense_complete_marker_value(state)
    mx, my = state._comm_marker
    mp = Position(mx, my)

    placed = False
    try:
        bid = c.get_tile_building_id(mp)
        if bid is None or bid == 0:
            if c.can_place_marker(mp):
                c.place_marker(mp, marker_val)
                placed = True
        else:
            same_team = c.get_team(bid) == c.get_team()
            is_marker = c.get_entity_type(bid) == EntityType.MARKER

            if same_team and is_marker:
                existing_val = c.get_marker_value(bid)
                # Backward compatibility with old one-byte marker value.
                if existing_val in (MSG_DEFENSE_COMPLETE, marker_val):
                    placed = True
                elif c.can_destroy(mp):
                    c.destroy(mp)
                    if c.can_place_marker(mp):
                        c.place_marker(mp, marker_val)
                        placed = True
            elif is_marker and c.can_destroy(mp):
                c.destroy(mp)
                if c.can_place_marker(mp):
                    c.place_marker(mp, marker_val)
                    placed = True
            # Never clobber non-marker buildings.
    except GameError:
        placed = False

    if not placed:
        state._comm_failed_markers.add((mx, my))
        state._comm_marker = None
        state._comm_stand = None
        return

    state.marker_tile = (mx, my)
    state._comm_marker = None
    state._comm_stand = None
    state._comm_failed_markers = set()
    log_event(rnd, uid, "defender", f"({pos.x},{pos.y})",
              "placed_defense_complete_marker", mx=mx, my=my, val=marker_val)

    state.phase = "communicate_return"
    log_event(rnd, uid, "defender", f"({pos.x},{pos.y})",
              "transition", to="communicate_return")


def _defender_phase_communicate_return(c, state, pos, rnd, uid):
    """Return to core centre after publish, then enter guard."""
    cx, cy = state.cx, state.cy
    launch_tile = _defender_launcher_launch_tile(state)

    if (pos.x, pos.y) != (cx, cy):
        _defender_move_toward(
            c,
            state,
            pos,
            cx,
            cy,
            rnd,
            uid,
            avoid_tiles={launch_tile},
        )
        return

    state.phase = "guard"
    log_event(rnd, uid, "defender", f"({pos.x},{pos.y})",
              "transition", to="guard")


# ---------------------------------------------------------------------------
# Defender guard phase — priority-based behavior tree
# ---------------------------------------------------------------------------

def _defender_phase_guard(c, state, pos, rnd, uid):
    """Active guard: monitor barriers, launcher, core. Repair as needed."""
    # If we have an active repair in progress, continue it
    if state.repair_target is not None:
        _defender_handle_repair(c, state, pos, rnd, uid)
        return

    # Priority 1: Check barriers for breaches (only visible tiles)
    breach = _defender_find_barrier_breach(c, state, pos)
    if breach is not None:
        bx, by, rtype, wx, wy = breach
        state.repair_target = (bx, by)
        state.repair_stand = (wx, wy)
        state.repair_type = rtype
        log_event(rnd, uid, "defender", f"({pos.x},{pos.y})",
                  "breach_detected", bx=bx, by=by, rtype=rtype, wx=wx, wy=wy)
        _defender_handle_repair(c, state, pos, rnd, uid)
        return

    # Priority 2: Check launcher
    launcher_issue = _defender_check_launcher(c, state, pos)
    if launcher_issue is not None:
        tx, ty, rtype, sx, sy = launcher_issue
        state.repair_target = (tx, ty)
        state.repair_stand = (sx, sy)
        state.repair_type = rtype
        log_event(rnd, uid, "defender", f"({pos.x},{pos.y})",
                  "launcher_issue", rtype=rtype)
        _defender_handle_repair(c, state, pos, rnd, uid)
        return

    # Priority 3: Heal core if damaged
    if c.get_action_cooldown() == 0:
        core_healed = _defender_try_heal_core(c, state, pos, rnd, uid)
        if core_healed:
            return

    # Keep launcher queue tile clear for economy bot pickup.
    wait_tile = _defender_launcher_wait_tile(state)
    if wait_tile is not None and (pos.x, pos.y) == wait_tile:
        _defender_move_toward(c, state, pos, state.cx, state.cy, rnd, uid)
        return

    # Idle: return to core centre if not there
    if pos.x != state.cx or pos.y != state.cy:
        _defender_move_toward(c, state, pos, state.cx, state.cy, rnd, uid)


def _defender_find_barrier_breach(c, state, pos):
    """
    Scan barrier positions for destroyed or damaged barriers.
    Only checks tiles in vision. Returns closest breach as
    (bx, by, repair_type, walk_x, walk_y) or None.
    """
    local_map = _state_local_map(state)
    my_team = c.get_team()
    best = None
    for bx, by in state.actually_built_barriers:
        if local_map is not None:
            if not local_map.is_visible(bx, by):
                continue

            rec = local_map.get_known_building(bx, by)
            if rec is None:
                wx, wy = state.barrier_to_walk[(bx, by)]
                dist = abs(bx - pos.x) + abs(by - pos.y)
                cand = (dist, bx, by, "rebuild", wx, wy)
                if best is None or cand[0] < best[0]:
                    best = cand
                continue

            if rec["team"] != my_team or rec["entity_type"] != EntityType.BARRIER:
                wx, wy = state.barrier_to_walk[(bx, by)]
                dist = abs(bx - pos.x) + abs(by - pos.y)
                cand = (dist, bx, by, "rebuild", wx, wy)
                if best is None or cand[0] < best[0]:
                    best = cand
                continue

            try:
                hp = c.get_hp(rec["id"])
            except GameError:
                continue
            if hp < BARRIER_MAX_HP:
                wx, wy = state.barrier_to_walk[(bx, by)]
                dist = abs(bx - pos.x) + abs(by - pos.y)
                cand = (dist, bx, by, "heal", wx, wy)
                if best is None or cand[0] < best[0]:
                    best = cand
            continue

        bp = Position(bx, by)
        try:
            if not c.is_in_vision(bp):
                continue
            bid = c.get_tile_building_id(bp)
            if bid is None or bid == 0:
                # Barrier destroyed — needs rebuild
                wx, wy = state.barrier_to_walk[(bx, by)]
                dist = abs(bx - pos.x) + abs(by - pos.y)
                cand = (dist, bx, by, "rebuild", wx, wy)
                if best is None or cand[0] < best[0]:
                    best = cand
            else:
                # Any non-allied or non-barrier occupant means the wall segment
                # is effectively breached and should be rebuilt when possible.
                if c.get_team(bid) != c.get_team() or c.get_entity_type(bid) != EntityType.BARRIER:
                    wx, wy = state.barrier_to_walk[(bx, by)]
                    dist = abs(bx - pos.x) + abs(by - pos.y)
                    cand = (dist, bx, by, "rebuild", wx, wy)
                    if best is None or cand[0] < best[0]:
                        best = cand
                    continue
                hp = c.get_hp(bid)
                if hp < BARRIER_MAX_HP:
                    # Barrier damaged — needs heal
                    wx, wy = state.barrier_to_walk[(bx, by)]
                    dist = abs(bx - pos.x) + abs(by - pos.y)
                    cand = (dist, bx, by, "heal", wx, wy)
                    if best is None or cand[0] < best[0]:
                        best = cand
        except GameError:
            continue

    if best is None:
        return None

    _, bx, by, rtype, wx, wy = best
    return (bx, by, rtype, wx, wy)


def _defender_check_launcher(c, state, pos):
    """
    Check if the launcher at (cx+2, cy) is damaged or destroyed.
    Returns (lx, ly, repair_type, stand_x, stand_y) or None.
    """
    def _can_afford_launcher() -> bool:
        try:
            ti, _ = c.get_launcher_cost()
            resources = c.get_global_resources()
            return resources[0] >= ti
        except GameError:
            return False

    lx, ly = state.launcher_pos
    lp = Position(lx, ly)
    sx, sy = _defender_launcher_build_stand_tile(state)
    local_map = _state_local_map(state)
    my_team = c.get_team()

    if local_map is not None:
        if not local_map.is_visible(lx, ly):
            return None

        rec = local_map.get_known_building(lx, ly)
        if rec is None:
            if _can_afford_launcher():
                return (lx, ly, "rebuild_launcher", sx, sy)
            return None

        if rec["team"] != my_team or rec["entity_type"] != EntityType.LAUNCHER:
            return (lx, ly, "rebuild_launcher", sx, sy)

        try:
            hp = c.get_hp(rec["id"])
            if hp < LAUNCHER_MAX_HP:
                return (lx, ly, "heal_launcher", sx, sy)
        except GameError:
            pass
        return None

    try:
        if not c.is_in_vision(lp):
            return None
        bid = c.get_tile_building_id(lp)
        if bid is None or bid == 0:
            # Launcher destroyed — check if we can afford to rebuild
            if _can_afford_launcher():
                return (lx, ly, "rebuild_launcher", sx, sy)
            return None
        else:
            if c.get_team(bid) != my_team or c.get_entity_type(bid) != EntityType.LAUNCHER:
                return (lx, ly, "rebuild_launcher", sx, sy)

            hp = c.get_hp(bid)
            if hp < LAUNCHER_MAX_HP:
                return (lx, ly, "heal_launcher", sx, sy)
    except GameError:
        pass
    return None


def _defender_try_heal_core(c, state, pos, rnd, uid):
    """Heal the core if it's damaged and we're at core centre. Returns True if healed."""
    my_team = c.get_team()
    core_id = None
    local_map = _state_local_map(state)
    if local_map is not None:
        rec = local_map.get_known_building(state.cx, state.cy)
        if rec is not None and rec["team"] == my_team and rec["entity_type"] == EntityType.CORE:
            core_id = rec["id"]

    if core_id is None:
        for bid in _defender_get_nearby_buildings(c, state, rnd):
            try:
                if c.get_entity_type(bid) == EntityType.CORE and c.get_team(bid) == my_team:
                    core_id = bid
                    break
            except GameError:
                continue

    if core_id is None:
        return False

    try:
        hp = c.get_hp(core_id)
    except GameError:
        return False

    max_hp = CORE_MAX_HP
    if hp < max_hp:
        core_tile = Position(state.cx, state.cy)
        try:
            if c.can_heal(core_tile):
                c.heal(core_tile)
                log_event(rnd, uid, "defender", f"({pos.x},{pos.y})",
                          "healed_core", hp=hp, max_hp=max_hp)
                return True
        except GameError:
            return False
    return False


def _defender_handle_repair(c, state, pos, rnd, uid):
    """Navigate to repair_stand and perform the repair action."""
    rx, ry = state.repair_target
    sx, sy = state.repair_stand
    rtype = state.repair_type

    # Navigate to stand tile if not there
    if pos.x != sx or pos.y != sy:
        _defender_move_toward(c, state, pos, sx, sy, rnd, uid)
        return

    # We're at the stand tile — perform the repair action
    if c.get_action_cooldown() > 0:
        return

    target = Position(rx, ry)

    if rtype == "rebuild":
        try:
            if c.can_build_barrier(target):
                c.build_barrier(target)
                log_event(rnd, uid, "defender", f"({pos.x},{pos.y})",
                          "repaired_barrier", bx=rx, by=ry)
                _defender_clear_repair(state)
            else:
                # Can't rebuild (wall, occupied, etc.) — give up
                log_event(rnd, uid, "defender", f"({pos.x},{pos.y})",
                          "repair_barrier_failed", bx=rx, by=ry)
                _defender_clear_repair(state)
        except GameError:
            _defender_clear_repair(state)

    elif rtype == "heal":
        try:
            bid = c.get_tile_building_id(target)
            if bid is not None and bid != 0:
                hp = c.get_hp(bid)
                max_hp = BARRIER_MAX_HP
                if hp >= max_hp:
                    # Fully healed — done
                    log_event(rnd, uid, "defender", f"({pos.x},{pos.y})",
                              "barrier_healed", bx=rx, by=ry)
                    _defender_clear_repair(state)
                    return
            if c.can_heal(target):
                c.heal(target)
                log_event(rnd, uid, "defender", f"({pos.x},{pos.y})",
                          "healing_barrier", bx=rx, by=ry)
            else:
                _defender_clear_repair(state)
        except GameError:
            _defender_clear_repair(state)

    elif rtype == "rebuild_launcher":
        # Destroy any debris first (free, no cd)
        try:
            if c.can_destroy(target):
                c.destroy(target)
        except GameError:
            pass
        # Check resources again before building
        try:
            ti, _ = c.get_launcher_cost()
            resources = c.get_global_resources()
            if resources[0] < ti:
                log_event(rnd, uid, "defender", f"({pos.x},{pos.y})",
                          "launcher_rebuild_no_resources")
                _defender_clear_repair(state)
                return
            if c.can_build_launcher(target):
                c.build_launcher(target)
                log_event(rnd, uid, "defender", f"({pos.x},{pos.y})",
                          "rebuilt_launcher")
                _defender_clear_repair(state)
            else:
                log_event(rnd, uid, "defender", f"({pos.x},{pos.y})",
                          "rebuild_launcher_failed")
                _defender_clear_repair(state)
        except GameError:
            _defender_clear_repair(state)

    elif rtype == "heal_launcher":
        try:
            bid = c.get_tile_building_id(target)
            if bid is not None and bid != 0:
                hp = c.get_hp(bid)
                max_hp = LAUNCHER_MAX_HP
                if hp >= max_hp:
                    log_event(rnd, uid, "defender", f"({pos.x},{pos.y})",
                              "launcher_healed")
                    _defender_clear_repair(state)
                    return
            if c.can_heal(target):
                c.heal(target)
                log_event(rnd, uid, "defender", f"({pos.x},{pos.y})",
                          "healing_launcher")
            else:
                _defender_clear_repair(state)
        except GameError:
            _defender_clear_repair(state)


def _defender_clear_repair(state):
    """Clear the active repair state."""
    state.repair_target = None
    state.repair_stand = None
    state.repair_type = None


# ---------------------------------------------------------------------------
# Healer bot — emergency core healer
# ---------------------------------------------------------------------------

class HealerState:
    def __init__(self, core_pos, local_map=None):
        self.cx = core_pos.x
        self.cy = core_pos.y
        self.local_map = local_map


def run_healer(c: Controller, state: HealerState):
    """Emergency healer: heal core each tick, self-destruct when core is full HP."""
    pos = c.get_position()
    rnd = c.get_current_round()
    uid = c.get_id()

    my_team = c.get_team()
    core_id = None
    local_map = _state_local_map(state)
    if local_map is not None:
        rec = local_map.get_known_building(state.cx, state.cy)
        if rec is not None and rec["team"] == my_team and rec["entity_type"] == EntityType.CORE:
            core_id = rec["id"]

    if core_id is None:
        for bid in c.get_nearby_buildings():
            try:
                if c.get_entity_type(bid) == EntityType.CORE and c.get_team(bid) == my_team:
                    core_id = bid
                    break
            except GameError:
                continue

    if core_id is None:
        log_event(rnd, uid, "healer", f"({pos.x},{pos.y})", "no_core_visible")
        return

    hp = c.get_hp(core_id)
    max_hp = CORE_MAX_HP

    if hp >= max_hp:
        log_event(rnd, uid, "healer", f"({pos.x},{pos.y})",
                  "core_full_hp_self_destruct")
        c.self_destruct()
        return

    if c.get_action_cooldown() == 0:
        core_tile = Position(state.cx, state.cy)
        try:
            if c.can_heal(core_tile):
                c.heal(core_tile)
                log_event(rnd, uid, "healer", f"({pos.x},{pos.y})",
                          "healed_core", hp=hp, max_hp=max_hp)
        except GameError:
            pass
