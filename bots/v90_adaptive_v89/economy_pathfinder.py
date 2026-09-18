import heapq
import time
from cambc import Controller, Direction, EntityType, Environment, GameError, Position
from constants import (ACTION_RADIUS_SQ, CARDINAL_DELTAS, PASSABLE_TILES, WALKABLE_TILES)
from logger import log_event

from economy_state import _ADJACENT_DELTAS_8


def _chebyshev(a_xy, b_xy) -> int:
    return max(abs(int(a_xy[0]) - int(b_xy[0])), abs(int(a_xy[1]) - int(b_xy[1])))


def _is_adjacent_step(a_xy, b_xy) -> bool:
    dx = abs(int(a_xy[0]) - int(b_xy[0]))
    dy = abs(int(a_xy[1]) - int(b_xy[1]))
    return (dx != 0 or dy != 0) and max(dx, dy) == 1


def _is_diagonal_step(a_xy, b_xy) -> bool:
    dx = abs(int(a_xy[0]) - int(b_xy[0]))
    dy = abs(int(a_xy[1]) - int(b_xy[1]))
    return dx == 1 and dy == 1


def _planner_step_cost(a_xy, b_xy) -> int:
    return 18 if _is_diagonal_step(a_xy, b_xy) else 10


def _planner_heuristic(a_xy, b_xy) -> int:
    dx = abs(int(a_xy[0]) - int(b_xy[0]))
    dy = abs(int(a_xy[1]) - int(b_xy[1]))
    diagonal = min(dx, dy)
    straight = max(dx, dy) - diagonal
    return (18 * diagonal) + (10 * straight)


def _manhattan(a, b):
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


def _astar_cardinal_start_session(start_xy, goal_xy, max_expansions: int):
    sx, sy = int(start_xy[0]), int(start_xy[1])
    gx, gy = int(goal_xy[0]), int(goal_xy[1])
    h0 = _planner_heuristic((sx, sy), (gx, gy))
    return {
        "start_xy": (sx, sy),
        "goal_xy": (gx, gy),
        "max_expansions": max(1, int(max_expansions)),
        "expansions_done": 0,
        "open_heap": [(h0 + (h0 >> 2), 0, sx, sy)],
        "parent": {},
        "g_score": {(sx, sy): 0},
        "closed": set(),
        "rounds": 0,
    }


def _astar_cardinal_continue_session(
    local_map,
    session: dict,
    step_expansions: int,
    walkable_only: bool = False,
    tile_passable_fn=None,
    tile_extra_cost_fn=None,
):
    start_xy = session["start_xy"]
    goal_xy = session["goal_xy"]
    gx, gy = goal_xy

    in_bounds = local_map.in_bounds
    get_tile = local_map.get

    if not in_bounds(gx, gy):
        return "unreachable", (), 0
    if tile_passable_fn is not None:
        if not tile_passable_fn(gx, gy):
            return "unreachable", (), 0
    else:
        goal_tile = get_tile(gx, gy)
        if walkable_only:
            if goal_tile not in WALKABLE_TILES:
                return "unreachable", (), 0
        elif goal_tile not in PASSABLE_TILES:
            return "unreachable", (), 0

    open_heap = session["open_heap"]
    parent = session["parent"]
    g_score = session["g_score"]
    closed = session["closed"]
    max_expansions = int(session["max_expansions"])
    expansions_done = int(session["expansions_done"])

    expanded_now = 0
    while (
        open_heap
        and expansions_done < max_expansions
        and expanded_now < max(1, int(step_expansions))
    ):
        _, g, x, y = heapq.heappop(open_heap)
        node = (x, y)
        if node in closed:
            continue
        closed.add(node)

        if x == gx and y == gy:
            session["expansions_done"] = expansions_done
            return "found", _reconstruct_path(parent, start_xy, goal_xy), expanded_now

        expansions_done += 1
        expanded_now += 1
        for dx, dy in _ADJACENT_DELTAS_8:
            nx = x + dx
            ny = y + dy
            n = (nx, ny)

            if n in closed or not in_bounds(nx, ny):
                continue
            if tile_passable_fn is not None:
                if not tile_passable_fn(nx, ny):
                    continue
            else:
                tile = get_tile(nx, ny)
                if walkable_only:
                    if tile not in WALKABLE_TILES:
                        continue
                elif tile not in PASSABLE_TILES:
                    continue

            step_cost = _planner_step_cost((x, y), (nx, ny))
            if tile_extra_cost_fn is not None:
                try:
                    extra = int(tile_extra_cost_fn(nx, ny))
                except (TypeError, ValueError):
                    extra = 0
                if extra > 0:
                    step_cost += extra

            ng = g + step_cost
            old_g = g_score.get(n)
            if old_g is not None and ng >= old_g:
                continue

            g_score[n] = ng
            parent[n] = node
            h = _planner_heuristic((nx, ny), (gx, gy))
            f = ng + h + (h >> 2)
            heapq.heappush(open_heap, (f, ng, nx, ny))

    session["expansions_done"] = expansions_done
    if open_heap and expansions_done < max_expansions:
        return "in_progress", (), expanded_now
    if open_heap:
        return "budget_exhausted", (), expanded_now
    return "unreachable", (), expanded_now


def _astar_cardinal_plan(
    local_map,
    start_xy,
    goal_xy,
    max_expansions: int,
    walkable_only: bool = False,
    tile_passable_fn=None,
    tile_extra_cost_fn=None,
    max_time_us: int | None = None,
    planner_stats: dict | None = None,
    cardinal_only: bool = False,
):
    if start_xy == goal_xy:
        return ()

    sx, sy = start_xy
    gx, gy = goal_xy

    in_bounds = local_map.in_bounds
    get_tile = local_map.get

    if not in_bounds(gx, gy):
        return ()
    if tile_passable_fn is not None:
        if not tile_passable_fn(gx, gy):
            return ()
    else:
        goal_tile = get_tile(gx, gy)
        if walkable_only:
            if goal_tile not in WALKABLE_TILES:
                return ()
        elif goal_tile not in PASSABLE_TILES:
            return ()

    # Weighted A*: faster convergence than strict optimal A* in this use-case.
    # f = g + h + h//4   (approx 1.25 * h)
    open_heap = []
    h0 = _planner_heuristic((sx, sy), (gx, gy))
    heapq.heappush(open_heap, (h0 + (h0 >> 2), 0, sx, sy))

    parent = {}
    g_score = {(sx, sy): 0}
    closed = set()
    _ = max_time_us

    move_deltas = CARDINAL_DELTAS if cardinal_only else _ADJACENT_DELTAS_8
    expansions = 0
    while open_heap and expansions < max_expansions:
        _, g, x, y = heapq.heappop(open_heap)
        node = (x, y)
        if node in closed:
            continue
        closed.add(node)

        if x == gx and y == gy:
            return _reconstruct_path(parent, start_xy, goal_xy)

        expansions += 1
        for dx, dy in move_deltas:
            nx = x + dx
            ny = y + dy
            n = (nx, ny)

            if n in closed or not in_bounds(nx, ny):
                continue
            if tile_passable_fn is not None:
                if not tile_passable_fn(nx, ny):
                    continue
            else:
                tile = get_tile(nx, ny)
                if walkable_only:
                    if tile not in WALKABLE_TILES:
                        continue
                elif tile not in PASSABLE_TILES:
                    continue

            step_cost = _planner_step_cost((x, y), (nx, ny))
            if tile_extra_cost_fn is not None:
                try:
                    extra = int(tile_extra_cost_fn(nx, ny))
                except (TypeError, ValueError):
                    extra = 0
                if extra > 0:
                    step_cost += extra

            ng = g + step_cost
            old_g = g_score.get(n)
            if old_g is not None and ng >= old_g:
                continue

            g_score[n] = ng
            parent[n] = node
            h = _planner_heuristic((nx, ny), (gx, gy))
            f = ng + h + (h >> 2)
            heapq.heappush(open_heap, (f, ng, nx, ny))

    if planner_stats is not None:
        planner_stats["timed_out"] = False

    return ()


def _reconstruct_path(parent, start_xy, goal_xy):
    cur = goal_xy
    out = []
    while cur != start_xy:
        out.append(cur)
        cur = parent.get(cur)
        if cur is None:
            return ()
    out.reverse()
    return tuple(out)


def _coverage_scanlines(v_min: int, v_max: int, radius: int, stride: int):
    if v_min > v_max:
        return ()

    # Very small interval: one center line is sufficient.
    if (v_max - v_min) <= (2 * radius):
        return ((v_min + v_max) // 2,)

    first = v_min + radius
    last = v_max - radius

    if first >= last:
        return ((v_min + v_max) // 2,)

    rows = [first]
    cur = first
    while (cur + stride) < last:
        cur += stride
        rows.append(cur)

    if rows[-1] != last:
        rows.append(last)

    return tuple(rows)


def _coverage_scanlines_edge_to_edge(v_min: int, v_max: int, stride: int):
    if v_min > v_max:
        return ()
    if v_min == v_max:
        return (v_min,)

    step = max(1, stride)
    rows = [v_min]
    cur = v_min
    while (cur + step) < v_max:
        cur += step
        rows.append(cur)

    if rows[-1] != v_max:
        rows.append(v_max)

    return tuple(rows)


def _coverage_row_endpoints(v_min: int, v_max: int, radius: int):
    if v_min > v_max:
        return (0, 0)

    if (v_max - v_min) <= (2 * radius):
        mid = (v_min + v_max) // 2
        return (mid, mid)

    return (v_min + radius, v_max - radius)


def _exploration_half_bounds(width: int, height: int, symmetry: str, start_xy):
    sx, sy = start_xy

    if width <= 0 or height <= 0:
        return 0, 0, 0, 0, "invalid"

    if symmetry == "VERTICAL":
        mid_x = (width - 1) // 2
        if sx <= mid_x:
            return 0, mid_x, 0, height - 1, "left"
        return mid_x, width - 1, 0, height - 1, "right"

    if symmetry == "HORIZONTAL":
        mid_y = (height - 1) // 2
        if sy <= mid_y:
            return 0, width - 1, 0, mid_y, "top"
        return 0, width - 1, mid_y, height - 1, "bottom"

    # ROTATIONAL/UNKNOWN: split by horizontal half.
    mid_y = (height - 1) // 2
    if sy <= mid_y:
        return 0, width - 1, 0, mid_y, "top"
    return 0, width - 1, mid_y, height - 1, "bottom"


def _format_plan_dump(start_xy, steps, max_nodes: int = 32):
    # Keep debug entries readable even if a replan becomes long.
    nodes = [start_xy, *steps]
    if len(nodes) <= max_nodes:
        return "->".join(f"({x},{y})" for x, y in nodes)

    head_keep = max_nodes // 2
    tail_keep = max_nodes - head_keep
    omitted = len(nodes) - max_nodes
    head_txt = "->".join(f"({x},{y})" for x, y in nodes[:head_keep])
    tail_txt = "->".join(f"({x},{y})" for x, y in nodes[-tail_keep:])
    return f"{head_txt}->...({omitted}_nodes_omitted)->{tail_txt}"


def _format_waypoint_dump(waypoints, max_nodes: int = 48):
    if len(waypoints) <= max_nodes:
        return "->".join(f"({x},{y})" for x, y in waypoints)

    head_keep = max_nodes // 2
    tail_keep = max_nodes - head_keep
    omitted = len(waypoints) - max_nodes
    head_txt = "->".join(f"({x},{y})" for x, y in waypoints[:head_keep])
    tail_txt = "->".join(f"({x},{y})" for x, y in waypoints[-tail_keep:])
    return f"{head_txt}->...({omitted}_waypoints_omitted)->{tail_txt}"
