import heapq
import math
import time

from cambc import Controller, Direction, EntityType, Environment, GameError, Position

from constants import (ACTION_RADIUS_SQ, CARDINAL_DELTAS, MAP_FREE,
                       MAP_OBSTACLE, MAP_ORE_AXIONITE, MAP_ORE_TITANIUM,
                       MAP_ROAD, PASSABLE_TILES, WALKABLE_TILES)
from logger import log_event

# Cardinal-only movement: Manhattan heuristic + 4-neighbor expansion.
_CARDINAL_DIRECTION_BY_DELTA = {
    (0, -1): Direction.NORTH,
    (1, 0): Direction.EAST,
    (0, 1): Direction.SOUTH,
    (-1, 0): Direction.WEST,
}

_ADJACENT_DELTAS_8 = (
    (-1, -1), (0, -1), (1, -1),
    (-1, 0),           (1, 0),
    (-1, 1),  (0, 1),  (1, 1),
)

_MOVE_DIRECTION_BY_DELTA_8 = {
    (-1, -1): Direction.NORTHWEST,
    (0, -1): Direction.NORTH,
    (1, -1): Direction.NORTHEAST,
    (-1, 0): Direction.WEST,
    (1, 0): Direction.EAST,
    (-1, 1): Direction.SOUTHWEST,
    (0, 1): Direction.SOUTH,
    (1, 1): Direction.SOUTHEAST,
}

_BRIDGE_TARGET_OFFSETS = tuple(
    sorted(
        (
            (dx, dy)
            for dx in range(-3, 4)
            for dy in range(-3, 4)
            if 0 < (dx * dx + dy * dy) <= 9
        ),
        key=lambda d: (d[0] * d[0] + d[1] * d[1],
                       abs(d[0]) + abs(d[1]), d[0], d[1]),
    )
)

_INDIRECT_LOCAL_SOURCE_MH_THRESHOLD = 8
_INDIRECT_SOURCE_NODES_PER_HISTORY_NETWORK = 8
_INDIRECT_SOURCE_NODES_PER_OTHER_NETWORK = 4
_NETWORK_SUCCESS_HISTORY_LIMIT = 8
_NETWORK_ASTAR_TIMEOUT_US = 2200
_NETWORK_SELECT_TIMEOUT_US = 0
_NETWORK_TIMEOUT_RETRY_COOLDOWN_ROUNDS = 96
_INDIRECT_TARGET_DENSITY_RADIUS_CHEB = 3
_INDIRECT_TARGET_MAX_BLOCKED_DENSITY = 0.68
_INDIRECT_TARGET_MIN_KNOWN_TILES = 24


_POST_LAUNCH_PHASES = {
    "launched",
    "launch_escape",
    "explore_generate_waypoints",
    "explore_replace_waypoint",
    "explore_plan_waypoint",
    "explore_follow_plan",
    "explore_done",
    "patrol_enter",
    "patrol_generate_waypoints",
    "patrol_replace_waypoint",
    "patrol_plan_waypoint",
    "patrol_follow_plan",
    "harvest_enter",
    "harvest_pick_ore",
    "harvest_pick_goal",
    "harvest_plan_goal",
    "harvest_follow_plan",
    "harvest_build",
    "network_wait",
    "network_select_candidate",
    "network_plan_path",
    "network_bridge_escape_check",
    "network_bridge_escape_execute",
    "conveyor_initialisation",
    "conveyor_execution",
    "conveyor_termination",
}

_NETWORK_BUILD_PHASES = {
    "network_wait",
    "network_select_candidate",
    "network_plan_path",
    "network_bridge_escape_check",
    "network_bridge_escape_execute",
    "conveyor_initialisation",
    "conveyor_execution",
    "conveyor_termination",
}

_HARVEST_TITANIUM_PHASES = {
    "harvest_enter",
    "harvest_pick_ore",
    "harvest_pick_goal",
    "harvest_plan_goal",
    "harvest_follow_plan",
    "harvest_build",
    "network_wait",
    "network_select_candidate",
    "network_plan_path",
    "network_bridge_escape_check",
    "network_bridge_escape_execute",
    "conveyor_initialisation",
    "conveyor_execution",
    "conveyor_termination",
}


class EconomyState:
    __slots__ = (
        "core_xy",
        "phase",
        "launcher_xy",
        "goal_xy",
        "plan_steps",
        "plan_index",
        "defer_step_once",
        "blocked_ticks",
        "last_xy",
        "issued_move_last_tick",
        "expected_xy_after_move",
        "wait_logged",
        "post_launch_logged",
        "explore_waypoints",
        "explore_waypoint_index",
        "explore_target_xy",
        "explore_done_logged",
        "patrol_unlocked",
        "harvest_ore_xy",
        "harvest_goal_xy",
        "harvest_blocked_ores",
        "network_wait_logged",
        "network_target",
        "network_path_nodes",
        "network_path_index",
        "network_escape_bridge_target",
        "network_highway_pending_harvester",
        "network_highway_active_harvester",
        "network_invalid_candidates",
        "network_timeout_candidates",
        "network_invalid_bridge_positions",
        "network_last_success_source_key",
        "network_last_success_source_conveyor",
        "network_success_source_history",
        "network_bridge_only_ti_links",
        "built_network_keys",
        "built_non_full_ti_networks",
        "seen_non_full_ti_networks",
        "last_symmetry",
        "last_symmetry_revision",
        "waypoint_refresh_round",
    )

    def __init__(self, core_pos: Position):
        self.core_xy: tuple[int, int] = (core_pos.x, core_pos.y)
        self.phase = "seek_launcher"
        self.launcher_xy: tuple[int, int] | None = None
        self.goal_xy: tuple[int, int] | None = None
        self.plan_steps: tuple[tuple[int, int], ...] = ()
        self.plan_index = 0
        self.defer_step_once = False
        self.blocked_ticks = 0

        # Launch detection: position change without our own move command.
        self.last_xy: tuple[int, int] | None = None
        self.issued_move_last_tick = False
        self.expected_xy_after_move: tuple[int, int] | None = None

        self.wait_logged = False
        self.post_launch_logged = False

        self.explore_waypoints: tuple[tuple[int, int], ...] = ()
        self.explore_waypoint_index = 0
        self.explore_target_xy: tuple[int, int] | None = None
        self.explore_done_logged = False
        self.patrol_unlocked = False

        self.harvest_ore_xy: tuple[int, int] | None = None
        self.harvest_goal_xy: tuple[int, int] | None = None
        self.harvest_blocked_ores = set()
        self.network_wait_logged = False
        self.network_target: dict | None = None
        self.network_path_nodes: tuple[tuple[int, int], ...] = ()
        self.network_path_index = 0
        self.network_escape_bridge_target: tuple[int, int] | None = None
        self.network_highway_pending_harvester: tuple[int, int] | None = None
        self.network_highway_active_harvester: tuple[int, int] | None = None
        self.network_invalid_candidates = set()
        self.network_timeout_candidates = {}
        self.network_invalid_bridge_positions = set()
        self.network_last_success_source_key: tuple[int, int] | None = None
        self.network_last_success_source_conveyor: tuple[int,
                                                         int] | None = None
        self.network_success_source_history: list[
            tuple[tuple[int, int], tuple[int, int] | None]
        ] = []
        self.network_bridge_only_ti_links: list[
            tuple[tuple[int, int], tuple[int, int], tuple[int, int]]
        ] = []
        self.built_network_keys = set()
        self.built_non_full_ti_networks: tuple[tuple[int, int], ...] = ()
        self.seen_non_full_ti_networks: tuple[tuple[int, int], ...] = ()

        self.last_symmetry: str | None = None
        self.last_symmetry_revision = -1
        self.waypoint_refresh_round: int | None = None


def run_economy(c: Controller, state: EconomyState, local_map):
    pos = c.get_position()
    cur_xy = (pos.x, pos.y)
    rnd = c.get_current_round()
    uid = c.get_id()

    _track_symmetry_revision(state, local_map, rnd, uid, cur_xy)
    _detect_external_relocation(state, cur_xy, rnd, uid)
    if state.phase in _POST_LAUNCH_PHASES:
        objective = _select_post_launch_objective(c, state, local_map)
        if objective == "explore":
            _run_post_launch_exploration(
                c,
                state,
                local_map,
                cur_xy,
                rnd,
                uid,
            )
        elif objective == "harvest_titanium":
            _run_titanium_harvest(
                c,
                state,
                local_map,
                cur_xy,
                rnd,
                uid,
            )
        return

    if state.launcher_xy is None:
        state.launcher_xy = _find_friendly_launcher_xy(c, local_map, cur_xy)
        if state.launcher_xy is None:
            return
        lx, ly = state.launcher_xy
        log_event(
            rnd,
            uid,
            "economy",
            f"({pos.x},{pos.y})",
            "economy_launcher_locked",
            lx=lx,
            ly=ly,
        )

    if state.goal_xy is None:
        queue_goal = _pick_launcher_wait_tile(
            local_map,
            state.core_xy,
            state.launcher_xy,
        )
        if queue_goal is not None:
            state.goal_xy = queue_goal
            goal_kind = "launcher_queue"
        else:
            state.goal_xy = _pick_launcher_adjacent_goal(
                local_map,
                cur_xy,
                state.launcher_xy,
                state.core_xy,
            )
            goal_kind = "launcher_adjacent"
        if state.goal_xy is None:
            return
        gx, gy = state.goal_xy
        state.phase = "plan_to_launcher"
        log_event(
            rnd,
            uid,
            "economy",
            f"({pos.x},{pos.y})",
            "economy_goal_selected",
            gx=gx,
            gy=gy,
            kind=goal_kind,
        )

    if state.phase == "plan_to_launcher":
        if cur_xy == state.goal_xy:
            state.phase = "wait_to_launch"
            state.plan_steps = ()
            state.plan_index = 0
            return

        # Keep one A* planning attempt per round to avoid late-round spikes.
        plan_budget = 512
        steps = _astar_cardinal_plan(
            local_map,
            cur_xy,
            state.goal_xy,
            max_expansions=plan_budget,
        )
        if not steps:
            # Path not available in current knowledge; retry next round.
            state.blocked_ticks += 1
            return

        state.plan_steps = steps
        state.plan_index = 0
        state.defer_step_once = True
        state.blocked_ticks = 0
        state.phase = "follow_plan"

        log_event(
            rnd,
            uid,
            "economy",
            f"({pos.x},{pos.y})",
            "economy_plan_ready",
            steps=len(steps),
            budget=plan_budget,
        )
        log_event(
            rnd,
            uid,
            "economy",
            f"({pos.x},{pos.y})",
            "economy_plan_dump",
            start=f"({cur_xy[0]},{cur_xy[1]})",
            goal=f"({state.goal_xy[0]},{state.goal_xy[1]})",
            path=_format_plan_dump(cur_xy, steps),
        )
        return

    if state.phase == "follow_plan":
        if cur_xy == state.goal_xy:
            state.phase = "wait_to_launch"
            state.plan_steps = ()
            state.plan_index = 0
            return

        if state.defer_step_once:
            state.defer_step_once = False
            return

        if state.plan_index >= len(state.plan_steps):
            state.phase = "plan_to_launcher"
            return

        nxt = state.plan_steps[state.plan_index]
        if _manhattan(cur_xy, nxt) != 1:
            # Drifted or stale plan; quickly replan from the current position.
            state.phase = "plan_to_launcher"
            return

        result = _execute_step_toward(c, local_map, cur_xy, nxt, rnd, uid)
        if result == "moved":
            state.issued_move_last_tick = True
            state.expected_xy_after_move = nxt
            state.plan_index += 1
            state.blocked_ticks = 0
            return

        if result == "built":
            state.blocked_ticks = 0
            return

        if result == "wait_cd":
            return

        # blocked: keep trying once, then replan if still blocked.
        state.blocked_ticks += 1
        if state.blocked_ticks >= 2:
            old_goal = state.goal_xy
            state.phase = "plan_to_launcher"
            state.blocked_ticks = 0
            state.plan_steps = ()
            state.plan_index = 0
            # Re-pick launcher-adjacent queue tile using freshest map state.
            state.goal_xy = None
            if old_goal is not None:
                log_event(
                    rnd,
                    uid,
                    "economy",
                    f"({pos.x},{pos.y})",
                    "economy_reselect_goal_after_block",
                    gx=old_goal[0],
                    gy=old_goal[1],
                )
        return

    if state.phase == "wait_to_launch":
        if not state.wait_logged:
            lx, ly = state.launcher_xy
            log_event(
                rnd,
                uid,
                "economy",
                f"({pos.x},{pos.y})",
                "economy_wait_to_launch",
                lx=lx,
                ly=ly,
            )
            state.wait_logged = True
        return


def _select_post_launch_objective(_c: Controller, state: EconomyState, local_map) -> str:
    if state.phase in _HARVEST_TITANIUM_PHASES:
        return "harvest_titanium"

    known_ti = _known_unharvested_titanium_unblocked(
        local_map,
        state.harvest_blocked_ores,
    )
    if known_ti:
        return "harvest_titanium"

    # Placeholder priority dispatcher: future objectives (network maintenance,
    # sabotage, etc.) can be inserted here ahead of exploration.
    return "explore"


def _run_titanium_harvest(
    c: Controller,
    state: EconomyState,
    local_map,
    cur_xy,
    rnd: int,
    uid: int,
):
    if state.phase not in _HARVEST_TITANIUM_PHASES:
        state.phase = "harvest_enter"
        state.harvest_ore_xy = None
        state.harvest_goal_xy = None
        state.plan_steps = ()
        state.plan_index = 0
        state.defer_step_once = False
        state.network_target = None
        state.network_path_nodes = ()
        state.network_path_index = 0
        state.network_wait_logged = False
        known_ti = _known_unharvested_titanium_unblocked(
            local_map,
            state.harvest_blocked_ores,
        )
        log_event(
            rnd,
            uid,
            "economy",
            f"({cur_xy[0]},{cur_xy[1]})",
            "economy_pause_explore_for_titanium",
            known_titanium=len(known_ti),
        )
        return

    if state.phase in _NETWORK_BUILD_PHASES:
        _run_titanium_network(
            c,
            state,
            local_map,
            cur_xy,
            rnd,
            uid,
        )
        return

    known_ti_all = _known_unharvested_titanium(local_map)
    known_ti_all_set = set(known_ti_all)
    state.harvest_blocked_ores.intersection_update(known_ti_all_set)
    known_ti = _known_unharvested_titanium_unblocked(
        local_map,
        state.harvest_blocked_ores,
    )

    if state.phase == "harvest_enter":
        state.phase = "harvest_pick_ore"
        return

    if state.phase == "harvest_pick_ore":
        if not known_ti:
            _resume_exploration_after_harvest(state)
            log_event(
                rnd,
                uid,
                "economy",
                f"({cur_xy[0]},{cur_xy[1]})",
                "economy_resume_explore_no_titanium",
                blocked_ores=len(state.harvest_blocked_ores),
            )
            return

        ore_xy = _pick_nearest_titanium_ore(cur_xy, known_ti)
        state.harvest_ore_xy = ore_xy
        state.harvest_goal_xy = None
        state.plan_steps = ()
        state.plan_index = 0
        state.defer_step_once = False
        state.phase = "harvest_pick_goal"
        log_event(
            rnd,
            uid,
            "economy",
            f"({cur_xy[0]},{cur_xy[1]})",
            "economy_harvest_target_selected",
            ox=ore_xy[0],
            oy=ore_xy[1],
        )
        return

    ore_xy = state.harvest_ore_xy
    if ore_xy is None or ore_xy not in known_ti_all_set:
        state.phase = "harvest_pick_ore"
        return
    if ore_xy in state.harvest_blocked_ores:
        state.harvest_ore_xy = None
        state.harvest_goal_xy = None
        state.phase = "harvest_pick_ore"
        return

    if state.phase == "harvest_pick_goal":
        goal_xy = _pick_titanium_adjacent_goal(local_map, ore_xy, cur_xy)
        if goal_xy is None:
            # No valid adjacent stand tile for this ore: blacklist and move on.
            state.harvest_blocked_ores.add(ore_xy)
            state.harvest_ore_xy = None
            state.harvest_goal_xy = None
            state.phase = "harvest_pick_ore"
            log_event(
                rnd,
                uid,
                "economy",
                f"({cur_xy[0]},{cur_xy[1]})",
                "economy_harvest_ore_blacklisted_no_adjacent",
                ox=ore_xy[0],
                oy=ore_xy[1],
                blocked_ores=len(state.harvest_blocked_ores),
            )
            return

        state.harvest_goal_xy = goal_xy
        state.plan_steps = ()
        state.plan_index = 0
        state.defer_step_once = False
        state.phase = "harvest_plan_goal"
        log_event(
            rnd,
            uid,
            "economy",
            f"({cur_xy[0]},{cur_xy[1]})",
            "economy_harvest_goal_selected",
            ox=ore_xy[0],
            oy=ore_xy[1],
            gx=goal_xy[0],
            gy=goal_xy[1],
        )
        return

    goal_xy = state.harvest_goal_xy
    if goal_xy is None:
        state.phase = "harvest_pick_goal"
        return

    if state.phase == "harvest_plan_goal":
        if cur_xy == goal_xy:
            state.phase = "harvest_build"
            return

        if not _is_titanium_goal_valid(local_map, ore_xy, goal_xy):
            state.harvest_goal_xy = None
            state.phase = "harvest_pick_goal"
            return

        # Keep one A* planning attempt per round to avoid same-round replans.
        plan_budget = 768
        steps = _astar_cardinal_plan(
            local_map,
            cur_xy,
            goal_xy,
            max_expansions=plan_budget,
        )

        if not steps:
            state.harvest_blocked_ores.add(ore_xy)
            state.harvest_ore_xy = None
            state.harvest_goal_xy = None
            state.phase = "harvest_pick_ore"
            log_event(
                rnd,
                uid,
                "economy",
                f"({cur_xy[0]},{cur_xy[1]})",
                "economy_harvest_ore_blacklisted_unreachable",
                ox=ore_xy[0],
                oy=ore_xy[1],
                gx=goal_xy[0],
                gy=goal_xy[1],
                blocked_ores=len(state.harvest_blocked_ores),
            )
            return

        state.plan_steps = steps
        state.plan_index = 0
        state.defer_step_once = True
        state.phase = "harvest_follow_plan"
        log_event(
            rnd,
            uid,
            "economy",
            f"({cur_xy[0]},{cur_xy[1]})",
            "economy_harvest_plan_ready",
            ox=ore_xy[0],
            oy=ore_xy[1],
            gx=goal_xy[0],
            gy=goal_xy[1],
            steps=len(steps),
            budget=plan_budget,
        )
        return

    if state.phase == "harvest_follow_plan":
        if cur_xy == goal_xy:
            state.plan_steps = ()
            state.plan_index = 0
            state.defer_step_once = False
            state.phase = "harvest_build"
            return

        if not _is_titanium_goal_valid(local_map, ore_xy, goal_xy):
            # Goal got blocked: stop and retry adjacent-goal selection next round.
            state.plan_steps = ()
            state.plan_index = 0
            state.defer_step_once = False
            state.harvest_goal_xy = None
            state.phase = "harvest_pick_goal"
            return

        if state.defer_step_once:
            state.defer_step_once = False
            return

        if state.plan_index >= len(state.plan_steps):
            state.phase = "harvest_plan_goal"
            return

        nxt = state.plan_steps[state.plan_index]
        if _manhattan(cur_xy, nxt) != 1:
            state.phase = "harvest_plan_goal"
            return

        result = _execute_step_toward(c, local_map, cur_xy, nxt, rnd, uid)
        if result == "moved":
            state.issued_move_last_tick = True
            state.expected_xy_after_move = nxt
            state.plan_index += 1
            return
        if result in ("built", "wait_cd"):
            return
        if result == "road_invalid":
            # Replan from current position on next round after knowledge update.
            state.plan_steps = ()
            state.plan_index = 0
            state.defer_step_once = False
            state.phase = "harvest_plan_goal"
            log_event(
                rnd,
                uid,
                "economy",
                f"({cur_xy[0]},{cur_xy[1]})",
                "economy_harvest_replan_after_road_invalid",
                nx=nxt[0],
                ny=nxt[1],
            )
            return

        # move_blocked: keep trying the same step; do not force immediate replan.
        return

    if state.phase == "harvest_build":
        if cur_xy != goal_xy:
            state.phase = "harvest_plan_goal"
            return

        ore_pos = Position(ore_xy[0], ore_xy[1])
        if c.get_action_cooldown() > 0:
            return

        try:
            if c.can_build_harvester(ore_pos):
                c.build_harvester(ore_pos)
                state.phase = "network_bridge_escape_check"
                state.network_wait_logged = False
                state.network_target = None
                state.network_path_nodes = ()
                state.network_path_index = 0
                state.network_escape_bridge_target = None
                state.network_highway_pending_harvester = ore_xy
                state.network_highway_active_harvester = None
                state.harvest_blocked_ores.discard(ore_xy)
                state.harvest_goal_xy = None
                state.harvest_ore_xy = None
                state.plan_steps = ()
                state.plan_index = 0
                state.defer_step_once = False
                log_event(
                    rnd,
                    uid,
                    "economy",
                    f"({cur_xy[0]},{cur_xy[1]})",
                    "economy_harvester_built",
                    ox=ore_xy[0],
                    oy=ore_xy[1],
                )
                log_event(
                    rnd,
                    uid,
                    "economy",
                    f"({cur_xy[0]},{cur_xy[1]})",
                    "economy_network_highway_bridge_escape_queued",
                    hx=ore_xy[0],
                    hy=ore_xy[1],
                )
                return
        except GameError:
            # Treat API failure as transient and retry next round.
            return

        # Build is currently illegal for a non-cooldown reason (often team
        # resources). Hold position and retry next round rather than oscillating
        # between adjacent goal tiles.
        return


def _run_post_launch_exploration(
    c: Controller,
    state: EconomyState,
    local_map,
    cur_xy,
    rnd: int,
    uid: int,
):
    if state.waypoint_refresh_round is not None and rnd >= state.waypoint_refresh_round:
        # Rebuild active waypoint mode after symmetry revision settles.
        state.phase = (
            "patrol_generate_waypoints"
            if state.patrol_unlocked
            else "explore_generate_waypoints"
        )
        state.plan_steps = ()
        state.plan_index = 0
        state.explore_target_xy = None
        state.explore_waypoint_index = 0
        state.waypoint_refresh_round = None
        log_event(
            rnd,
            uid,
            "economy",
            f"({cur_xy[0]},{cur_xy[1]})",
            "economy_refresh_waypoints_after_symmetry",
            symmetry=state.last_symmetry,
            revision=state.last_symmetry_revision,
        )
        return

    if state.phase == "launched":
        state.phase = "launch_escape"
        state.plan_steps = ()
        state.plan_index = 0
        state.explore_target_xy = None
        state.explore_waypoints = ()
        state.explore_waypoint_index = 0
        state.explore_done_logged = False
        if not state.post_launch_logged:
            log_event(
                rnd,
                uid,
                "economy",
                f"({cur_xy[0]},{cur_xy[1]})",
                "economy_post_launch_start",
            )
            state.post_launch_logged = True
        return

    if state.phase == "launch_escape":
        escape_result = _attempt_launch_escape(c, local_map, cur_xy, rnd, uid)
        if escape_result in ("moved", "built", "wait_cd"):
            return

        state.phase = "explore_generate_waypoints"
        state.plan_steps = ()
        state.plan_index = 0
        state.explore_target_xy = None
        state.explore_waypoints = ()
        state.explore_waypoint_index = 0
        log_event(
            rnd,
            uid,
            "economy",
            f"({cur_xy[0]},{cur_xy[1]})",
            "economy_launch_escape_complete",
            mode=escape_result,
        )
        return

    if state.phase == "patrol_enter":
        state.patrol_unlocked = True
        state.phase = "patrol_generate_waypoints"
        state.plan_steps = ()
        state.plan_index = 0
        state.defer_step_once = False
        state.explore_target_xy = None
        log_event(
            rnd,
            uid,
            "economy",
            f"({cur_xy[0]},{cur_xy[1]})",
            "economy_patrol_enter",
        )
        return

    if state.phase == "explore_generate_waypoints":
        vision_radius_sq = 20
        try:
            vision_radius_sq = c.get_vision_radius_sq()
        except GameError:
            pass

        waypoints, scan_meta = _build_exploration_waypoints(
            local_map,
            cur_xy,
            vision_radius_sq,
        )
        state.explore_waypoints = waypoints
        state.explore_waypoint_index = 0
        state.explore_target_xy = None
        state.phase = "explore_plan_waypoint"

        log_event(
            rnd,
            uid,
            "economy",
            f"({cur_xy[0]},{cur_xy[1]})",
            "economy_explore_waypoints_ready",
            count=len(waypoints),
            symmetry=scan_meta["symmetry"],
            half=scan_meta["half"],
            bounds=scan_meta["bounds"],
            vision_r_sq=scan_meta["vision_r_sq"],
            stride=scan_meta["stride"],
        )
        log_event(
            rnd,
            uid,
            "economy",
            f"({cur_xy[0]},{cur_xy[1]})",
            "economy_explore_waypoints_dump",
            waypoints=_format_waypoint_dump(waypoints),
        )
        return

    if state.phase == "patrol_generate_waypoints":
        vision_radius_sq = 20
        try:
            vision_radius_sq = c.get_vision_radius_sq()
        except GameError:
            pass

        waypoints, scan_meta = _build_patrol_waypoints(
            local_map,
            cur_xy,
            vision_radius_sq,
        )
        state.explore_waypoints = waypoints
        state.explore_waypoint_index = 0
        state.explore_target_xy = None
        state.phase = "patrol_plan_waypoint"

        log_event(
            rnd,
            uid,
            "economy",
            f"({cur_xy[0]},{cur_xy[1]})",
            "economy_patrol_waypoints_ready",
            count=len(waypoints),
            bounds=scan_meta["bounds"],
            vision_r_sq=scan_meta["vision_r_sq"],
            stride=scan_meta["stride"],
        )
        log_event(
            rnd,
            uid,
            "economy",
            f"({cur_xy[0]},{cur_xy[1]})",
            "economy_patrol_waypoints_dump",
            waypoints=_format_waypoint_dump(waypoints),
        )
        return

    if state.phase in ("explore_replace_waypoint", "patrol_replace_waypoint"):
        patrol_mode = state.phase == "patrol_replace_waypoint"
        plan_phase = "patrol_plan_waypoint" if patrol_mode else "explore_plan_waypoint"

        waypoints = list(state.explore_waypoints)
        idx = state.explore_waypoint_index
        target_xy = state.explore_target_xy

        if target_xy is None or not waypoints:
            state.phase = plan_phase
            return

        if idx < 0 or idx >= len(waypoints) or waypoints[idx] != target_xy:
            try:
                idx = waypoints.index(target_xy)
            except ValueError:
                state.phase = plan_phase
                return
            state.explore_waypoint_index = idx

        replacement = _find_replacement_waypoint(
            local_map,
            cur_xy,
            target_xy,
            tuple(waypoints),
            patrol_mode,
        )

        state.plan_steps = ()
        state.plan_index = 0
        state.explore_target_xy = None

        if replacement is not None:
            waypoints[idx] = replacement
            state.explore_waypoints = tuple(waypoints)
            state.explore_waypoint_index = idx
            state.phase = plan_phase
            log_event(
                rnd,
                uid,
                "economy",
                f"({cur_xy[0]},{cur_xy[1]})",
                "economy_patrol_waypoint_replaced" if patrol_mode else "economy_explore_waypoint_replaced",
                tx=target_xy[0],
                ty=target_xy[1],
                rx=replacement[0],
                ry=replacement[1],
                idx=idx,
            )
            return

        dropped = waypoints.pop(idx)
        if waypoints:
            state.explore_waypoints = tuple(waypoints)
            state.explore_waypoint_index = idx if idx < len(waypoints) else 0
            state.phase = plan_phase
            log_event(
                rnd,
                uid,
                "economy",
                f"({cur_xy[0]},{cur_xy[1]})",
                "economy_patrol_waypoint_dropped" if patrol_mode else "economy_explore_waypoint_dropped",
                tx=dropped[0],
                ty=dropped[1],
                remaining=len(waypoints),
            )
            return

        # Nothing left: finish this behavior branch immediately.
        state.explore_waypoints = ()
        state.explore_waypoint_index = 0
        state.defer_step_once = False
        if patrol_mode:
            state.phase = "patrol_generate_waypoints"
            log_event(
                rnd,
                uid,
                "economy",
                f"({cur_xy[0]},{cur_xy[1]})",
                "economy_patrol_cycle_complete",
                waypoints=0,
            )
        else:
            state.patrol_unlocked = True
            state.phase = "patrol_enter"
            log_event(
                rnd,
                uid,
                "economy",
                f"({cur_xy[0]},{cur_xy[1]})",
                "economy_explore_complete_enter_patrol",
                waypoints=0,
            )
        return

    if state.phase in ("explore_plan_waypoint", "patrol_plan_waypoint"):
        patrol_mode = state.phase == "patrol_plan_waypoint"
        plan_phase = "patrol_plan_waypoint" if patrol_mode else "explore_plan_waypoint"
        follow_phase = "patrol_follow_plan" if patrol_mode else "explore_follow_plan"

        waypoints = state.explore_waypoints
        idx = state.explore_waypoint_index

        while idx < len(waypoints) and waypoints[idx] == cur_xy:
            idx += 1
        state.explore_waypoint_index = idx

        if idx >= len(waypoints):
            if patrol_mode:
                state.explore_waypoint_index = 0
                state.explore_target_xy = None
                state.plan_steps = ()
                state.plan_index = 0
                state.defer_step_once = False
                state.phase = "patrol_generate_waypoints"
                log_event(
                    rnd,
                    uid,
                    "economy",
                    f"({cur_xy[0]},{cur_xy[1]})",
                    "economy_patrol_cycle_complete",
                    waypoints=len(waypoints),
                )
            else:
                state.patrol_unlocked = True
                state.explore_target_xy = None
                state.plan_steps = ()
                state.plan_index = 0
                state.defer_step_once = False
                state.phase = "patrol_enter"
                log_event(
                    rnd,
                    uid,
                    "economy",
                    f"({cur_xy[0]},{cur_xy[1]})",
                    "economy_explore_complete_enter_patrol",
                    waypoints=len(waypoints),
                )
            return

        target_xy = waypoints[idx]
        state.explore_target_xy = target_xy

        # Keep one A* planning attempt per round to avoid same-round replans.
        plan_budget = 768 if patrol_mode else 512
        steps = _astar_cardinal_plan(
            local_map,
            cur_xy,
            target_xy,
            max_expansions=plan_budget,
        )

        if not steps and cur_xy != target_xy:
            deferred_tag = (
                "economy_patrol_waypoint_deferred"
                if patrol_mode
                else "economy_explore_waypoint_deferred"
            )
            log_event(
                rnd,
                uid,
                "economy",
                f"({cur_xy[0]},{cur_xy[1]})",
                deferred_tag,
                tx=target_xy[0],
                ty=target_xy[1],
                idx=idx,
            )
            state.plan_steps = ()
            state.plan_index = 0
            state.explore_target_xy = target_xy
            state.phase = "patrol_replace_waypoint" if patrol_mode else "explore_replace_waypoint"
            return

        state.plan_steps = steps
        state.plan_index = 0
        state.phase = follow_phase

        log_event(
            rnd,
            uid,
            "economy",
            f"({cur_xy[0]},{cur_xy[1]})",
            "economy_patrol_plan_ready" if patrol_mode else "economy_explore_plan_ready",
            tx=target_xy[0],
            ty=target_xy[1],
            idx=idx,
            steps=len(steps),
            budget=plan_budget,
        )
        log_event(
            rnd,
            uid,
            "economy",
            f"({cur_xy[0]},{cur_xy[1]})",
            "economy_patrol_plan_dump" if patrol_mode else "economy_explore_plan_dump",
            start=f"({cur_xy[0]},{cur_xy[1]})",
            goal=f"({target_xy[0]},{target_xy[1]})",
            path=_format_plan_dump(cur_xy, steps),
        )
        return

    if state.phase in ("explore_follow_plan", "patrol_follow_plan"):
        patrol_mode = state.phase == "patrol_follow_plan"
        plan_phase = "patrol_plan_waypoint" if patrol_mode else "explore_plan_waypoint"

        target_xy = state.explore_target_xy
        if target_xy is None:
            state.phase = plan_phase
            return

        if cur_xy == target_xy:
            state.explore_waypoint_index += 1
            state.plan_steps = ()
            state.plan_index = 0
            state.phase = plan_phase
            return

        if state.plan_index >= len(state.plan_steps):
            state.phase = plan_phase
            return

        nxt = state.plan_steps[state.plan_index]
        if _manhattan(cur_xy, nxt) != 1:
            state.phase = plan_phase
            return

        result = _execute_step_toward(c, local_map, cur_xy, nxt, rnd, uid)
        if result == "moved":
            state.issued_move_last_tick = True
            state.expected_xy_after_move = nxt
            state.plan_index += 1
            return

        if result == "wait_cd":
            return

        if result == "built":
            return

        # Invalid movement (new obstacle/occupancy): replan next round.
        replan_tag = (
            "economy_patrol_replan_after_invalid_move"
            if patrol_mode
            else "economy_explore_replan_after_invalid_move"
        )
        log_event(
            rnd,
            uid,
            "economy",
            f"({cur_xy[0]},{cur_xy[1]})",
            replan_tag,
            tx=target_xy[0],
            ty=target_xy[1],
            nx=nxt[0],
            ny=nxt[1],
            idx=state.explore_waypoint_index,
        )
        state.plan_steps = ()
        state.plan_index = 0
        state.phase = plan_phase
        return

    if state.phase == "explore_done":
        # Legacy fallback: move terminal explore state into explicit patrol entry.
        state.patrol_unlocked = True
        state.phase = "patrol_enter"
        state.explore_target_xy = None
        state.explore_waypoint_index = 0
        state.plan_steps = ()
        state.plan_index = 0
        state.defer_step_once = False
        log_event(
            rnd,
            uid,
            "economy",
            f"({cur_xy[0]},{cur_xy[1]})",
            "economy_explore_done_legacy_patrol",
        )
        return


def _run_titanium_network(
    c: Controller,
    state: EconomyState,
    local_map,
    cur_xy,
    rnd: int,
    uid: int,
):
    known_ti = _known_unharvested_titanium_unblocked(
        local_map,
        state.harvest_blocked_ores,
    )

    if state.phase == "network_wait":
        if not state.network_wait_logged:
            log_event(
                rnd,
                uid,
                "economy",
                f"({cur_xy[0]},{cur_xy[1]})",
                "economy_network_behavior_pending",
            )
            state.network_wait_logged = True
            # Keep invalid target memory across rounds so failed bridge targets
            # are not retried after each new harvester build.
            state.network_escape_bridge_target = None
            state.network_highway_active_harvester = None
        state.phase = "network_select_candidate"
        return

    if state.phase == "network_select_candidate":
        phase_start_ns = time.perf_counter_ns()
        deadline_ns = None
        if _NETWORK_SELECT_TIMEOUT_US > 0:
            deadline_ns = phase_start_ns + (_NETWORK_SELECT_TIMEOUT_US * 1000)
        selection_stats = {}

        networks = _refresh_network_lists(
            state,
            local_map,
            deadline_ns=deadline_ns,
            refresh_stats=selection_stats,
        )
        candidate = _select_network_candidate(
            state,
            local_map,
            cur_xy,
            networks,
            rnd,
            deadline_ns=deadline_ns,
            selection_stats=selection_stats,
        )
        select_timed_out = bool(
            selection_stats.get("scan_timed_out", False)
            or selection_stats.get("ordering_timed_out", False)
            or selection_stats.get("timed_out", False)
        )

        if candidate is None:
            if select_timed_out:
                elapsed_us = (time.perf_counter_ns() - phase_start_ns) // 1000
                log_event(
                    rnd,
                    uid,
                    "economy",
                    f"({cur_xy[0]},{cur_xy[1]})",
                    "economy_network_select_timeout",
                    elapsed=elapsed_us,
                    scan_to=1 if selection_stats.get(
                        "scan_timed_out", False) else 0,
                    order_to=1 if selection_stats.get(
                        "ordering_timed_out", False) else 0,
                    cand_to=1 if selection_stats.get(
                        "timed_out", False) else 0,
                    built_non_full=len(state.built_non_full_ti_networks),
                    seen_non_full=len(state.seen_non_full_ti_networks),
                )
                return

            _network_fallback_to_next_objective(state, cur_xy, known_ti)
            log_event(
                rnd,
                uid,
                "economy",
                f"({cur_xy[0]},{cur_xy[1]})",
                "economy_network_no_candidate",
                built_non_full=len(state.built_non_full_ti_networks),
                seen_non_full=len(state.seen_non_full_ti_networks),
                invalid_candidates=len(state.network_invalid_candidates),
                timeout_candidates=_count_active_timeout_candidates(
                    state, rnd),
                invalid_bridge_positions=len(
                    state.network_invalid_bridge_positions),
                direct_total=selection_stats.get("direct_total", 0),
                direct_mem_reject=selection_stats.get(
                    "direct_reject_memory", 0),
                indirect_sources=selection_stats.get(
                    "indirect_sources", 0),
                indirect_density_reject=selection_stats.get(
                    "indirect_reject_density", 0),
                indirect_mem_reject=selection_stats.get(
                    "indirect_reject_memory", 0),
                indirect_viability_reject=selection_stats.get(
                    "indirect_reject_viability", 0),
                indirect_region_reject=selection_stats.get(
                    "indirect_reject_region", 0),
                indirect_access_reject=selection_stats.get(
                    "indirect_reject_access", 0),
                indirect_candidates=selection_stats.get(
                    "indirect_candidates", 0),
            )
            return

        state.network_target = candidate
        state.network_path_nodes = ()
        state.network_path_index = 0
        state.phase = "network_plan_path"
        log_event(
            rnd,
            uid,
            "economy",
            f"({cur_xy[0]},{cur_xy[1]})",
            "economy_network_target_selected",
            mode=candidate["mode"],
            bx=candidate["bridge_pos"][0],
            by=candidate["bridge_pos"][1],
            tx=candidate["bridge_target"][0],
            ty=candidate["bridge_target"][1],
        )
        return

    candidate = state.network_target

    if state.phase == "network_bridge_escape_check":
        highway_source = state.network_highway_pending_harvester
        if isinstance(highway_source, tuple) and len(highway_source) == 2:
            highway_source = (
                int(highway_source[0]),
                int(highway_source[1]),
            )
        else:
            highway_source = None

        if highway_source is not None:
            target_xy = _select_harvester_highway_bridge_target(
                local_map,
                highway_source,
                cur_xy,
                state.core_xy,
            )
        else:
            target_xy = _select_bridge_escape_target(
                local_map,
                cur_xy,
                state.core_xy,
            )

        if target_xy is None:
            if candidate is not None:
                _invalidate_network_candidate(state, candidate)
            state.network_target = None
            state.network_path_nodes = ()
            state.network_path_index = 0
            state.network_escape_bridge_target = None
            state.network_highway_pending_harvester = None
            state.network_highway_active_harvester = None
            state.phase = "network_select_candidate"
            log_event(
                rnd,
                uid,
                "economy",
                f"({cur_xy[0]},{cur_xy[1]})",
                "economy_network_highway_bridge_escape_none"
                if highway_source is not None
                else "economy_network_bridge_escape_none",
                hx=highway_source[0] if highway_source is not None else None,
                hy=highway_source[1] if highway_source is not None else None,
            )
            return

        state.network_escape_bridge_target = target_xy
        state.network_highway_active_harvester = highway_source
        state.network_highway_pending_harvester = None
        state.phase = "network_bridge_escape_execute"
        log_event(
            rnd,
            uid,
            "economy",
            f"({cur_xy[0]},{cur_xy[1]})",
            "economy_network_highway_bridge_escape_ready"
            if highway_source is not None
            else "economy_network_bridge_escape_ready",
            tx=target_xy[0],
            ty=target_xy[1],
            hx=highway_source[0] if highway_source is not None else None,
            hy=highway_source[1] if highway_source is not None else None,
        )
        return

    if state.phase == "network_bridge_escape_execute":
        target_xy = state.network_escape_bridge_target
        if not (isinstance(target_xy, tuple) and len(target_xy) == 2):
            state.phase = "network_bridge_escape_check"
            return

        bridge_result = _build_bridge_on_tile(
            c,
            cur_xy,
            target_xy,
            rnd,
            uid,
        )
        if bridge_result == "wait_cd":
            return

        highway_source = state.network_highway_active_harvester
        if isinstance(highway_source, tuple) and len(highway_source) == 2:
            highway_source = (
                int(highway_source[0]),
                int(highway_source[1]),
            )
        else:
            highway_source = None

        if bridge_result in ("built", "already_built"):
            if highway_source is not None:
                _record_bridge_only_titanium_link(
                    state,
                    highway_source,
                    cur_xy,
                    target_xy,
                )

            state.network_target = None
            state.network_path_nodes = ()
            state.network_path_index = 0
            state.network_escape_bridge_target = None
            state.network_highway_active_harvester = None
            state.network_highway_pending_harvester = None
            _network_fallback_to_next_objective(state, cur_xy, known_ti)
            log_event(
                rnd,
                uid,
                "economy",
                f"({cur_xy[0]},{cur_xy[1]})",
                "economy_network_highway_bridge_escape_complete"
                if highway_source is not None
                else "economy_network_bridge_escape_complete",
                tx=target_xy[0],
                ty=target_xy[1],
                hx=highway_source[0] if highway_source is not None else None,
                hy=highway_source[1] if highway_source is not None else None,
                bridge_only_count=len(state.network_bridge_only_ti_links),
            )
            return

        if candidate is not None:
            _invalidate_network_candidate(state, candidate)
        state.network_target = None
        state.network_path_nodes = ()
        state.network_path_index = 0
        state.network_escape_bridge_target = None
        state.network_highway_active_harvester = None
        state.network_highway_pending_harvester = None
        state.phase = "network_select_candidate"
        log_event(
            rnd,
            uid,
            "economy",
            f"({cur_xy[0]},{cur_xy[1]})",
            "economy_network_highway_bridge_escape_invalid"
            if highway_source is not None
            else "economy_network_bridge_escape_invalid",
            tx=target_xy[0],
            ty=target_xy[1],
            hx=highway_source[0] if highway_source is not None else None,
            hy=highway_source[1] if highway_source is not None else None,
        )
        return

    candidate = state.network_target
    if candidate is None:
        state.phase = "network_select_candidate"
        return

    if state.phase == "network_plan_path":
        phase_start_ns = time.perf_counter_ns()
        goal_xy = candidate["bridge_pos"]
        plan_timed_out = False

        if candidate.get("mode") == "indirect":
            known_bridge_targets = _known_friendly_bridge_targets(local_map)
            target_xy = candidate.get("bridge_target")
            if (
                isinstance(target_xy, tuple)
                and len(target_xy) == 2
                and not _bridge_target_region_clear(target_xy, known_bridge_targets, radius_cheb=0)
            ):
                alt_target = _pick_alternate_bridge_target(
                    local_map,
                    candidate,
                    goal_xy,
                    known_bridge_targets,
                )
                if alt_target is not None:
                    old_target = candidate["bridge_target"]
                    candidate["bridge_target"] = alt_target
                    log_event(
                        rnd,
                        uid,
                        "economy",
                        f"({cur_xy[0]},{cur_xy[1]})",
                        "economy_network_bridge_target_retarget",
                        stage="planning",
                        ox=old_target[0],
                        oy=old_target[1],
                        tx=alt_target[0],
                        ty=alt_target[1],
                    )
                else:
                    _invalidate_network_candidate(state, candidate)
                    state.network_target = None
                    state.network_path_nodes = ()
                    state.network_path_index = 0
                    state.phase = "network_select_candidate"
                    log_event(
                        rnd,
                        uid,
                        "economy",
                        f"({cur_xy[0]},{cur_xy[1]})",
                        "economy_network_bridge_target_congested",
                        bx=goal_xy[0],
                        by=goal_xy[1],
                    )
                    return

        if cur_xy == goal_xy:
            steps = ()
            plan_budget = 0
        else:
            def passable_fn(x: int, y: int) -> bool:
                return _is_conveyor_planner_passable(
                    local_map,
                    x,
                    y,
                    goal_xy,
                )

            # Keep one A* planning attempt per round to avoid same-round replans.
            goal_mh = _manhattan(cur_xy, goal_xy)
            plan_budget = max(960, min(2048, goal_mh * 20))
            plan_stats = {}
            elapsed_us = (time.perf_counter_ns() - phase_start_ns) // 1000
            remaining_us = _NETWORK_ASTAR_TIMEOUT_US - int(elapsed_us)
            if remaining_us <= 0:
                steps = ()
                plan_timed_out = True
            else:
                steps = _astar_cardinal_plan(
                    local_map,
                    cur_xy,
                    goal_xy,
                    max_expansions=plan_budget,
                    tile_passable_fn=passable_fn,
                    max_time_us=remaining_us,
                    planner_stats=plan_stats,
                )
                plan_timed_out = bool(plan_stats.get("timed_out", False))

        if cur_xy != goal_xy and not steps:
            if (
                not plan_timed_out
                and _is_bridge_escape_source_tile(local_map, cur_xy[0], cur_xy[1])
            ):
                state.network_path_nodes = ()
                state.network_path_index = 0
                state.network_escape_bridge_target = None
                state.phase = "network_bridge_escape_check"
                log_event(
                    rnd,
                    uid,
                    "economy",
                    f"({cur_xy[0]},{cur_xy[1]})",
                    "economy_network_plan_blocked_try_bridge_escape",
                    bx=goal_xy[0],
                    by=goal_xy[1],
                )
                return

            _invalidate_network_candidate(
                state,
                candidate,
                timeout_round=rnd if plan_timed_out else None,
            )
            state.network_target = None
            state.network_path_nodes = ()
            state.network_path_index = 0
            state.network_escape_bridge_target = None
            state.phase = "network_select_candidate"
            log_event(
                rnd,
                uid,
                "economy",
                f"({cur_xy[0]},{cur_xy[1]})",
                "economy_network_plan_unreachable",
                bx=goal_xy[0],
                by=goal_xy[1],
                timeout=1 if plan_timed_out else 0,
            )
            return

        state.network_path_nodes = (cur_xy, *steps)
        state.network_path_index = 0
        if len(state.network_path_nodes) <= 1:
            state.phase = "conveyor_termination"
        else:
            state.phase = "conveyor_initialisation"

        log_event(
            rnd,
            uid,
            "economy",
            f"({cur_xy[0]},{cur_xy[1]})",
            "economy_network_plan_ready",
            bx=goal_xy[0],
            by=goal_xy[1],
            tx=candidate["bridge_target"][0],
            ty=candidate["bridge_target"][1],
            steps=max(0, len(state.network_path_nodes) - 1),
            budget=plan_budget,
        )
        return

    nodes = state.network_path_nodes
    idx = state.network_path_index
    if not nodes:
        state.phase = "network_plan_path"
        return

    if idx < 0 or idx >= len(nodes):
        state.phase = "network_plan_path"
        return

    if cur_xy != nodes[idx]:
        state.phase = "network_plan_path"
        return

    if state.phase == "conveyor_initialisation":
        if len(nodes) - idx <= 1:
            state.phase = "conveyor_termination"
            return

        nxt = nodes[idx + 1]
        if _enemy_armoured_transport_on_tile(local_map, nxt[0], nxt[1]):
            _invalidate_network_candidate(state, candidate)
            state.network_target = None
            state.network_path_nodes = ()
            state.network_path_index = 0
            state.phase = "network_select_candidate"
            log_event(
                rnd,
                uid,
                "economy",
                f"({cur_xy[0]},{cur_xy[1]})",
                "economy_network_armoured_blocked",
                bx=nxt[0],
                by=nxt[1],
            )
            return

        out_dir = _direction_from_delta(nodes[idx], nxt)
        result = _build_conveyor_on_tile(
            c,
            nodes[idx],
            out_dir,
            rnd,
            uid,
            "economy_network_conveyor_init",
        )
        if result == "built":
            state.phase = "conveyor_execution"
            return
        if result == "wait_cd":
            return

        # If we cannot stamp a conveyor on the current tile (for example an
        # existing road cannot be replaced this round), continue with execution.
        # Execution will still attempt to lay downstream conveyors and can
        # recover by reselection only if the route truly fails.
        state.phase = "conveyor_execution"
        log_event(
            rnd,
            uid,
            "economy",
            f"({cur_xy[0]},{cur_xy[1]})",
            "economy_network_init_skipped",
            bx=nodes[idx][0],
            by=nodes[idx][1],
        )
        return

    if state.phase == "conveyor_execution":
        remaining = (len(nodes) - 1) - idx
        if remaining <= 1:
            state.phase = "conveyor_termination"
            return

        cur_enemy_kind = _enemy_transport_kind_on_tile(
            local_map, cur_xy[0], cur_xy[1])
        if cur_enemy_kind == "armoured":
            _invalidate_network_candidate(state, candidate)
            state.network_target = None
            state.network_path_nodes = ()
            state.network_path_index = 0
            state.phase = "network_select_candidate"
            return
        if cur_enemy_kind == "replaceable":
            if c.get_action_cooldown() > 0:
                return
            cur_pos = Position(cur_xy[0], cur_xy[1])
            try:
                if c.can_fire(cur_pos):
                    c.fire(cur_pos)
                    log_event(
                        rnd,
                        uid,
                        "economy",
                        f"({cur_xy[0]},{cur_xy[1]})",
                        "economy_network_attack_enemy_conveyor",
                    )
            except GameError:
                pass
            return

        nxt = nodes[idx + 1]
        nxt2 = nodes[idx + 2]

        if _enemy_armoured_transport_on_tile(local_map, nxt[0], nxt[1]):
            _invalidate_network_candidate(state, candidate)
            state.network_target = None
            state.network_path_nodes = ()
            state.network_path_index = 0
            state.phase = "network_select_candidate"
            return

        nxt_enemy_kind = _enemy_transport_kind_on_tile(
            local_map, nxt[0], nxt[1])
        if nxt_enemy_kind == "replaceable":
            move_result = _move_only_step(
                c,
                cur_xy,
                nxt,
                rnd,
                uid,
                "economy_network_step_on_enemy_conveyor",
            )
            if move_result == "moved":
                state.issued_move_last_tick = True
                state.expected_xy_after_move = nxt
                state.network_path_index += 1
            return

        out_dir = _direction_from_delta(nxt, nxt2)
        build_result = _build_conveyor_on_tile(
            c,
            nxt,
            out_dir,
            rnd,
            uid,
            "economy_network_conveyor_step",
        )
        if build_result == "wait_cd":
            return
        if build_result != "built":
            _invalidate_network_candidate(state, candidate)
            state.network_target = None
            state.network_path_nodes = ()
            state.network_path_index = 0
            state.phase = "network_select_candidate"
            return

        move_result = _move_only_step(
            c,
            cur_xy,
            nxt,
            rnd,
            uid,
            "economy_network_move_step",
        )
        if move_result == "moved":
            state.issued_move_last_tick = True
            state.expected_xy_after_move = nxt
            state.network_path_index += 1
        return

    if state.phase == "conveyor_termination":
        if idx >= len(nodes) - 1:
            _invalidate_network_candidate(state, candidate)
            state.network_target = None
            state.network_path_nodes = ()
            state.network_path_index = 0
            state.phase = "network_select_candidate"
            return

        final_xy = nodes[idx + 1]
        if _enemy_armoured_transport_on_tile(local_map, final_xy[0], final_xy[1]):
            _invalidate_network_candidate(state, candidate)
            state.network_target = None
            state.network_path_nodes = ()
            state.network_path_index = 0
            state.phase = "network_select_candidate"
            return

        if candidate.get("mode") == "indirect":
            known_bridge_targets = _known_friendly_bridge_targets(local_map)
            target_xy = candidate.get("bridge_target")
            if (
                isinstance(target_xy, tuple)
                and len(target_xy) == 2
                and not _bridge_target_region_clear(target_xy, known_bridge_targets, radius_cheb=0)
            ):
                alt_target = _pick_alternate_bridge_target(
                    local_map,
                    candidate,
                    final_xy,
                    known_bridge_targets,
                )
                if alt_target is not None:
                    old_target = candidate["bridge_target"]
                    candidate["bridge_target"] = alt_target
                    log_event(
                        rnd,
                        uid,
                        "economy",
                        f"({cur_xy[0]},{cur_xy[1]})",
                        "economy_network_bridge_target_retarget",
                        stage="termination",
                        ox=old_target[0],
                        oy=old_target[1],
                        tx=alt_target[0],
                        ty=alt_target[1],
                    )
                else:
                    _invalidate_network_candidate(state, candidate)
                    state.network_target = None
                    state.network_path_nodes = ()
                    state.network_path_index = 0
                    state.phase = "network_select_candidate"
                    log_event(
                        rnd,
                        uid,
                        "economy",
                        f"({cur_xy[0]},{cur_xy[1]})",
                        "economy_network_bridge_target_congested",
                        bx=final_xy[0],
                        by=final_xy[1],
                    )
                    return

        bridge_result = _build_bridge_on_tile(
            c,
            final_xy,
            candidate["bridge_target"],
            rnd,
            uid,
        )
        if bridge_result == "wait_cd":
            return
        if bridge_result not in ("built", "already_built"):
            _invalidate_network_candidate(state, candidate)
            state.network_target = None
            state.network_path_nodes = ()
            state.network_path_index = 0
            state.phase = "network_select_candidate"
            log_event(
                rnd,
                uid,
                "economy",
                f"({cur_xy[0]},{cur_xy[1]})",
                "economy_network_bridge_invalid",
                bx=final_xy[0],
                by=final_xy[1],
            )
            return

        state.built_network_keys.add(candidate["network_key"])
        _record_network_success_source(state, candidate)
        state.network_target = None
        state.network_path_nodes = ()
        state.network_path_index = 0

        if known_ti:
            state.phase = "harvest_pick_ore"
            return

        _resume_exploration_nearest_waypoint(state, cur_xy)
        log_event(
            rnd,
            uid,
            "economy",
            f"({cur_xy[0]},{cur_xy[1]})",
            "economy_network_complete_resume_explore",
        )


def _refresh_network_lists(
    state: EconomyState,
    local_map,
    deadline_ns: int | None = None,
    refresh_stats: dict | None = None,
):
    networks = _scan_complete_titanium_networks(
        local_map,
        deadline_ns=deadline_ns,
        scan_stats=refresh_stats,
    )

    seen_non_full = []
    built_non_full = []
    for key, net in networks.items():
        if net["branch_count"] >= 3:
            continue
        seen_non_full.append(key)
        if key in state.built_network_keys:
            built_non_full.append(key)

    state.seen_non_full_ti_networks = tuple(sorted(seen_non_full))
    state.built_non_full_ti_networks = tuple(sorted(built_non_full))
    return networks


def _record_network_success_source(state: EconomyState, candidate):
    source_key = candidate.get("source_network_key")
    source_conveyor = candidate.get("source_conveyor")
    if not (isinstance(source_key, tuple) and len(source_key) == 2):
        return
    if not (isinstance(source_conveyor, tuple) and len(source_conveyor) == 2):
        return

    skey = (int(source_key[0]), int(source_key[1]))
    scon = (int(source_conveyor[0]), int(source_conveyor[1]))

    new_history: list[tuple[tuple[int, int], tuple[int, int] | None]] = [
        (skey, scon)
    ]
    for item in state.network_success_source_history:
        if not (isinstance(item, tuple) and len(item) == 2):
            continue
        hkey, hanchor = item
        if not (isinstance(hkey, tuple) and len(hkey) == 2):
            continue
        hkey_norm = (int(hkey[0]), int(hkey[1]))
        if hkey_norm == skey:
            continue

        hanchor_norm = None
        if isinstance(hanchor, tuple) and len(hanchor) == 2:
            hanchor_norm = (int(hanchor[0]), int(hanchor[1]))
        new_history.append((hkey_norm, hanchor_norm))
        if len(new_history) >= _NETWORK_SUCCESS_HISTORY_LIMIT:
            break

    state.network_success_source_history = new_history
    state.network_last_success_source_key = skey
    state.network_last_success_source_conveyor = scon


def _record_bridge_only_titanium_link(
    state: EconomyState,
    harvester_xy,
    bridge_xy,
    target_xy,
):
    rec = (
        (int(harvester_xy[0]), int(harvester_xy[1])),
        (int(bridge_xy[0]), int(bridge_xy[1])),
        (int(target_xy[0]), int(target_xy[1])),
    )
    if rec in state.network_bridge_only_ti_links:
        return

    state.network_bridge_only_ti_links.append(rec)
    if len(state.network_bridge_only_ti_links) > 64:
        state.network_bridge_only_ti_links = state.network_bridge_only_ti_links[-64:]


def _ordered_indirect_network_entries(
    state: EconomyState,
    networks,
    cur_xy,
    deadline_ns: int | None = None,
    ordering_stats: dict | None = None,
):
    ordered = []
    seen = set()
    timed_out = False

    history = []
    for item in state.network_success_source_history:
        if not (isinstance(item, tuple) and len(item) == 2):
            continue
        source_key, anchor = item
        if not (isinstance(source_key, tuple) and len(source_key) == 2):
            continue

        source_key = (int(source_key[0]), int(source_key[1]))
        if source_key in seen:
            continue

        net = networks.get(source_key)
        if net is None:
            continue
        if net["branch_count"] >= 3:
            continue

        conv_nodes = net["conveyor_nodes"]
        if not conv_nodes:
            continue

        anchor_xy = None
        if isinstance(anchor, tuple) and len(anchor) == 2:
            anchor_xy = (int(anchor[0]), int(anchor[1]))
            if anchor_xy not in conv_nodes:
                anchor_xy = None

        seen.add(source_key)
        history.append((source_key, anchor_xy))
        ordered.append((len(ordered), source_key, net, anchor_xy))
        if len(history) >= _NETWORK_SUCCESS_HISTORY_LIMIT:
            break

    state.network_success_source_history = history
    if history:
        state.network_last_success_source_key = history[0][0]
        state.network_last_success_source_conveyor = history[0][1]
    else:
        state.network_last_success_source_key = None
        state.network_last_success_source_conveyor = None

    cx, cy = state.core_xy
    others = []
    for source_key, net in networks.items():
        if deadline_ns is not None and time.perf_counter_ns() >= deadline_ns:
            timed_out = True
            break

        if source_key in seen:
            continue
        if net["branch_count"] >= 3:
            continue
        conv_nodes = net["conveyor_nodes"]
        if not conv_nodes:
            continue

        harvester_xy = None
        harvester = net.get("harvester")
        if isinstance(harvester, tuple) and len(harvester) == 2:
            harvester_xy = (int(harvester[0]), int(harvester[1]))

        nearest_cur = 1 << 30
        nearest_harv = 1 << 30
        nearest_core = 1 << 30
        for idx, node in enumerate(conv_nodes):
            if (
                deadline_ns is not None
                and (idx & 31) == 0
                and time.perf_counter_ns() >= deadline_ns
            ):
                timed_out = True
                break

            cur_dist = _manhattan(cur_xy, node)
            if cur_dist < nearest_cur:
                nearest_cur = cur_dist

            if harvester_xy is not None:
                harv_dist = _manhattan(harvester_xy, node)
                if harv_dist < nearest_harv:
                    nearest_harv = harv_dist

            core_dist = abs(node[0] - cx) + abs(node[1] - cy)
            if core_dist < nearest_core:
                nearest_core = core_dist

            if (
                nearest_cur == 0
                and nearest_core == 0
                and (harvester_xy is None or nearest_harv == 0)
            ):
                break

        if timed_out:
            break

        if harvester_xy is None:
            nearest_harv = nearest_core

        others.append((nearest_cur, nearest_harv, nearest_core,
                      source_key[0], source_key[1], source_key, net))

    others.sort()
    rank = len(ordered)
    for _, _, _, _, _, source_key, net in others:
        ordered.append((rank, source_key, net, None))
        rank += 1

    if ordering_stats is not None:
        ordering_stats["ordering_timed_out"] = timed_out

    return ordered


def _ordered_source_nodes_for_network(
    conv_nodes,
    anchor_xy,
    harvester_xy,
    cur_xy,
    max_nodes: int,
):
    if max_nodes <= 0:
        return []

    def key_fn(p):
        harv_dist = (
            _manhattan(harvester_xy, p)
            if harvester_xy is not None
            else 0
        )
        if anchor_xy is None:
            return (harv_dist, _manhattan(cur_xy, p), p[0], p[1])
        return (
            harv_dist,
            0 if p == anchor_xy else 1,
            _manhattan(anchor_xy, p),
            _manhattan(cur_xy, p),
            p[0],
            p[1],
        )

    if len(conv_nodes) <= max_nodes:
        return sorted(conv_nodes, key=key_fn)

    return heapq.nsmallest(max_nodes, conv_nodes, key=key_fn)


def _select_network_candidate(
    state: EconomyState,
    local_map,
    cur_xy,
    networks,
    rnd: int,
    deadline_ns: int | None = None,
    selection_stats: dict | None = None,
):
    direct_candidates = []
    direct_total = 0
    direct_reject_memory = 0
    direct_reject_viability = 0

    indirect_sources = 0
    indirect_candidates = 0
    indirect_reject_region = 0
    indirect_reject_density = 0
    indirect_reject_viability = 0
    indirect_reject_access = 0
    indirect_reject_memory = 0

    cx, cy = state.core_xy
    core_tiles = [
        (cx + dx, cy + dy)
        for dx in (-1, 0, 1)
        for dy in (-1, 0, 1)
    ]

    # Direct core-feed candidates.
    for bx, by in _core_bridge_anchor_candidates(cx, cy):
        direct_total += 1
        if not _is_bridge_build_tile_viable(local_map, bx, by):
            direct_reject_viability += 1
            continue
        tx, ty = _nearest_core_tile_target((bx, by), core_tiles)
        if ((tx - bx) * (tx - bx) + (ty - by) * (ty - by)) > 9:
            direct_reject_viability += 1
            continue

        cand = {
            "mode": "direct",
            "bridge_pos": (bx, by),
            "bridge_target": (tx, ty),
            "network_key": (bx, by),
            "source_network_key": None,
            "source_conveyor": None,
            "score": (1, _manhattan(cur_xy, (bx, by)), abs(bx - cx) + abs(by - cy), bx, by),
        }
        if _candidate_is_invalid(state, _candidate_key(cand), rnd):
            direct_reject_memory += 1
            continue
        direct_candidates.append(cand)

    # Direct-to-core has priority. Only branch indirectly when direct anchors
    # are exhausted by current map knowledge.
    if direct_candidates:
        if selection_stats is not None:
            selection_stats["timed_out"] = False
            selection_stats["direct_total"] = direct_total
            selection_stats["direct_reject_memory"] = direct_reject_memory
            selection_stats["direct_reject_viability"] = direct_reject_viability
            selection_stats["indirect_sources"] = 0
            selection_stats["indirect_candidates"] = 0
            selection_stats["indirect_reject_region"] = 0
            selection_stats["indirect_reject_density"] = 0
            selection_stats["indirect_reject_viability"] = 0
            selection_stats["indirect_reject_access"] = 0
            selection_stats["indirect_reject_memory"] = 0
        return min(direct_candidates, key=lambda cnd: cnd["score"])

    # Indirect candidates: branch into an existing complete friendly titanium network.
    known_bridge_targets = _known_friendly_bridge_targets(local_map)
    density_ok_cache = {}
    best = None
    best_key = None
    timed_out = False
    for history_rank, source_key, net, anchor_xy in _ordered_indirect_network_entries(
        state,
        networks,
        cur_xy,
        deadline_ns=deadline_ns,
        ordering_stats=selection_stats,
    ):
        if deadline_ns is not None and time.perf_counter_ns() >= deadline_ns:
            timed_out = True
            break

        harvester_xy = None
        harvester = net.get("harvester")
        if isinstance(harvester, tuple) and len(harvester) == 2:
            harvester_xy = (int(harvester[0]), int(harvester[1]))

        source_nodes = _ordered_source_nodes_for_network(
            net["conveyor_nodes"],
            anchor_xy,
            harvester_xy,
            cur_xy,
            _INDIRECT_SOURCE_NODES_PER_HISTORY_NETWORK
            if anchor_xy is not None
            else _INDIRECT_SOURCE_NODES_PER_OTHER_NETWORK,
        )

        for sx, sy in source_nodes:
            indirect_sources += 1
            if deadline_ns is not None and time.perf_counter_ns() >= deadline_ns:
                timed_out = True
                break

            # Avoid congested sink points: do not target conveyor points that
            # already have friendly bridge targets nearby.
            if not _bridge_target_region_clear((sx, sy), known_bridge_targets, radius_cheb=2):
                indirect_reject_region += 1
                continue

            density_ok = density_ok_cache.get((sx, sy))
            if density_ok is None:
                density_ok = _indirect_bridge_target_density_viable(
                    local_map,
                    (sx, sy),
                )
                density_ok_cache[(sx, sy)] = density_ok
            if not density_ok:
                indirect_reject_density += 1
                continue

            for dx, dy in _BRIDGE_TARGET_OFFSETS:
                if deadline_ns is not None and time.perf_counter_ns() >= deadline_ns:
                    timed_out = True
                    break

                bx = sx + dx
                by = sy + dy
                if not _is_indirect_bridge_build_tile_viable(local_map, bx, by):
                    indirect_reject_viability += 1
                    continue
                if not _indirect_bridge_build_position_accessible(local_map, bx, by):
                    indirect_reject_access += 1
                    continue

                source_harv_mh = (
                    _manhattan(harvester_xy, (sx, sy))
                    if harvester_xy is not None
                    else abs(sx - cx) + abs(sy - cy)
                )

                cand = {
                    "mode": "indirect",
                    "bridge_pos": (bx, by),
                    "bridge_target": (sx, sy),
                    "network_key": (bx, by),
                    "source_network_key": source_key,
                    "source_conveyor": (sx, sy),
                    "score": (0, _manhattan(cur_xy, (bx, by)), source_harv_mh, bx, by),
                }
                if _candidate_is_invalid(state, _candidate_key(cand), rnd):
                    indirect_reject_memory += 1
                    continue
                indirect_candidates += 1

                local_ref = anchor_xy if anchor_xy is not None else cur_xy
                source_local_mh = (
                    _manhattan(anchor_xy, (sx, sy))
                    if anchor_xy is not None
                    else 0
                )
                if (
                    anchor_xy is not None
                    and source_local_mh <= _INDIRECT_LOCAL_SOURCE_MH_THRESHOLD
                ):
                    local_tier = 0
                elif anchor_xy is not None:
                    local_tier = 1
                else:
                    local_tier = 2

                key = (
                    history_rank,
                    local_tier,
                    source_harv_mh,
                    source_local_mh,
                    _manhattan(local_ref, (bx, by)),
                    _manhattan(cur_xy, (bx, by)),
                    bx,
                    by,
                )
                if best_key is None or key < best_key:
                    best = cand
                    best_key = key

            if timed_out:
                break

        if timed_out:
            break

    if selection_stats is not None:
        selection_stats["timed_out"] = bool(
            selection_stats.get("timed_out", False) or timed_out
        )
        selection_stats["direct_total"] = direct_total
        selection_stats["direct_reject_memory"] = direct_reject_memory
        selection_stats["direct_reject_viability"] = direct_reject_viability
        selection_stats["indirect_sources"] = indirect_sources
        selection_stats["indirect_candidates"] = indirect_candidates
        selection_stats["indirect_reject_region"] = indirect_reject_region
        selection_stats["indirect_reject_density"] = indirect_reject_density
        selection_stats["indirect_reject_viability"] = indirect_reject_viability
        selection_stats["indirect_reject_access"] = indirect_reject_access
        selection_stats["indirect_reject_memory"] = indirect_reject_memory

    return best


def _select_local_bridge_fallback_candidate(
    state: EconomyState,
    local_map,
    cur_xy,
    networks,
    rnd: int,
):
    bx, by = cur_xy
    if not local_map.in_bounds(bx, by):
        return None
    if not _is_indirect_bridge_build_tile_viable(local_map, bx, by):
        return None

    candidates = []
    candidate_keys = []
    known_bridge_targets = _known_friendly_bridge_targets(local_map)
    density_ok_cache = {}

    # Prefer complete titanium networks that still have branch budget.
    for history_rank, source_key, net, anchor_xy in _ordered_indirect_network_entries(
        state,
        networks,
        cur_xy,
    ):
        harvester_xy = None
        harvester = net.get("harvester")
        if isinstance(harvester, tuple) and len(harvester) == 2:
            harvester_xy = (int(harvester[0]), int(harvester[1]))

        for tx, ty in _ordered_source_nodes_for_network(
            net["conveyor_nodes"],
            anchor_xy,
            harvester_xy,
            cur_xy,
            _INDIRECT_SOURCE_NODES_PER_HISTORY_NETWORK
            if anchor_xy is not None
            else _INDIRECT_SOURCE_NODES_PER_OTHER_NETWORK,
        ):
            if not _bridge_target_region_clear((tx, ty), known_bridge_targets, radius_cheb=2):
                continue
            density_ok = density_ok_cache.get((tx, ty))
            if density_ok is None:
                density_ok = _indirect_bridge_target_density_viable(
                    local_map,
                    (tx, ty),
                )
                density_ok_cache[(tx, ty)] = density_ok
            if not density_ok:
                continue
            if tx == bx and ty == by:
                continue
            dist_sq = (tx - bx) * (tx - bx) + (ty - by) * (ty - by)
            if dist_sq <= 0 or dist_sq > 9:
                continue

            cand = {
                "mode": "local_bridge",
                "bridge_pos": (bx, by),
                "bridge_target": (tx, ty),
                "network_key": ("local", bx, by, tx, ty),
                "source_network_key": source_key,
                "source_conveyor": (tx, ty),
                "score": (0, dist_sq, tx, ty),
            }
            if _candidate_is_invalid(state, _candidate_key(cand), rnd):
                continue
            candidates.append(cand)
            source_local_mh = (
                _manhattan(anchor_xy, (tx, ty))
                if anchor_xy is not None
                else 0
            )
            if (
                anchor_xy is not None
                and source_local_mh <= _INDIRECT_LOCAL_SOURCE_MH_THRESHOLD
            ):
                local_tier = 0
            elif anchor_xy is not None:
                local_tier = 1
            else:
                local_tier = 2
            candidate_keys.append(
                (
                    history_rank,
                    local_tier,
                    source_local_mh,
                    dist_sq,
                    abs(tx - bx) + abs(ty - by),
                    tx,
                    ty,
                )
            )

    if not candidates:
        return None

    best_idx = min(range(len(candidates)), key=lambda i: candidate_keys[i])
    return candidates[best_idx]


def _candidate_key(candidate) -> tuple:
    bp = candidate["bridge_pos"]
    bt = candidate["bridge_target"]
    source_key = candidate.get("source_network_key")
    source_conv = candidate.get("source_conveyor")

    if source_key is None:
        source_key = (-1, -1)
    if source_conv is None:
        source_conv = (-1, -1)

    return (
        candidate["mode"],
        bp[0],
        bp[1],
        bt[0],
        bt[1],
        source_key[0],
        source_key[1],
        source_conv[0],
        source_conv[1],
    )


def _count_active_timeout_candidates(state: EconomyState, rnd: int) -> int:
    expired = []
    count = 0
    for key, until_round in state.network_timeout_candidates.items():
        if rnd >= until_round:
            expired.append(key)
            continue
        count += 1
    for key in expired:
        state.network_timeout_candidates.pop(key, None)
    return count


def _candidate_is_invalid(state: EconomyState, candidate_key: tuple, rnd: int) -> bool:
    if candidate_key in state.network_invalid_candidates:
        return True

    until_round = state.network_timeout_candidates.get(candidate_key)
    if until_round is None:
        return False
    if rnd >= until_round:
        state.network_timeout_candidates.pop(candidate_key, None)
        return False
    return True


def _invalidate_network_candidate(
    state: EconomyState,
    candidate,
    timeout_round: int | None = None,
):
    key = _candidate_key(candidate)
    if timeout_round is not None:
        state.network_timeout_candidates[key] = (
            int(timeout_round) + _NETWORK_TIMEOUT_RETRY_COOLDOWN_ROUNDS
        )
        return

    state.network_timeout_candidates.pop(key, None)
    state.network_invalid_candidates.add(key)


def _scan_complete_titanium_networks(
    local_map,
    deadline_ns: int | None = None,
    scan_stats: dict | None = None,
):
    networks = {}
    titanium_harvesters = getattr(local_map, "titanium_harvesters", set())
    trace_cache = {}
    timed_out = False

    for hx, hy in titanium_harvesters:
        if deadline_ns is not None and time.perf_counter_ns() >= deadline_ns:
            timed_out = True
            break

        harvester = _known_building_at(local_map, hx, hy)
        if not isinstance(harvester, dict):
            continue
        if harvester.get("entity_type") != EntityType.HARVESTER:
            continue

        for dx, dy in _ADJACENT_DELTAS_8:
            if deadline_ns is not None and time.perf_counter_ns() >= deadline_ns:
                timed_out = True
                break

            sx = hx + dx
            sy = hy + dy
            start = (sx, sy)
            if start in trace_cache:
                chain = trace_cache[start]
            else:
                trace_stats = {}
                chain = _trace_friendly_titanium_chain(
                    local_map,
                    start,
                    deadline_ns=deadline_ns,
                    trace_stats=trace_stats,
                )
                if trace_stats.get("timed_out", False):
                    timed_out = True
                    trace_cache[start] = None
                    break
                trace_cache[start] = chain
            if chain is None:
                continue

            key = chain["terminal_bridge"]
            existing = networks.get(key)
            if existing is None or len(chain["conveyor_nodes"]) < len(existing["conveyor_nodes"]):
                chain["harvester"] = (hx, hy)
                networks[key] = chain

        if timed_out:
            break

    bridge_target_counts, bridge_target_by_pos = _collect_friendly_bridge_target_maps(
        local_map)
    for net in networks.values():
        if deadline_ns is not None and time.perf_counter_ns() >= deadline_ns:
            timed_out = True
            break

        net["branch_count"] = _count_network_branches(
            net["conveyor_nodes"],
            net["terminal_bridge"],
            bridge_target_counts,
            bridge_target_by_pos,
        )

    if timed_out:
        for net in networks.values():
            net.setdefault("branch_count", 3)

    if scan_stats is not None:
        scan_stats["scan_timed_out"] = timed_out
        scan_stats["scan_network_count"] = len(networks)

    return networks


def _trace_friendly_titanium_chain(
    local_map,
    start_xy,
    deadline_ns: int | None = None,
    trace_stats: dict | None = None,
):
    sx, sy = start_xy
    if not local_map.in_bounds(sx, sy):
        return None

    conveyor_nodes = []
    seen = set()
    cur_x, cur_y = sx, sy
    steps = 0

    while steps < 256:
        if (
            deadline_ns is not None
            and (steps & 15) == 0
            and time.perf_counter_ns() >= deadline_ns
        ):
            if trace_stats is not None:
                trace_stats["timed_out"] = True
            return None

        if (cur_x, cur_y) in seen:
            return None
        seen.add((cur_x, cur_y))
        rec = _known_building_at(local_map, cur_x, cur_y)
        if not isinstance(rec, dict):
            return None
        if rec.get("team") != getattr(local_map, "my_team", rec.get("team")):
            return None

        etype = rec.get("entity_type")
        if etype == EntityType.BRIDGE:
            target = rec.get("bridge_target")
            if not (isinstance(target, tuple) and len(target) == 2):
                return None
            return {
                "conveyor_nodes": tuple(conveyor_nodes),
                "terminal_bridge": (cur_x, cur_y),
                "bridge_target": (int(target[0]), int(target[1])),
            }

        if etype not in (EntityType.CONVEYOR, EntityType.ARMOURED_CONVEYOR, EntityType.SPLITTER):
            return None

        direction = rec.get("direction")
        if direction is None:
            return None

        conveyor_nodes.append((cur_x, cur_y))
        ddx, ddy = direction.delta()
        cur_x += ddx
        cur_y += ddy
        if not local_map.in_bounds(cur_x, cur_y):
            return None

        steps += 1

    return None


def _collect_friendly_bridge_target_maps(local_map):
    bridge_target_counts = {}
    bridge_target_by_pos = {}

    entities = getattr(local_map, "entities", None)
    my_team = getattr(local_map, "my_team", None)
    if not isinstance(entities, dict):
        return bridge_target_counts, bridge_target_by_pos

    for rec in entities.values():
        if not isinstance(rec, dict):
            continue
        if not rec.get("alive", False):
            continue
        if rec.get("entity_type") != EntityType.BRIDGE:
            continue
        if rec.get("team") != my_team:
            continue

        pos = rec.get("position")
        target = rec.get("bridge_target")
        if not (isinstance(pos, tuple) and len(pos) == 2):
            continue
        if not (isinstance(target, tuple) and len(target) == 2):
            continue

        bxy = (int(pos[0]), int(pos[1]))
        txy = (int(target[0]), int(target[1]))
        bridge_target_by_pos[bxy] = txy
        bridge_target_counts[txy] = bridge_target_counts.get(txy, 0) + 1

    return bridge_target_counts, bridge_target_by_pos


def _count_network_branches(
    conveyor_nodes,
    terminal_bridge,
    bridge_target_counts,
    bridge_target_by_pos,
):
    conv_set = set(conveyor_nodes)
    if not conv_set:
        return 0

    count = 0
    for txy in conv_set:
        count += bridge_target_counts.get(txy, 0)

    own_target = bridge_target_by_pos.get(terminal_bridge)
    if own_target in conv_set:
        count -= 1

    if count < 0:
        return 0
    return count


def _known_building_at(local_map, x: int, y: int):
    getter = getattr(local_map, "get_known_building", None)
    if not callable(getter):
        return None
    try:
        return getter(x, y)
    except (TypeError, GameError):
        return None


def _known_friendly_bridge_targets(local_map):
    out = set()
    entities = getattr(local_map, "entities", None)
    my_team = getattr(local_map, "my_team", None)
    if not isinstance(entities, dict):
        return out

    for rec in entities.values():
        if not isinstance(rec, dict):
            continue
        if not rec.get("alive", False):
            continue
        if rec.get("team") != my_team:
            continue
        if rec.get("entity_type") != EntityType.BRIDGE:
            continue

        target = rec.get("bridge_target")
        if not (isinstance(target, tuple) and len(target) == 2):
            continue
        out.add((int(target[0]), int(target[1])))

    return out


def _bridge_target_region_clear(target_xy, known_bridge_targets, radius_cheb: int) -> bool:
    tx, ty = target_xy
    for bx, by in known_bridge_targets:
        if max(abs(bx - tx), abs(by - ty)) <= radius_cheb:
            return False
    return True


def _is_bridge_escape_source_tile(local_map, x: int, y: int) -> bool:
    rec = _known_building_at(local_map, x, y)
    if not isinstance(rec, dict):
        return False

    if rec.get("team") != getattr(local_map, "my_team", None):
        return False

    return rec.get("entity_type") in (
        EntityType.CONVEYOR,
        EntityType.ARMOURED_CONVEYOR,
        EntityType.SPLITTER,
        EntityType.BRIDGE,
    )


def _select_bridge_escape_target(local_map, source_xy, core_xy):
    sx, sy = source_xy
    if not _is_bridge_escape_source_tile(local_map, sx, sy):
        return None

    my_team = getattr(local_map, "my_team", None)
    known_bridge_targets = _known_friendly_bridge_targets(local_map)
    cx, cy = core_xy
    best = None
    best_key = None

    for dx, dy in _BRIDGE_TARGET_OFFSETS:
        if max(abs(dx), abs(dy)) > 3:
            continue

        tx = sx + dx
        ty = sy + dy
        if not local_map.in_bounds(tx, ty):
            continue

        rec = _known_building_at(local_map, tx, ty)
        if not isinstance(rec, dict):
            continue
        if rec.get("team") != my_team:
            continue
        if rec.get("entity_type") not in (
            EntityType.CONVEYOR,
            EntityType.ARMOURED_CONVEYOR,
            EntityType.SPLITTER,
        ):
            continue

        if not _bridge_target_region_clear((tx, ty), known_bridge_targets, radius_cheb=0):
            continue

        key = (
            max(abs(dx), abs(dy)),
            abs(dx) + abs(dy),
            abs(tx - cx) + abs(ty - cy),
            tx,
            ty,
        )
        if best_key is None or key < best_key:
            best_key = key
            best = (tx, ty)

    return best


def _select_harvester_highway_bridge_target(
    local_map,
    harvester_xy,
    bridge_build_xy,
    core_xy,
):
    hx, hy = int(harvester_xy[0]), int(harvester_xy[1])
    bx, by = int(bridge_build_xy[0]), int(bridge_build_xy[1])

    if not _is_bridge_build_tile_viable(local_map, bx, by):
        return None

    my_team = getattr(local_map, "my_team", None)
    known_bridge_targets = _known_friendly_bridge_targets(local_map)
    cx, cy = int(core_xy[0]), int(core_xy[1])
    best = None
    best_key = None

    entities = getattr(local_map, "entities", None)
    if not isinstance(entities, dict):
        return None

    for rec in entities.values():
        if not isinstance(rec, dict):
            continue
        if not rec.get("alive", False):
            continue
        if rec.get("team") != my_team:
            continue
        if rec.get("entity_type") not in (
            EntityType.CONVEYOR,
            EntityType.ARMOURED_CONVEYOR,
            EntityType.SPLITTER,
        ):
            continue

        pos = rec.get("position")
        if not (isinstance(pos, tuple) and len(pos) == 2):
            continue

        tx = int(pos[0])
        ty = int(pos[1])

        if max(abs(tx - hx), abs(ty - hy)) > 3:
            continue
        if (tx, ty) == (bx, by):
            continue

        dist_sq = (tx - bx) * (tx - bx) + (ty - by) * (ty - by)
        if dist_sq <= 0 or dist_sq > 9:
            continue

        # Keep this highway bridge sparse to avoid local sink congestion.
        if not _bridge_target_region_clear((tx, ty), known_bridge_targets, radius_cheb=2):
            continue

        key = (
            max(abs(tx - hx), abs(ty - hy)),
            abs(tx - hx) + abs(ty - hy),
            abs(tx - cx) + abs(ty - cy),
            dist_sq,
            tx,
            ty,
        )
        if best_key is None or key < best_key:
            best_key = key
            best = (tx, ty)

    return best


def _pick_alternate_bridge_target(local_map, candidate, build_xy, known_bridge_targets):
    source_key = candidate.get("source_network_key")
    if not (isinstance(source_key, tuple) and len(source_key) == 2):
        return None

    networks = _scan_complete_titanium_networks(local_map)
    source_key = (int(source_key[0]), int(source_key[1]))
    net = networks.get(source_key)
    if net is None:
        return None

    old_target = candidate.get("bridge_target")
    if not (isinstance(old_target, tuple) and len(old_target) == 2):
        return None

    ox, oy = int(old_target[0]), int(old_target[1])
    bx, by = int(build_xy[0]), int(build_xy[1])

    best = None
    best_key = None
    for tx, ty in net["conveyor_nodes"]:
        tx = int(tx)
        ty = int(ty)
        if (tx, ty) == (ox, oy):
            continue
        if max(abs(tx - ox), abs(ty - oy)) > 2:
            continue

        dist_sq = (tx - bx) * (tx - bx) + (ty - by) * (ty - by)
        if dist_sq <= 0 or dist_sq > 9:
            continue

        # Keep spacing around conveyor sink points to reduce congestion.
        if not _bridge_target_region_clear((tx, ty), known_bridge_targets, radius_cheb=3):
            continue

        key = (
            max(abs(tx - ox), abs(ty - oy)),
            abs(tx - ox) + abs(ty - oy),
            dist_sq,
            tx,
            ty,
        )
        if best_key is None or key < best_key:
            best_key = key
            best = (tx, ty)

    return best


def _core_bridge_anchor_candidates(cx: int, cy: int):
    out = []
    for dy in (-1, 0, 1):
        out.append((cx - 4, cy + dy))
        out.append((cx + 4, cy + dy))
    for dx in (-1, 0, 1):
        out.append((cx + dx, cy - 4))
        out.append((cx + dx, cy + 4))
    return tuple(out)


def _nearest_core_tile_target(anchor_xy, core_tiles):
    ax, ay = anchor_xy
    return min(
        core_tiles,
        key=lambda p: (
            (p[0] - ax) * (p[0] - ax) + (p[1] - ay) * (p[1] - ay),
            p[0],
            p[1],
        ),
    )


def _is_bridge_build_tile_viable(local_map, x: int, y: int) -> bool:
    if not local_map.in_bounds(x, y):
        return False
    if not _tile_is_known(local_map, x, y):
        return False

    tile = local_map.get(x, y)
    if tile in (MAP_OBSTACLE, MAP_ORE_TITANIUM, MAP_ORE_AXIONITE):
        return False

    rec = _known_building_at(local_map, x, y)
    if isinstance(rec, dict):
        etype = rec.get("entity_type")
        if etype not in (EntityType.ROAD, EntityType.BRIDGE):
            return False
    return True


def _is_indirect_bridge_build_tile_viable(local_map, x: int, y: int) -> bool:
    if not local_map.in_bounds(x, y):
        return False
    if not _tile_is_known(local_map, x, y):
        return False

    tile = local_map.get(x, y)
    if tile not in (MAP_FREE, MAP_ROAD):
        return False

    rec = _known_building_at(local_map, x, y)
    if isinstance(rec, dict):
        etype = rec.get("entity_type")
        if etype not in (EntityType.ROAD, EntityType.BRIDGE):
            return False
    return True


def _indirect_bridge_target_density_viable(local_map, target_xy) -> bool:
    tx, ty = target_xy
    known_tiles = 0
    blocked_tiles = 0
    my_team = getattr(local_map, "my_team", None)

    for dx in range(-_INDIRECT_TARGET_DENSITY_RADIUS_CHEB, _INDIRECT_TARGET_DENSITY_RADIUS_CHEB + 1):
        for dy in range(-_INDIRECT_TARGET_DENSITY_RADIUS_CHEB, _INDIRECT_TARGET_DENSITY_RADIUS_CHEB + 1):
            x = tx + dx
            y = ty + dy
            if not local_map.in_bounds(x, y):
                continue
            if not _tile_is_known(local_map, x, y):
                continue

            known_tiles += 1
            tile = local_map.get(x, y)
            if tile in (MAP_OBSTACLE, MAP_ORE_TITANIUM, MAP_ORE_AXIONITE):
                blocked_tiles += 1
                continue

            rec = _known_building_at(local_map, x, y)
            if not isinstance(rec, dict):
                continue

            etype = rec.get("entity_type")
            team = rec.get("team")
            if (
                team == my_team
                and etype in (
                    EntityType.CONVEYOR,
                    EntityType.ARMOURED_CONVEYOR,
                    EntityType.SPLITTER,
                    EntityType.BRIDGE,
                )
            ):
                blocked_tiles += 1
                continue

            if etype in (
                EntityType.BARRIER,
                EntityType.CORE,
                EntityType.HARVESTER,
                EntityType.FOUNDRY,
                EntityType.LAUNCHER,
                EntityType.GUNNER,
                EntityType.SENTINEL,
                EntityType.BREACH,
            ):
                blocked_tiles += 1

    if known_tiles < _INDIRECT_TARGET_MIN_KNOWN_TILES:
        return True

    blocked_ratio = blocked_tiles / float(known_tiles)
    return blocked_ratio <= _INDIRECT_TARGET_MAX_BLOCKED_DENSITY


def _indirect_bridge_build_position_accessible(local_map, x: int, y: int) -> bool:
    # Reject bridge build tiles that are boxed in by known conveyor structures.
    # At least one cardinal neighbor must be planner-passable as an approach tile.
    goal_xy = (x, y)
    for dx, dy in CARDINAL_DELTAS:
        nx = x + dx
        ny = y + dy
        if _is_conveyor_planner_passable(local_map, nx, ny, goal_xy):
            return True
    return False


def _is_conveyor_planner_passable(local_map, x: int, y: int, goal_xy) -> bool:
    if not local_map.in_bounds(x, y):
        return False

    rec = _known_building_at(local_map, x, y)
    my_team = getattr(local_map, "my_team", None)
    if isinstance(rec, dict):
        etype = rec.get("entity_type")
        team = rec.get("team")

        if team == my_team:
            if etype in (
                EntityType.CONVEYOR,
                EntityType.ARMOURED_CONVEYOR,
                EntityType.SPLITTER,
                EntityType.BRIDGE,
            ):
                return (x, y) == goal_xy
        else:
            if etype == EntityType.ARMOURED_CONVEYOR:
                return False
            if etype in (EntityType.CONVEYOR, EntityType.SPLITTER, EntityType.BRIDGE):
                return True
            return False

    return local_map.get(x, y) in PASSABLE_TILES


def _enemy_transport_kind_on_tile(local_map, x: int, y: int) -> str | None:
    rec = _known_building_at(local_map, x, y)
    if not isinstance(rec, dict):
        return None

    my_team = getattr(local_map, "my_team", None)
    if rec.get("team") == my_team:
        return None

    etype = rec.get("entity_type")
    if etype == EntityType.ARMOURED_CONVEYOR:
        return "armoured"
    if etype in (EntityType.CONVEYOR, EntityType.SPLITTER, EntityType.BRIDGE):
        return "replaceable"
    return None


def _enemy_armoured_transport_on_tile(local_map, x: int, y: int) -> bool:
    return _enemy_transport_kind_on_tile(local_map, x, y) == "armoured"


def _direction_from_delta(a_xy, b_xy):
    dx = b_xy[0] - a_xy[0]
    dy = b_xy[1] - a_xy[1]
    direction = _CARDINAL_DIRECTION_BY_DELTA.get((dx, dy))
    if direction is None:
        raise ValueError("non-cardinal conveyor direction")
    return direction


def _build_conveyor_on_tile(
    c: Controller,
    build_xy,
    out_dir: Direction,
    rnd: int,
    uid: int,
    tag: str,
) -> str:
    if c.get_action_cooldown() > 0:
        return "wait_cd"

    bp = Position(build_xy[0], build_xy[1])
    try:
        if not c.can_build_conveyor(bp, out_dir):
            # Retry path: if this is our own replaceable transport/road tile,
            # clear it and stamp the new conveyor direction in the same round.
            try:
                existing = c.get_tile_building_id(bp)
            except GameError:
                existing = None

            if existing is not None:
                try:
                    same_team = c.get_team(existing) == c.get_team()
                    existing_type = c.get_entity_type(existing)
                    if (
                        same_team
                        and existing_type in (
                            EntityType.ROAD,
                            EntityType.CONVEYOR,
                            EntityType.ARMOURED_CONVEYOR,
                            EntityType.SPLITTER,
                            EntityType.BRIDGE,
                        )
                        and c.can_destroy(bp)
                    ):
                        c.destroy(bp)
                except GameError:
                    return "invalid"

            if not c.can_build_conveyor(bp, out_dir):
                return "invalid"

        c.build_conveyor(bp, out_dir)
        log_event(
            rnd,
            uid,
            "economy",
            f"({build_xy[0]},{build_xy[1]})",
            tag,
            d=out_dir.name,
        )
        return "built"
    except GameError:
        return "invalid"


def _build_bridge_on_tile(c: Controller, bridge_xy, target_xy, rnd: int, uid: int) -> str:
    bp = Position(bridge_xy[0], bridge_xy[1])
    tp = Position(target_xy[0], target_xy[1])
    try:
        try:
            existing = c.get_tile_building_id(bp)
        except GameError:
            existing = None

        if existing is not None:
            try:
                if (
                    c.get_team(existing) == c.get_team()
                    and c.get_entity_type(existing) == EntityType.BRIDGE
                ):
                    match_target = 0
                    try:
                        ex_target = c.get_bridge_target(existing)
                        if (ex_target.x, ex_target.y) == (target_xy[0], target_xy[1]):
                            match_target = 1
                    except GameError:
                        pass

                    log_event(
                        rnd,
                        uid,
                        "economy",
                        f"({bridge_xy[0]},{bridge_xy[1]})",
                        "economy_network_bridge_already_present",
                        tx=target_xy[0],
                        ty=target_xy[1],
                        match_target=match_target,
                    )
                    return "already_built"
            except GameError:
                pass

        if c.get_action_cooldown() > 0:
            return "wait_cd"

        if not c.can_build_bridge(bp, tp):
            if existing is not None:
                try:
                    if (
                        c.get_team(existing) == c.get_team()
                        and c.get_entity_type(existing) in (
                            EntityType.ROAD,
                            EntityType.CONVEYOR,
                            EntityType.ARMOURED_CONVEYOR,
                            EntityType.SPLITTER,
                        )
                        and c.can_destroy(bp)
                    ):
                        c.destroy(bp)
                except GameError:
                    return "invalid"

            if not c.can_build_bridge(bp, tp):
                return "invalid"
        c.build_bridge(bp, tp)
        log_event(
            rnd,
            uid,
            "economy",
            f"({bridge_xy[0]},{bridge_xy[1]})",
            "economy_network_bridge_built",
            tx=target_xy[0],
            ty=target_xy[1],
        )
        return "built"
    except GameError:
        return "invalid"


def _move_only_step(c: Controller, cur_xy, nxt_xy, rnd: int, uid: int, tag: str) -> str:
    if c.get_move_cooldown() > 0:
        return "wait_cd"

    dx = nxt_xy[0] - cur_xy[0]
    dy = nxt_xy[1] - cur_xy[1]
    move_dir = _CARDINAL_DIRECTION_BY_DELTA.get((dx, dy))
    if move_dir is None:
        return "invalid"

    try:
        if not c.can_move(move_dir):
            return "blocked"
        c.move(move_dir)
        log_event(
            rnd,
            uid,
            "economy",
            f"({cur_xy[0]},{cur_xy[1]})",
            tag,
            nx=nxt_xy[0],
            ny=nxt_xy[1],
            d=move_dir.name,
        )
        return "moved"
    except GameError:
        return "blocked"


def _network_fallback_to_next_objective(state: EconomyState, cur_xy, known_ti):
    state.network_target = None
    state.network_path_nodes = ()
    state.network_path_index = 0
    state.network_escape_bridge_target = None
    state.network_highway_pending_harvester = None
    state.network_highway_active_harvester = None
    if known_ti:
        state.phase = "harvest_pick_ore"
        return
    _resume_exploration_nearest_waypoint(state, cur_xy)


def _resume_exploration_nearest_waypoint(state: EconomyState, cur_xy):
    if not state.explore_waypoints:
        state.phase = (
            "patrol_generate_waypoints"
            if state.patrol_unlocked
            else "explore_generate_waypoints"
        )
        state.explore_target_xy = None
        state.explore_waypoint_index = 0
        state.plan_steps = ()
        state.plan_index = 0
        state.defer_step_once = False
        return

    idx = min(
        range(len(state.explore_waypoints)),
        key=lambda i: _manhattan(cur_xy, state.explore_waypoints[i]),
    )
    state.phase = "patrol_plan_waypoint" if state.patrol_unlocked else "explore_plan_waypoint"
    state.explore_waypoint_index = idx
    state.explore_target_xy = None
    state.plan_steps = ()
    state.plan_index = 0
    state.defer_step_once = False


# ---------------------------
# Internal helpers
# ---------------------------


def _tile_is_known(local_map, x: int, y: int) -> bool:
    fn = getattr(local_map, "is_tile_known", None)
    if callable(fn):
        try:
            return bool(fn(x, y))
        except (TypeError, GameError):
            return False

    vis = getattr(local_map, "is_visible", None)
    if callable(vis):
        try:
            return bool(vis(x, y))
        except (TypeError, GameError):
            return False

    return True


def _known_unharvested_titanium(local_map):
    getter = getattr(local_map, "get_known_unharvested_titanium", None)
    if callable(getter):
        try:
            data = getter()
        except (TypeError, GameError):
            return []

        if not isinstance(data, (list, tuple, set)):
            return []

        out = []
        for item in data:
            if not isinstance(item, tuple) or len(item) != 2:
                continue
            out.append((int(item[0]), int(item[1])))
        return out

    out = []
    for x, y in getattr(local_map, "titanium_unharvested", set()):
        if _tile_is_known(local_map, x, y):
            out.append((x, y))
    return out


def _known_unharvested_titanium_unblocked(local_map, blocked_ores):
    known_ti = _known_unharvested_titanium(local_map)
    if not blocked_ores:
        return known_ti
    return [ore for ore in known_ti if ore not in blocked_ores]


def _pick_nearest_titanium_ore(cur_xy, ores):
    return min(
        ores,
        key=lambda p: (
            abs(p[0] - cur_xy[0]) + abs(p[1] - cur_xy[1]),
            p[0],
            p[1],
        ),
    )


def _is_titanium_goal_valid(local_map, ore_xy, goal_xy) -> bool:
    ox, oy = ore_xy
    gx, gy = goal_xy

    if abs(gx - ox) + abs(gy - oy) != 1:
        return False
    if not local_map.in_bounds(gx, gy):
        return False
    if not _tile_is_known(local_map, gx, gy):
        return False

    return local_map.get(gx, gy) == MAP_FREE


def _pick_titanium_adjacent_goal(local_map, ore_xy, cur_xy):
    ox, oy = ore_xy
    candidates = []
    for dx, dy in CARDINAL_DELTAS:
        gx = ox + dx
        gy = oy + dy
        if not local_map.in_bounds(gx, gy):
            continue
        if not _tile_is_known(local_map, gx, gy):
            continue

        tile = local_map.get(gx, gy)
        if tile != MAP_FREE:
            continue

        candidates.append(
            (
                abs(gx - cur_xy[0]) + abs(gy - cur_xy[1]),
                gx,
                gy,
            )
        )

    if not candidates:
        return None

    _, gx, gy = min(candidates)
    return gx, gy


def _resume_exploration_after_harvest(state: EconomyState):
    state.phase = (
        "patrol_generate_waypoints"
        if state.patrol_unlocked
        else "explore_generate_waypoints"
    )
    state.harvest_ore_xy = None
    state.harvest_goal_xy = None
    state.plan_steps = ()
    state.plan_index = 0
    state.defer_step_once = False


def _track_symmetry_revision(state: EconomyState, local_map, rnd: int, uid: int, cur_xy):
    symmetry_name = str(getattr(local_map, "symmetry", "UNKNOWN") or "UNKNOWN")
    revision = int(getattr(local_map, "symmetry_revision", 0))

    if state.last_symmetry is None:
        state.last_symmetry = symmetry_name
        state.last_symmetry_revision = revision
        return

    if (
        symmetry_name == state.last_symmetry
        and revision == state.last_symmetry_revision
    ):
        return

    state.last_symmetry = symmetry_name
    state.last_symmetry_revision = revision

    if state.phase in _POST_LAUNCH_PHASES:
        state.waypoint_refresh_round = rnd + 1

    log_event(
        rnd,
        uid,
        "economy",
        f"({cur_xy[0]},{cur_xy[1]})",
        "economy_symmetry_revision_seen",
        symmetry=symmetry_name,
        revision=revision,
        refresh_round=state.waypoint_refresh_round,
    )


def _detect_external_relocation(state: EconomyState, cur_xy, rnd: int, uid: int):
    prev_xy = state.last_xy
    launched = False
    if prev_xy is not None and cur_xy != prev_xy:
        if not state.issued_move_last_tick:
            launched = True
        elif state.expected_xy_after_move is not None and cur_xy != state.expected_xy_after_move:
            # We moved, but landed somewhere else by next round:
            # this strongly indicates launcher throw (or another external relocation).
            launched = True

    if launched:
        state.phase = "launched"
        if prev_xy is None:
            prev_xy = cur_xy
        fx, fy = prev_xy
        tx, ty = cur_xy
        log_event(
            rnd,
            uid,
            "economy",
            f"({tx},{ty})",
            "economy_detected_launch",
            fx=fx,
            fy=fy,
            tx=tx,
            ty=ty,
        )

    state.issued_move_last_tick = False
    state.expected_xy_after_move = None
    state.last_xy = cur_xy


def _find_friendly_launcher_xy(c: Controller, local_map, cur_xy):
    my_team = c.get_team()

    best: tuple[int, int] | None = None
    best_dist = 10**9

    entities = getattr(local_map, "entities", None)
    if isinstance(entities, dict):
        for rec in entities.values():
            if rec.get("entity_type") != EntityType.LAUNCHER:
                continue
            if rec.get("team") != my_team:
                continue
            if not rec.get("alive", False):
                continue
            p = rec.get("position")
            if not isinstance(p, tuple) or len(p) != 2:
                continue
            px = int(p[0])
            py = int(p[1])
            cand = (px, py)
            d = _manhattan(cur_xy, cand)
            if d < best_dist:
                best_dist = d
                best = cand

    if best is not None:
        return best

    for bid in c.get_nearby_buildings():
        try:
            if c.get_entity_type(bid) != EntityType.LAUNCHER:
                continue
            if c.get_team(bid) != my_team:
                continue
            p = c.get_position(bid)
            d = _manhattan(cur_xy, (p.x, p.y))
            if d < best_dist:
                best_dist = d
                best = (p.x, p.y)
        except GameError:
            continue

    return best


def _pick_launcher_adjacent_goal(local_map, cur_xy, launcher_xy, core_xy):
    lx, ly = launcher_xy
    cx, cy = core_xy

    best = None
    best_key = None

    in_bounds = local_map.in_bounds
    get_tile = local_map.get

    for dx, dy in CARDINAL_DELTAS:
        gx = lx + dx
        gy = ly + dy
        if not in_bounds(gx, gy):
            continue

        tile = get_tile(gx, gy)
        if tile not in WALKABLE_TILES:
            continue

        # Prioritise queue-friendly inside-core tiles near the core center,
        # then nearer from current position.
        in_core_3x3 = 1 if (abs(gx - cx) <= 1 and abs(gy - cy) <= 1) else 0
        key = (
            -in_core_3x3,
            abs(gx - cx) + abs(gy - cy),
            abs(gx - cur_xy[0]) + abs(gy - cur_xy[1]),
        )

        if best_key is None or key < best_key:
            best_key = key
            best = (gx, gy)

    return best


def _pick_launcher_wait_tile(local_map, core_xy, launcher_xy):
    cx, cy = core_xy
    lx, ly = launcher_xy

    in_bounds = getattr(local_map, "in_bounds", None)

    candidates = []
    for dx in (-1, 0, 1):
        for dy in (-1, 0, 1):
            wx = cx + dx
            wy = cy + dy
            if callable(in_bounds) and not in_bounds(wx, wy):
                continue

            ddx = wx - lx
            ddy = wy - ly
            if ddx * ddx + ddy * ddy <= ACTION_RADIUS_SQ:
                candidates.append((wx, wy))

    if not candidates:
        return None

    return min(
        candidates,
        key=lambda t: abs(t[0] - cx) + abs(t[1] - cy),
    )


def _build_exploration_waypoints(local_map, start_xy, vision_radius_sq: int):
    width = getattr(local_map, "width", 0)
    height = getattr(local_map, "height", 0)
    symmetry = str(getattr(local_map, "symmetry",
                   "ROTATIONAL") or "ROTATIONAL")

    x_min, x_max, y_min, y_max, half_name = _exploration_half_bounds(
        width,
        height,
        symmetry,
        start_xy,
    )

    # Vision-aware coarse serpentine:
    # choose sweep rows based on vision radius and only traverse each row end-to-end.
    axis_radius = max(1, int(math.isqrt(max(1, vision_radius_sq))))
    stride = max(2, axis_radius * 2)

    y_rows = _coverage_scanlines(y_min, y_max, axis_radius, stride)
    x_lo, x_hi = _coverage_row_endpoints(x_min, x_max, axis_radius)

    waypoints = []
    if y_rows:
        sx, _ = start_xy
        start_left = abs(sx - x_lo) <= abs(sx - x_hi)
        first_a, first_b = (x_lo, x_hi) if start_left else (x_hi, x_lo)

        for row_idx, y in enumerate(y_rows):
            if first_a == first_b:
                waypoints.append((first_a, y))
                continue

            if (row_idx % 2) == 0:
                a, b = first_a, first_b
            else:
                a, b = first_b, first_a
            waypoints.append((a, y))
            waypoints.append((b, y))

    compact = []
    for p in waypoints:
        if not compact or compact[-1] != p:
            compact.append(p)

    return tuple(compact), {
        "symmetry": symmetry,
        "half": half_name,
        "bounds": f"[{x_min},{x_max}]x[{y_min},{y_max}]",
        "vision_r_sq": vision_radius_sq,
        "stride": stride,
    }


def _build_patrol_waypoints(local_map, start_xy, vision_radius_sq: int):
    width = getattr(local_map, "width", 0)
    height = getattr(local_map, "height", 0)

    if width <= 0 or height <= 0:
        return (), {
            "bounds": "[0,0]x[0,0]",
            "vision_r_sq": vision_radius_sq,
            "stride": 0,
        }

    x_min, x_max = 0, max(0, width - 1)
    y_min, y_max = 0, max(0, height - 1)

    # Full-map patrol coverage guarantee:
    # - sweep from map edge to edge on each patrol row,
    # - choose row spacing by vision radius so adjacent sweeps overlap/touch.
    axis_radius = max(1, int(math.isqrt(max(1, vision_radius_sq))))
    stride = max(1, axis_radius * 2)

    y_rows = _coverage_scanlines_edge_to_edge(y_min, y_max, stride)
    x_lo, x_hi = x_min, x_max

    waypoints = []
    if y_rows:
        sx, sy = start_xy
        start_row_idx = min(range(len(y_rows)),
                            key=lambda i: abs(y_rows[i] - sy))
        ordered_rows = list(y_rows[start_row_idx:]) + \
            list(y_rows[:start_row_idx])

        start_left = abs(sx - x_lo) <= abs(sx - x_hi)
        first_a, first_b = (x_lo, x_hi) if start_left else (x_hi, x_lo)

        for row_idx, y in enumerate(ordered_rows):
            row_a = _pick_patrol_row_anchor_x(local_map, y, first_a, first_b)
            row_b = _pick_patrol_row_anchor_x(local_map, y, first_b, first_a)

            if row_a == row_b:
                waypoints.append((row_a, y))
                continue

            if (row_idx % 2) == 0:
                a, b = row_a, row_b
            else:
                a, b = row_b, row_a
            waypoints.append((a, y))
            waypoints.append((b, y))

    compact = []
    for p in waypoints:
        if not compact or compact[-1] != p:
            compact.append(p)

    return tuple(compact), {
        "bounds": f"[{x_min},{x_max}]x[{y_min},{y_max}]",
        "vision_r_sq": vision_radius_sq,
        "stride": stride,
    }


def _rotate_waypoints_to_nearest(waypoints, start_xy):
    if not waypoints:
        return ()

    idx = min(
        range(len(waypoints)),
        key=lambda i: _manhattan(start_xy, waypoints[i]),
    )
    return waypoints[idx:] + waypoints[:idx]


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


def _launch_tile_cardinal_exit_count(local_map, xy) -> int:
    x, y = xy
    count = 0
    for dx, dy in CARDINAL_DELTAS:
        nx = x + dx
        ny = y + dy
        if not local_map.in_bounds(nx, ny):
            continue
        if local_map.get(nx, ny) in PASSABLE_TILES:
            count += 1
    return count


def _launch_tile_is_good(local_map, xy) -> bool:
    if not local_map.in_bounds(xy[0], xy[1]):
        return False
    if _enemy_armoured_transport_on_tile(local_map, xy[0], xy[1]):
        return False
    if _launch_tile_cardinal_exit_count(local_map, xy) > 0:
        return True
    tile = local_map.get(xy[0], xy[1])
    return tile in PASSABLE_TILES and _enemy_transport_kind_on_tile(local_map, xy[0], xy[1]) is None


def _attempt_launch_escape(c: Controller, local_map, cur_xy, rnd: int, uid: int) -> str:
    if _launch_tile_is_good(local_map, cur_xy):
        return "ready"

    best = None
    best_key = None

    for dx, dy in _ADJACENT_DELTAS_8:
        nx = cur_xy[0] + dx
        ny = cur_xy[1] + dy
        if not local_map.in_bounds(nx, ny):
            continue

        move_dir = _MOVE_DIRECTION_BY_DELTA_8.get((dx, dy))
        if move_dir is None:
            continue

        exit_count = _launch_tile_cardinal_exit_count(local_map, (nx, ny))
        enemy_kind = _enemy_transport_kind_on_tile(local_map, nx, ny)
        armoured = 1 if enemy_kind == "armoured" else 0
        replaceable = 1 if enemy_kind == "replaceable" else 0
        tile = local_map.get(nx, ny)
        tile_priority = 0 if tile in WALKABLE_TILES else 1

        can_move_now = False
        if c.get_move_cooldown() == 0:
            try:
                can_move_now = bool(c.can_move(move_dir))
            except GameError:
                can_move_now = False

        can_build_road = False
        if c.get_action_cooldown() == 0:
            try:
                can_build_road = bool(c.can_build_road(Position(nx, ny)))
            except GameError:
                can_build_road = False

        if not can_move_now and not can_build_road:
            continue

        key = (
            armoured,
            replaceable,
            -exit_count,
            tile_priority,
            abs(nx - cur_xy[0]) + abs(ny - cur_xy[1]),
            nx,
            ny,
        )
        if best_key is None or key < best_key:
            best_key = key
            best = (nx, ny, move_dir, can_move_now, can_build_road)

    if best is None:
        return "ready"

    nx, ny, move_dir, can_move_now, can_build_road = best

    if can_move_now:
        try:
            c.move(move_dir)
            log_event(
                rnd,
                uid,
                "economy",
                f"({cur_xy[0]},{cur_xy[1]})",
                "economy_launch_escape_move",
                nx=nx,
                ny=ny,
                d=move_dir.name,
            )
            return "moved"
        except GameError:
            pass

    if can_build_road:
        rp = Position(nx, ny)
        try:
            c.build_road(rp)
            log_event(
                rnd,
                uid,
                "economy",
                f"({cur_xy[0]},{cur_xy[1]})",
                "economy_launch_escape_road",
                rx=nx,
                ry=ny,
            )
            if c.get_move_cooldown() == 0:
                try:
                    if c.can_move(move_dir):
                        c.move(move_dir)
                        log_event(
                            rnd,
                            uid,
                            "economy",
                            f"({cur_xy[0]},{cur_xy[1]})",
                            "economy_launch_escape_move",
                            nx=nx,
                            ny=ny,
                            d=move_dir.name,
                        )
                        return "moved"
                except GameError:
                    pass
            return "built"
        except GameError:
            return "wait_cd"

    return "wait_cd"


def _pick_patrol_row_anchor_x(local_map, y: int, preferred_x: int, fallback_x: int):
    in_bounds = getattr(local_map, "in_bounds", None)
    get_tile = getattr(local_map, "get", None)

    if callable(in_bounds) and callable(get_tile):
        if in_bounds(preferred_x, y) and get_tile(preferred_x, y) in PASSABLE_TILES:
            return preferred_x

        step = 1 if fallback_x >= preferred_x else -1
        x = preferred_x
        while x != fallback_x:
            x += step
            if in_bounds(x, y) and get_tile(x, y) in PASSABLE_TILES:
                return x

        if in_bounds(fallback_x, y) and get_tile(fallback_x, y) in PASSABLE_TILES:
            return fallback_x

    return preferred_x


def _find_replacement_waypoint(local_map, cur_xy, target_xy, waypoints, patrol_mode: bool):
    tx, ty = target_xy
    in_bounds = local_map.in_bounds
    get_tile = local_map.get
    existing = set(waypoints)

    plan_budget = 896 if patrol_mode else 640
    # Search close-to-target first, then expand until a reachable surrogate is found.
    radii = (1, 2, 3, 4, 5, 6, 8)

    for r in radii:
        best = None
        best_key = None

        for dy in range(-r, r + 1):
            for dx in range(-r, r + 1):
                if max(abs(dx), abs(dy)) != r:
                    continue

                nx = tx + dx
                ny = ty + dy
                cand = (nx, ny)

                if not in_bounds(nx, ny):
                    continue
                if cand in existing:
                    continue
                if get_tile(nx, ny) not in PASSABLE_TILES:
                    continue

                steps = _astar_cardinal_plan(
                    local_map,
                    cur_xy,
                    cand,
                    max_expansions=plan_budget,
                )
                if not steps and cur_xy != cand:
                    continue

                key = (
                    abs(dx) + abs(dy),
                    len(steps),
                    abs(nx - cur_xy[0]) + abs(ny - cur_xy[1]),
                    nx,
                    ny,
                )
                if best_key is None or key < best_key:
                    best_key = key
                    best = cand

        if best is not None:
            return best

    return None


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


def _astar_cardinal_plan(
    local_map,
    start_xy,
    goal_xy,
    max_expansions: int,
    walkable_only: bool = False,
    tile_passable_fn=None,
    max_time_us: int | None = None,
    planner_stats: dict | None = None,
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
    h0 = abs(sx - gx) + abs(sy - gy)
    heapq.heappush(open_heap, (h0 + (h0 >> 2), 0, sx, sy))

    parent = {}
    g_score = {(sx, sy): 0}
    closed = set()

    timed_out = False
    deadline_ns = None
    if max_time_us is not None and max_time_us > 0:
        deadline_ns = time.perf_counter_ns() + (int(max_time_us) * 1000)

    expansions = 0
    while open_heap and expansions < max_expansions:
        if deadline_ns is not None and time.perf_counter_ns() >= deadline_ns:
            timed_out = True
            break

        _, g, x, y = heapq.heappop(open_heap)
        node = (x, y)
        if node in closed:
            continue
        closed.add(node)

        if x == gx and y == gy:
            return _reconstruct_path(parent, start_xy, goal_xy)

        expansions += 1
        ng = g + 1
        for dx, dy in CARDINAL_DELTAS:
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

            old_g = g_score.get(n)
            if old_g is not None and ng >= old_g:
                continue

            g_score[n] = ng
            parent[n] = node
            h = abs(nx - gx) + abs(ny - gy)
            f = ng + h + (h >> 2)
            heapq.heappush(open_heap, (f, ng, nx, ny))

    if planner_stats is not None:
        planner_stats["timed_out"] = timed_out

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


def _execute_step_toward(c: Controller, local_map, cur_xy, nxt_xy, rnd: int, uid: int):
    cx, cy = cur_xy
    nx, ny = nxt_xy
    dx = nx - cx
    dy = ny - cy
    move_dir = _CARDINAL_DIRECTION_BY_DELTA.get((dx, dy))
    if move_dir is None:
        return "move_blocked"

    nxt_pos = Position(nx, ny)
    nxt_tile = local_map.get(nx, ny)
    needs_road = nxt_tile not in WALKABLE_TILES and nxt_tile != MAP_ROAD

    # Movement truth comes from the engine. Try moving first even if local-map
    # tile classification says "non-walkable" (map state can be stale).
    if c.get_move_cooldown() == 0:
        try:
            if c.can_move(move_dir):
                c.move(move_dir)
                log_event(
                    rnd,
                    uid,
                    "economy",
                    f"({cx},{cy})",
                    "economy_move_step",
                    nx=nx,
                    ny=ny,
                    d=move_dir.name,
                )
                return "moved"
        except GameError:
            pass

    # For non-walkable planned steps, build road first.
    # Build action is independent from move cooldown.
    if needs_road:
        if c.get_action_cooldown() > 0:
            return "wait_cd"

        # Explicit preflight before road-build attempts.
        # This prevents trying to road over occupied/ore/wall tiles.
        try:
            if c.get_tile_building_id(nxt_pos) is not None:
                return "road_invalid"
        except GameError:
            pass
        try:
            if c.get_tile_env(nxt_pos) != Environment.EMPTY:
                return "road_invalid"
        except GameError:
            pass

        try:
            if not c.can_build_road(nxt_pos):
                return "road_invalid"
            c.build_road(nxt_pos)
            log_event(
                rnd,
                uid,
                "economy",
                f"({cx},{cy})",
                "economy_built_road",
                rx=nx,
                ry=ny,
            )

            try:
                if c.get_move_cooldown() == 0 and c.can_move(move_dir):
                    c.move(move_dir)
                    log_event(
                        rnd,
                        uid,
                        "economy",
                        f"({cx},{cy})",
                        "economy_move_step",
                        nx=nx,
                        ny=ny,
                        d=move_dir.name,
                    )
                    return "moved"
            except GameError:
                return "built"
            return "built"
        except GameError:
            return "road_invalid"

    # Next tile is already walkable: move only.
    if c.get_move_cooldown() > 0:
        return "wait_cd"

    return "move_blocked"


def _manhattan(a, b):
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


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