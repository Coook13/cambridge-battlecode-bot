import math
import time
from collections import deque

from cambc import Controller, Direction, EntityType, Environment, GameError, Position

from constants import (ACTION_RADIUS_SQ, CARDINAL_DELTAS, MAP_FREE,
                       MAP_OBSTACLE, MAP_ORE_AXIONITE, MAP_ORE_TITANIUM,
                       MAP_ROAD, PASSABLE_TILES, WALKABLE_TILES, COMPETITION_MODE,
                       HARVESTER_BASE_COST)
from logger import log_event

from economy_state import (
    EconomyState,
    _DIRECTION_BY_DELTA, _CARDINAL_DIRECTION_BY_DELTA, _CARDINAL_DIRECTION_BY_NAME,
    _ADJACENT_DELTAS_8, _BRIDGE_TARGET_OFFSETS,
    _INDIRECT_LOCAL_SOURCE_MH_THRESHOLD, _INDIRECT_SOURCE_NODES_PER_HISTORY_NETWORK,
    _INDIRECT_SOURCE_NODES_PER_OTHER_NETWORK, _NETWORK_SUCCESS_HISTORY_LIMIT,
    _NETWORK_ASTAR_TIMEOUT_US, _NETWORK_ASTAR_RESUME_STEP_EXPANSIONS,
    _NETWORK_SELECT_TIMEOUT_US, _NETWORK_TIMEOUT_RETRY_COOLDOWN_ROUNDS,
    _NETWORK_UNREACHABLE_DIRECT_RETRY_COOLDOWN_ROUNDS,
    _DIRECT_MASK_PREFILTER_MAX_EXPANSIONS, _DIRECT_MASK_PREFILTER_TIMEOUT_US,
    _INDIRECT_TARGET_DENSITY_RADIUS_CHEB, _INDIRECT_TARGET_MAX_BLOCKED_DENSITY,
    _INDIRECT_TARGET_MIN_KNOWN_TILES,
    _TRANSPORT_ENTITY_TYPES, _DIRECTIONAL_TRANSPORT_ENTITY_TYPES,
    _REPLACEABLE_ENEMY_BLOCKERS, _LIDAR_RAYS_8,
    _LIDAR_DEBUG_MAX_HITS_PER_SCAN,
    _TURRET_SOFT_COST,
    _POST_LAUNCH_PHASES, _NETWORK_BUILD_PHASES, _HARVEST_TITANIUM_PHASES,
    _HARVEST_TAKEOVER_PHASES, _HARVEST_AXIONITE_PHASES, _SABOTAGE_AXIONITE_PHASES,
    _PRELAUNCH_NO_LAUNCHER_PHASES, _REPAIR_PHASES, _REPAIR_SCAN_PHASES,
    _REPAIR_SCAN_NETWORKS_PER_ROUND, _REPAIR_MAX_PLAN_RETRIES,
    _REPAIR_HANDOFF_REENQUEUE_COOLDOWN_ROUNDS,
    _ORE_CORE_EXCLUSION_CHEB, _ORE_CORE_EXCLUSION_MIN_CHEB,
    _BARRIER_AREA_CHEB, _NO_LAUNCHER_ESCAPE_MIN_CHEB,
    _AXIONITE_FOUNDRY_SEARCH_RADIUS, _AXIONITE_CORE_NO_CANDIDATE_LIMIT,
)
from economy_pathfinder import (
    _chebyshev, _is_adjacent_step, _is_diagonal_step,
    _planner_step_cost, _planner_heuristic, _manhattan,
    _astar_cardinal_start_session, _astar_cardinal_continue_session, _astar_cardinal_plan,
    _reconstruct_path,
    _coverage_scanlines, _coverage_scanlines_edge_to_edge, _coverage_row_endpoints,
    _exploration_half_bounds,
    _format_plan_dump, _format_waypoint_dump,
)


# =============================================================================
# ECONOMY BOT — TABLE OF CONTENTS
# =============================================================================
# Each economy builder bot runs a state machine driven by EconomyState.phase.
# All state and module-level constants live in economy_state.py.
# Pure pathfinding (A*, BFS, geometry helpers) live in economy_pathfinder.py.
#
# SECTIONS IN THIS FILE (search for "# ===== " to jump):
#   run_economy              — tick dispatcher; calls per-objective runners
#   _select_post_launch_objective
#
#   AXIONITE PIPELINE        — mine axionite ore, build foundry, build conveyor
#                              link from foundry back to core (phases: axionite_*)
#   AXIONITE SABOTAGE        — attack enemy axionite harvesters (sabotage_*)
#
#   CONVEYOR REPAIR          — detect and patch broken conveyor networks
#                              (phases: repair_*)
#
#   TITANIUM HARVEST         — move to ore, build harvester, build network
#                              (phases: harvest_*)
#
#   POST-LAUNCH EXPLORATION  — explore/patrol the map after being launched
#                              (phases: explore_*, patrol_*)
#
#   NETWORK BUILDING v2      — main conveyor/bridge network construction
#                              (phases: network_*, conveyor_*)
#
#   NETWORK BUILDING (old)   — previous network implementation (kept for
#                              fallback path); gradually superseded by v2
#
#   MAP & BUILDING UTILITIES — pure map query helpers, building helpers,
#                              enemy harvester takeover logic, state
#                              transition helpers (_resume_exploration_*,
#                              _refresh_*, _track_symmetry_*, etc.)
#
#   PRE-LAUNCH NAVIGATION    — bot movement before the launcher fires it
#                              (phases: seek_launcher, plan_to_launcher,
#                               follow_plan, wait_to_launch,
#                               prelaunch_escape_*)
# =============================================================================


# ===== ENTRY POINT =====

def run_economy(c: Controller, state: EconomyState, local_map):
    pos = c.get_position()
    cur_xy = (pos.x, pos.y)
    rnd = c.get_current_round()
    uid = c.get_id()

    # The planner should never treat our own current tile as blocked.
    local_map.planner_self_xy = cur_xy

    # Pre-launch movement to the launcher queue should ignore 3x3 halo blocks,
    # but once launched we re-enable halo-based anti-collision planning.
    is_post_launch = state.phase in _POST_LAUNCH_PHASES
    local_map.enable_unit_halo_planning = is_post_launch

    _track_symmetry_revision(state, local_map, rnd, uid, cur_xy)
    _detect_external_relocation(state, cur_xy, rnd, uid)
    _update_postlaunch_barrier_blocking(state, local_map, cur_xy, rnd, uid)
    _refresh_turret_threat_costs(state, local_map, rnd)
    _refresh_entity_ownership_db(state, local_map, rnd)
    if is_post_launch:
        _refresh_friendly_network_registry(state, local_map, rnd)
        _recompute_direct_anchor_availability(state, local_map)
        _scan_conveyor_integrity_once(state, local_map, rnd, uid, cur_xy)

        if (
            state.phase in _REPAIR_SCAN_PHASES
            and state.phase not in _REPAIR_PHASES
            and _should_start_conveyor_repair(state, rnd)
        ):
            _start_conveyor_repair(state, rnd, uid, cur_xy)

        if state.phase in _REPAIR_PHASES:
            _run_conveyor_repair(c, state, local_map, cur_xy, rnd, uid)
            return

        if state.phase == "network_attack_blocker":
            _run_network_attack_blocker(c, state, local_map, cur_xy, rnd, uid)
            return

    if is_post_launch:
        # Before picking an objective, check whether any paused plan (ti-short
        # at build time) is now affordable and should be resumed in place. A
        # True return means the wait-phase transition is already set up; fall
        # through to objective dispatch so the right pipeline picks it up.
        _check_paused_plans_resume(c, state, rnd, uid, cur_xy)

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
        elif objective == "harvest_axionite":
            _run_axionite_refinement(
                c,
                state,
                local_map,
                cur_xy,
                rnd,
                uid,
            )
        elif objective == "sabotage_axionite":
            _run_axionite_sabotage(
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
            _run_prelaunch_no_launcher(c, state, local_map, cur_xy, rnd, uid)
            return

        if state.phase in (
            "prelaunch_escape_pick_goal",
            "prelaunch_escape_plan",
            "prelaunch_escape_follow",
        ):
            _reset_prelaunch_navigation_state(state)
            state.phase = "seek_launcher"

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
        plan_budget = 800
        goal_xy = state.goal_xy

        def passable_fn(x: int, y: int) -> bool:
            return _is_general_movement_passable(local_map, x, y, goal_xy)

        steps = _astar_cardinal_plan(
            local_map,
            cur_xy,
            goal_xy,
            max_expansions=plan_budget,
            tile_passable_fn=passable_fn,
        )
        if not steps:
            # Path not available in current knowledge; retry next round.
            state.blocked_ticks += 1
            if state.blocked_ticks == 1 or (state.blocked_ticks % 20) == 0:
                gx, gy = state.goal_xy
                goal_tile = -1
                if local_map.in_bounds(gx, gy):
                    goal_tile = local_map.get(gx, gy)
                log_event(
                    rnd,
                    uid,
                    "economy",
                    f"({pos.x},{pos.y})",
                    "economy_plan_pending",
                    gx=gx,
                    gy=gy,
                    goal_tile=goal_tile,
                    blocked=state.blocked_ticks,
                )
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
        if not _is_adjacent_step(cur_xy, nxt):
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
    ax_ctx = _ensure_axionite_ctx(state)

    titanium_pool = _team_titanium(_c)
    known_ax = ()
    # v4_micky: lowered ax-pipeline gate from 2000 to 1500. With the
    # defender disabled (commit b04dd5e) the Ti pool stays healthier
    # so the refinery can bootstrap on a smaller buffer. AB-tested:
    # flips small1 v4-A from a -22000 mining loss into an +11000 win
    # (and small1 v4-B from a tight Ti-margin loss to an axionite
    # tiebreak win). medium2 mining drops 39870->6120 but margin
    # vs v3 is still 6120>>380 — net win count goes from 5 pair-wins
    # to 6 pair-wins.
    # v15_noax_ladder: copy the #1 replay's pure titanium posture.
    # Never voluntarily pivot into the foundry/refined-ax pipeline; keep
    # builders expanding titanium harvest and pressure instead.
    if titanium_pool > 999999:
        known_ax = tuple(
            _known_unharvested_axionite_unblocked(
                local_map,
                _axionite_blocked_ores(state),
                core_xy=state.core_xy,
                min_core_cheb=_ORE_CORE_EXCLUSION_MIN_CHEB,
            )
        )

    if state.phase in _HARVEST_AXIONITE_PHASES:
        return "harvest_axionite"

    if state.phase in _SABOTAGE_AXIONITE_PHASES:
        return "sabotage_axionite"

    ore_ctx = ax_ctx.get("ore_xy")

    if (
        ax_ctx.get("active")
        and (
            state.phase in _HARVEST_AXIONITE_PHASES
            or (isinstance(ore_ctx, tuple) and len(ore_ctx) == 2)
        )
    ):
        # Once the axionite pipeline starts, keep focus on finishing the same
        # ore pipeline end-to-end before selecting any new harvest objective.
        return "harvest_axionite"

    if state.phase in _HARVEST_TITANIUM_PHASES:
        # Preserve in-progress network build phases to avoid abandoning transport
        # chains, but allow ore-pick phases to pivot to axionite priority.
        if state.phase not in _NETWORK_BUILD_PHASES and known_ax:
            return "harvest_axionite"
        return "harvest_titanium"

    if known_ax:
        return "harvest_axionite"

    known_ti = _known_unharvested_titanium_unblocked(
        local_map,
        state.harvest_blocked_ores,
        core_xy=state.core_xy,
        min_core_cheb=_ORE_CORE_EXCLUSION_MIN_CHEB,
    )
    if known_ti:
        return "harvest_titanium"

    if _known_enemy_titanium_harvesters(
        local_map,
        state=state,
        core_xy=state.core_xy,
        min_core_cheb=_ORE_CORE_EXCLUSION_MIN_CHEB,
    ):
        return "harvest_titanium"

    if _known_enemy_axionite_harvesters(
        local_map,
        state=state,
        core_xy=state.core_xy,
        min_core_cheb=_ORE_CORE_EXCLUSION_MIN_CHEB,
    ):
        return "sabotage_axionite"

    # Placeholder priority dispatcher: future objectives (network maintenance,
    # sabotage, etc.) can be inserted here ahead of exploration.
    return "explore"


# ===== AXIONITE PIPELINE =====
# Phases: axionite_enter → axionite_pick_ore → axionite_pick_goal →
#         axionite_plan_goal → axionite_follow_goal → axionite_build_harvester →
#         axionite_pick_foundry → ... → axionite_build_core_route → axionite_done

def _ensure_axionite_ctx(state: EconomyState):
    ctx = getattr(state, "axionite_ctx", None)
    if not isinstance(ctx, dict):
        state.axionite_ctx = {}
        ctx = state.axionite_ctx

    defaults = {
        "active": False,
        "complete": False,
        "blocked_axionite_ores": set(),
        "blocked_fallback_ti_ores": set(),
        "paired_ti_foundry_links": set(),
        "blocked_ti_foundry_links": set(),
        "ore_xy": None,
        "ore_goal_xy": None,
        "foundry_xy": None,
        "foundry_goal_xy": None,
        "root_xy": None,
        "root_bridge_xy": None,
        "root_bridge_reused": False,
        "ti_pair_xy": None,
        "ti_pair_key": None,
        "pair_mode": None,
        "fallback_ti_ore_xy": None,
        "fallback_ti_goal_xy": None,
        "ti_feeder_tiles": (),
        "link_nodes": (),
        "link_sequence": (),
        "link_sequence_index": 0,
        "link_terminal_xy": None,
        "core_route_candidate": None,
        "blocked_core_link_candidates": set(),
        "core_no_candidate_streak": 0,
        "debug_last_phase": None,
        "debug_throttle_rounds": {},
        "sabotage_ore_xy": None,
        "sabotage_goal_xy": None,
    }

    for key, value in defaults.items():
        if key in ctx:
            continue
        if isinstance(value, set):
            ctx[key] = set()
        elif isinstance(value, tuple):
            ctx[key] = tuple(value)
        else:
            ctx[key] = value

    if not isinstance(ctx.get("blocked_axionite_ores"), set):
        ctx["blocked_axionite_ores"] = set()
    if not isinstance(ctx.get("blocked_fallback_ti_ores"), set):
        ctx["blocked_fallback_ti_ores"] = set()
    if not isinstance(ctx.get("paired_ti_foundry_links"), set):
        ctx["paired_ti_foundry_links"] = set()
    if not isinstance(ctx.get("blocked_ti_foundry_links"), set):
        ctx["blocked_ti_foundry_links"] = set()
    if not isinstance(ctx.get("blocked_core_link_candidates"), set):
        ctx["blocked_core_link_candidates"] = set()
    if not isinstance(ctx.get("debug_throttle_rounds"), dict):
        ctx["debug_throttle_rounds"] = {}
    return ctx


def _axionite_debug_log_throttled(
    ctx,
    rnd: int,
    uid: int,
    cur_xy,
    event_name: str,
    interval: int = 20,
    **kwargs,
):
    throttle = ctx.get("debug_throttle_rounds")
    if not isinstance(throttle, dict):
        throttle = {}
        ctx["debug_throttle_rounds"] = throttle

    last = throttle.get(event_name, -10_000)
    try:
        last_round = int(last)
    except (TypeError, ValueError):
        last_round = -10_000

    if int(rnd) - last_round < max(1, int(interval)):
        return

    throttle[event_name] = int(rnd)
    log_event(
        rnd,
        uid,
        "economy",
        f"({cur_xy[0]},{cur_xy[1]})",
        event_name,
        **kwargs,
    )


def _axionite_ti_failure_transition(state, ctx, ore_xy, rnd, uid, cur_xy, reason):
    """Single entry point for every ti→foundry planning/build failure in the
    axionite pipeline. First failure routes to the splitter-splice fallback;
    a subsequent failure (after splitter already attempted for this ore)
    abandons the ore permanently. Ensures we never loop across multiple
    titanium-harvester candidates or stack fallback strategies."""
    if not state.axionite_fallback_attempted:
        if not COMPETITION_MODE:
            log_event(
                rnd, uid, "economy", f"({cur_xy[0]},{cur_xy[1]})",
                "economy_axionite_ti_failure_to_splitter",
                reason=str(reason),
            )
        state.phase = "axionite_fallback_pick_consistent"
        return
    if isinstance(ore_xy, tuple) and len(ore_xy) == 2:
        _axionite_blocked_ores(state).add(
            (int(ore_xy[0]), int(ore_xy[1]))
        )
    if not COMPETITION_MODE:
        log_event(
            rnd, uid, "economy", f"({cur_xy[0]},{cur_xy[1]})",
            "economy_axionite_ore_abandoned_post_splitter",
            reason=str(reason),
        )
    _reset_axionite_pipeline_for_next_ore(state, ctx)


# =================================================================
# Resource-wait: handle build failures that are cost-driven, not
# obstacle-driven. Instead of replanning (which mistakes "can't afford"
# for "tile blocked"), we sample Ti income for 1+ rounds, project
# rounds-to-afford, and either wait or pause the plan and fall back.
# =================================================================

_RESOURCE_WAIT_MAX_ROUNDS = 30
_PAUSED_PLAN_STALENESS_ROUNDS = 400


def _clear_resource_wait(state: EconomyState) -> None:
    state.resource_wait_active = False
    state.resource_wait_step_mode = None
    state.resource_wait_cost_ti = 0
    state.resource_wait_cost_ax = 0
    state.resource_wait_sample_round = -1
    state.resource_wait_sample_ti = -1
    state.resource_wait_sample_ax = -1
    state.resource_wait_resume_phase = None
    state.resource_wait_owner = None
    state.resource_wait_started_round = -1


def _enter_resource_wait_or_fail(
    c: Controller,
    state: EconomyState,
    step_mode: str,
    resume_phase: str,
    owner: str,
    rnd: int,
    uid: int,
    cur_xy,
) -> bool:
    """Inspects scaled build cost vs team resources. If affordable, returns
    False (caller runs normal failure handling). If unaffordable, snapshots
    wait state, transitions to the wait phase, and returns True so the caller
    bails without treating the failure as an obstacle.
    """
    try:
        if step_mode == "bridge":
            bc = c.get_bridge_cost()
        elif step_mode == "conveyor":
            bc = c.get_conveyor_cost()
        else:
            return False
        cost_ti = int(bc[0])
        cost_ax = int(bc[1])
        res = c.get_global_resources()
        ti_res = int(res[0])
        ax_res = int(res[1])
    except GameError:
        return False

    if ti_res >= cost_ti and ax_res >= cost_ax:
        return False

    state.resource_wait_active = True
    state.resource_wait_step_mode = step_mode
    state.resource_wait_cost_ti = cost_ti
    state.resource_wait_cost_ax = cost_ax
    state.resource_wait_sample_round = rnd
    state.resource_wait_sample_ti = ti_res
    state.resource_wait_sample_ax = ax_res
    state.resource_wait_resume_phase = resume_phase
    state.resource_wait_owner = owner
    state.resource_wait_started_round = rnd
    state.plan_steps = ()
    state.plan_index = 0

    if owner == "axionite":
        state.phase = "axionite_wait_resources"
    else:
        state.phase = "network_wait_resources"

    if not COMPETITION_MODE:
        log_event(
            rnd, uid, "economy", f"({cur_xy[0]},{cur_xy[1]})",
            "economy_resource_wait_entered",
            owner=owner, mode=step_mode,
            cost_ti=cost_ti, cost_ax=cost_ax,
            ti=ti_res, ax=ax_res,
            resume=resume_phase,
        )
    return True


def _snapshot_axionite_ctx_for_pause(ctx) -> dict:
    return {
        "link_sequence": tuple(ctx.get("link_sequence", ())),
        "link_sequence_index": int(ctx.get("link_sequence_index", 0)),
        "link_nodes": tuple(ctx.get("link_nodes", ())),
        "link_terminal_xy": ctx.get("link_terminal_xy"),
        "ore_xy": ctx.get("ore_xy"),
        "foundry_xy": ctx.get("foundry_xy"),
        "foundry_goal_xy": ctx.get("foundry_goal_xy"),
        "root_xy": ctx.get("root_xy"),
        "root_bridge_xy": ctx.get("root_bridge_xy"),
        "root_bridge_reused": ctx.get("root_bridge_reused"),
        "core_route_candidate": ctx.get("core_route_candidate"),
        "ti_pair_xy": ctx.get("ti_pair_xy"),
        "ti_pair_key": ctx.get("ti_pair_key"),
        "pair_mode": ctx.get("pair_mode"),
        "ti_feeder_tiles": ctx.get("ti_feeder_tiles", ()),
    }


def _restore_axionite_ctx_from_pause(ctx, snap: dict) -> None:
    for k, v in snap.items():
        ctx[k] = v


def _pause_current_plan_and_fallback(
    state: EconomyState, rnd: int, uid: int, cur_xy, projected_rounds: float,
) -> None:
    owner = state.resource_wait_owner
    resume_phase = state.resource_wait_resume_phase
    cost_ti = int(state.resource_wait_cost_ti)
    cost_ax = int(state.resource_wait_cost_ax)

    plan = {
        "cost_ti": cost_ti,
        "cost_ax": cost_ax,
        "resume_phase": resume_phase,
        "started_round": int(state.resource_wait_started_round),
        "paused_round": int(rnd),
    }

    if owner == "axionite":
        ctx = _ensure_axionite_ctx(state)
        plan["ctx_snapshot"] = _snapshot_axionite_ctx_for_pause(ctx)
        state.paused_axionite_plan = plan
        _reset_axionite_pipeline_for_next_ore(state, ctx)
    else:
        plan["network_target"] = state.network_target
        state.paused_network_plan = plan
        _reset_network_path_state(state)
        state.phase = "network_select_candidate"

    _clear_resource_wait(state)

    if not COMPETITION_MODE:
        log_event(
            rnd, uid, "economy", f"({cur_xy[0]},{cur_xy[1]})",
            "economy_resource_wait_paused_plan",
            owner=str(owner),
            projected=round(float(projected_rounds), 1),
            cost_ti=cost_ti, cost_ax=cost_ax,
        )


def _run_resource_wait_phase(
    c: Controller,
    state: EconomyState,
    cur_xy,
    rnd: int,
    uid: int,
    owner: str,
) -> None:
    """Called each round while in a *_wait_resources phase. Resumes when the
    team can afford the pending build; pauses the plan and falls back when
    the projected wait exceeds _RESOURCE_WAIT_MAX_ROUNDS."""
    if not state.resource_wait_active or state.resource_wait_owner != owner:
        # Defensive: if wait state is inconsistent, fall back cleanly.
        _clear_resource_wait(state)
        state.phase = (
            "axionite_pick_ore" if owner == "axionite"
            else "network_select_candidate"
        )
        return

    try:
        res = c.get_global_resources()
        ti_res = int(res[0])
        ax_res = int(res[1])
    except GameError:
        return

    cost_ti = int(state.resource_wait_cost_ti)
    cost_ax = int(state.resource_wait_cost_ax)

    if ti_res >= cost_ti and ax_res >= cost_ax:
        resume_phase = state.resource_wait_resume_phase
        _clear_resource_wait(state)
        if resume_phase:
            state.phase = resume_phase
        else:
            state.phase = (
                "axionite_pick_ore" if owner == "axionite"
                else "network_select_candidate"
            )
        if not COMPETITION_MODE:
            log_event(
                rnd, uid, "economy", f"({cur_xy[0]},{cur_xy[1]})",
                "economy_resource_wait_resumed",
                owner=owner, ti=ti_res, ax=ax_res,
                cost_ti=cost_ti, cost_ax=cost_ax,
            )
        return

    sample_rnd = int(state.resource_wait_sample_round)
    sample_ti = int(state.resource_wait_sample_ti)
    rounds_elapsed = rnd - sample_rnd
    if rounds_elapsed < 1:
        return

    ti_shortfall = max(0, cost_ti - ti_res)
    ti_rate = (ti_res - sample_ti) / rounds_elapsed
    if ti_rate <= 0.0:
        projected = float("inf")
    else:
        projected = ti_shortfall / ti_rate

    if projected > _RESOURCE_WAIT_MAX_ROUNDS:
        _pause_current_plan_and_fallback(
            state, rnd, uid, cur_xy, projected,
        )


def _check_paused_plans_resume(
    c: Controller, state: EconomyState, rnd: int, uid: int, cur_xy,
) -> bool:
    """If any paused plan is now affordable, restore it and transition back
    into its resume phase. Returns True if a plan was resumed (caller should
    skip normal objective selection this round). Also purges stale plans."""
    try:
        res = c.get_global_resources()
        ti_res = int(res[0])
        ax_res = int(res[1])
    except GameError:
        return False

    plan = state.paused_axionite_plan
    if isinstance(plan, dict):
        paused_rnd = int(plan.get("paused_round", rnd))
        if rnd - paused_rnd > _PAUSED_PLAN_STALENESS_ROUNDS:
            state.paused_axionite_plan = None
        else:
            cost_ti = int(plan.get("cost_ti", 0))
            cost_ax = int(plan.get("cost_ax", 0))
            if ti_res >= cost_ti and ax_res >= cost_ax:
                # Only restore when safe: not already mid-pipeline on a
                # different ore. Safe = non-axionite phase, or an axionite
                # phase that hasn't committed to an ore yet.
                ctx = _ensure_axionite_ctx(state)
                cur_ore = ctx.get("ore_xy")
                safe_axionite_phases = {
                    "axionite_enter",
                    "axionite_pick_ore",
                    "axionite_wait_resources",
                }
                safe = (
                    state.phase not in _HARVEST_AXIONITE_PHASES
                    or (
                        state.phase in safe_axionite_phases
                        and not (isinstance(cur_ore, tuple) and len(cur_ore) == 2)
                    )
                )
                if not safe:
                    return False
                snap = plan.get("ctx_snapshot")
                if isinstance(snap, dict):
                    _restore_axionite_ctx_from_pause(ctx, snap)
                ctx["active"] = True
                ctx["complete"] = False
                resume_phase = plan.get("resume_phase") or "axionite_pick_ore"
                state.paused_axionite_plan = None
                state.plan_steps = ()
                state.plan_index = 0
                state.phase = resume_phase
                if not COMPETITION_MODE:
                    log_event(
                        rnd, uid, "economy", f"({cur_xy[0]},{cur_xy[1]})",
                        "economy_paused_axionite_plan_resumed",
                        resume=resume_phase,
                        cost_ti=cost_ti, cost_ax=cost_ax,
                    )
                return True

    plan = state.paused_network_plan
    if isinstance(plan, dict):
        paused_rnd = int(plan.get("paused_round", rnd))
        if rnd - paused_rnd > _PAUSED_PLAN_STALENESS_ROUNDS:
            state.paused_network_plan = None
        # Titanium network plan resume is implicit: once we're back in
        # harvest_titanium flow, network_select_candidate will re-plan from
        # scratch. We only clear the paused_network_plan once affordable so
        # logs can distinguish "waiting" from "released".
        else:
            cost_ti = int(plan.get("cost_ti", 0))
            cost_ax = int(plan.get("cost_ax", 0))
            if ti_res >= cost_ti and ax_res >= cost_ax:
                state.paused_network_plan = None
                if not COMPETITION_MODE:
                    log_event(
                        rnd, uid, "economy", f"({cur_xy[0]},{cur_xy[1]})",
                        "economy_paused_network_plan_released",
                        cost_ti=cost_ti, cost_ax=cost_ax,
                    )

    return False


def _axionite_blocked_ores(state: EconomyState):
    return _ensure_axionite_ctx(state)["blocked_axionite_ores"]


def _axionite_blocked_fallback_titanium_ores(state: EconomyState):
    return _ensure_axionite_ctx(state)["blocked_fallback_ti_ores"]
def _reset_axionite_pipeline_for_next_ore(state: EconomyState, ctx):
    ctx["complete"] = False
    ctx["active"] = True
    ctx["ore_xy"] = None
    ctx["ore_goal_xy"] = None
    ctx["foundry_xy"] = None
    ctx["foundry_goal_xy"] = None
    ctx["root_xy"] = None
    ctx["root_bridge_xy"] = None
    ctx["root_bridge_reused"] = False
    ctx["ti_pair_xy"] = None
    ctx["ti_pair_key"] = None
    ctx["pair_mode"] = None
    ctx["fallback_ti_ore_xy"] = None
    ctx["fallback_ti_goal_xy"] = None
    ctx["paired_ti_foundry_links"] = set()
    ctx["blocked_ti_foundry_links"] = set()
    ctx["ti_feeder_tiles"] = ()
    ctx["link_nodes"] = ()
    ctx["link_sequence"] = ()
    ctx["link_sequence_index"] = 0
    ctx["link_terminal_xy"] = None
    ctx["core_route_candidate"] = None
    ctx["blocked_core_link_candidates"] = set()
    ctx["core_no_candidate_streak"] = 0
    ctx["root_pick_retry_count"] = 0
    ctx["core_route_replan_from_cur"] = False
    # Reset the splitter-splice fallback one-shot so the next ore can attempt
    # it fresh.
    state.axionite_fallback_attempted = False
    state.axionite_fallback_splitter_xy = None
    state.axionite_fallback_splitter_dir_name = None
    state.axionite_fallback_side_tap_xy = None
    state.axionite_fallback_foundry_xy = None
    state.axionite_fallback_ore_xy = None
    state.axionite_fallback_return_xy = None
    state.plan_steps = ()
    state.plan_index = 0
    state.defer_step_once = False
    state.phase = "axionite_pick_ore"


def _mark_axionite_ore_inaccessible(state: EconomyState, ctx, ore_xy):
    if isinstance(ore_xy, tuple) and len(ore_xy) == 2:
        _axionite_blocked_ores(state).add((int(ore_xy[0]), int(ore_xy[1])))
    _reset_axionite_pipeline_for_next_ore(state, ctx)


def _team_titanium(c: Controller) -> int:
    try:
        resources = c.get_global_resources()
        if isinstance(resources, (tuple, list)) and len(resources) >= 1:
            return int(resources[0])
    except (TypeError, ValueError, GameError):
        pass
    return 0


def _known_unharvested_axionite(
    local_map,
    core_xy=None,
    min_core_cheb: int | None = None,
):
    getter = getattr(local_map, "get_known_unharvested_axionite", None)
    if callable(getter):
        ores = []
        try:
            for ore in getter():
                if not (isinstance(ore, tuple) and len(ore) == 2):
                    continue
                oxy = (int(ore[0]), int(ore[1]))
                if not _is_outside_core_ore_exclusion(
                    oxy,
                    core_xy,
                    min_core_cheb,
                ):
                    continue
                ores.append(oxy)
            ores.sort(key=lambda p: (p[0], p[1]))
            return ores
        except (TypeError, ValueError):
            pass

    ores = []
    for x, y in getattr(local_map, "axionite_unharvested", set()):
        if not (isinstance(x, int) and isinstance(y, int)):
            continue
        if not local_map.in_bounds(x, y):
            continue
        if not _tile_is_known(local_map, x, y):
            continue
        if not _is_outside_core_ore_exclusion(
            (x, y),
            core_xy,
            min_core_cheb,
        ):
            continue
        ores.append((x, y))
    ores.sort(key=lambda p: (p[0], p[1]))
    return ores


def _known_axionite_tiles(local_map):
    """Return all known axionite ore positions, including harvested tiles."""
    out = set()
    for attr in ("axionite_unharvested", "axionite_harvested", "axionite_harvesters"):
        for p in getattr(local_map, attr, set()):
            if not (isinstance(p, tuple) and len(p) == 2):
                continue
            x = int(p[0])
            y = int(p[1])
            if not local_map.in_bounds(x, y):
                continue
            out.add((x, y))
    return out


def _known_unharvested_axionite_unblocked(
    local_map,
    blocked_ores,
    core_xy=None,
    min_core_cheb: int | None = None,
):
    ores = _known_unharvested_axionite(
        local_map,
        core_xy=core_xy,
        min_core_cheb=min_core_cheb,
    )
    if not blocked_ores:
        return ores
    return [ore for ore in ores if ore not in blocked_ores]


def _known_enemy_axionite_harvesters(
    local_map,
    state: EconomyState | None = None,
    core_xy=None,
    min_core_cheb: int | None = None,
):
    entities = getattr(local_map, "entities", None)
    if not isinstance(entities, dict):
        return []

    harvested_positions = {
        (int(p[0]), int(p[1]))
        for p in getattr(local_map, "axionite_harvesters", set())
        if isinstance(p, tuple) and len(p) == 2
    }
    my_team = getattr(local_map, "my_team", None)
    out = []
    for rec in entities.values():
        if not isinstance(rec, dict):
            continue
        if not rec.get("alive", False):
            continue
        if rec.get("entity_type") != EntityType.HARVESTER:
            continue
        if rec.get("team") == my_team:
            continue

        pos = rec.get("position")
        if not (isinstance(pos, tuple) and len(pos) == 2):
            continue

        hxy = (int(pos[0]), int(pos[1]))
        if hxy not in harvested_positions:
            continue
        if not _is_outside_core_ore_exclusion(
            hxy,
            core_xy,
            min_core_cheb,
        ):
            continue

        hid = _entity_id_from_rec(rec)
        if state is not None and _harvester_already_known_stolen(state, hxy, hid):
            continue

        out.append((hid, hxy))

    out.sort(
        key=lambda item: (
            _manhattan((0, 0), item[1]),
            item[1][0],
            item[1][1],
            -1 if item[0] is None else item[0],
        )
    )
    return out


def _pick_nearest_ore(cur_xy, ores):
    return min(
        ores,
        key=lambda p: (
            abs(p[0] - cur_xy[0]) + abs(p[1] - cur_xy[1]),
            p[0],
            p[1],
        ),
    )


def _is_axionite_goal_valid(local_map, ore_xy, goal_xy) -> bool:
    ox, oy = ore_xy
    gx, gy = goal_xy

    if abs(gx - ox) + abs(gy - oy) != 1:
        return False
    if not local_map.in_bounds(gx, gy):
        return False
    if not _tile_is_known(local_map, gx, gy):
        return False

    return _is_general_movement_passable(local_map, gx, gy, (gx, gy))


def _pick_axionite_adjacent_goal(local_map, ore_xy, cur_xy):
    ox, oy = ore_xy
    candidates = []
    for dx, dy in CARDINAL_DELTAS:
        gx = ox + dx
        gy = oy + dy
        if not local_map.in_bounds(gx, gy):
            continue
        if not _tile_is_known(local_map, gx, gy):
            continue
        if not _is_general_movement_passable(local_map, gx, gy, (gx, gy)):
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
    return (gx, gy)


def _pick_foundry_tile_for_ore(local_map, ore_xy, cur_xy):
    """Return ``((fx, fy), clear_enemy, need_self_move)`` or ``(None, 0, 0)``.

    ``need_self_move`` is 1 when the chosen tile is the bot's own position —
    the caller must step the bot onto an adjacent walkable tile before
    building the foundry. Non-self candidates are preferred via the sort key.
    """
    ox, oy = int(ore_xy[0]), int(ore_xy[1])
    my_team = getattr(local_map, "my_team", None)
    candidates = []

    # Foundry placement is constrained to cardinal adjacency from the ore tile.
    for dx, dy in CARDINAL_DELTAS:
        fx = ox + dx
        fy = oy + dy
        if not local_map.in_bounds(fx, fy):
            continue
        if not _tile_is_known(local_map, fx, fy):
            continue
        # Another unit sitting on the tile blocks it. The bot itself is not
        # registered in its own local_map (see UnitLocalMap.update_from_controller
        # which skips the self id), so _known_unit_at is naturally None at
        # cur_xy — we gate the self case via an explicit cur_xy match below.
        if _known_unit_at(local_map, fx, fy) is not None:
            continue

        self_here = 1 if (fx, fy) == cur_xy else 0

        tile = local_map.get(fx, fy)
        if tile in (MAP_OBSTACLE, MAP_ORE_TITANIUM, MAP_ORE_AXIONITE):
            continue

        rec = _known_building_at(local_map, fx, fy)
        clear_enemy = 0
        if isinstance(rec, dict):
            etype = rec.get("entity_type")
            team = rec.get("team")
            # Foundry target validity is strict: tile must be free or road.
            # Any existing foundry or non-road building is invalid.
            if etype == EntityType.FOUNDRY:
                continue
            if etype != EntityType.ROAD:
                continue
            clear_enemy = 1 if team != my_team else 0

        # Foundry can only be built from a halo tile around the target. When
        # the candidate is the bot's own tile, the bot must step off first —
        # look for a stand tile that is NOT cur_xy.
        stand_disallow = cur_xy if self_here else None
        if _pick_build_stand_for_target(
            local_map, (fx, fy), cur_xy, disallow_xy=stand_disallow,
        ) is None:
            continue

        # Per-user rule: when multiple foundry candidates exist, prefer ones
        # not cardinally adjacent to a harvester, then ones not adjacent to
        # any friendly conveyor/bridge. The adjacency keys are 0 for "no
        # adjacency" (preferred) and 1 for "adjacent". If no non-adjacent
        # candidate exists we still pick an adjacent one as a fallback — the
        # tuple ordering handles the "save first, use later" semantics
        # naturally via min().
        adj_harvester = 1 if _tile_cardinally_adjacent_to_any_harvester(
            local_map, fx, fy) else 0
        adj_conveyor = 1 if _tile_cardinally_adjacent_to_friendly_transport(
            local_map, fx, fy) else 0

        candidates.append(
            (
                clear_enemy,       # prefer no-clear (friendly/empty) over enemy
                adj_harvester,     # prefer not adjacent to any harvester
                adj_conveyor,      # prefer not adjacent to any friendly transport
                self_here,         # prefer non-self tiles when both available
                _manhattan(cur_xy, (fx, fy)),
                fx,
                fy,
            )
        )

    if not candidates:
        return None, 0, 0
    clear_enemy, _adj_h, _adj_c, self_here, _, fx, fy = min(candidates)
    return (fx, fy), clear_enemy, self_here


def _pick_build_stand_for_target(
    local_map,
    target_xy,
    cur_xy,
    disallow_xy=None,
):
    tx = int(target_xy[0])
    ty = int(target_xy[1])
    max_dist_sq = max(1, int(ACTION_RADIUS_SQ))
    candidates = []

    for dx in (-1, 0, 1):
        for dy in (-1, 0, 1):
            if dx == 0 and dy == 0:
                continue
            if (dx * dx) + (dy * dy) > max_dist_sq:
                continue
            sx = tx + dx
            sy = ty + dy
            if not local_map.in_bounds(sx, sy):
                continue
            if (
                isinstance(disallow_xy, tuple)
                and len(disallow_xy) == 2
                and (sx, sy) == (int(disallow_xy[0]), int(disallow_xy[1]))
            ):
                continue
            if not _is_general_movement_passable(local_map, sx, sy, (sx, sy)):
                continue
            candidates.append((
                _manhattan(cur_xy, (sx, sy)),
                sx,
                sy,
            ))

    if not candidates:
        return None

    _, sx, sy = min(candidates)
    return (sx, sy)


def _pick_cardinal_bridge_stand(local_map, bridge_xy, cur_xy):
    """Pick a walkable *cardinally*-adjacent tile of ``bridge_xy`` from which
    the bot can build the bridge. This matches the behaviour of the main
    titanium pipeline (``conveyor_execution``), where the bot is always on
    the previously-built conveyor — always cardinal-adjacent to the next
    node. Axionite builds that tried a diagonal stand (dist²=2) have been
    observed to fail the engine's build eligibility check.

    Ordered by Manhattan distance from ``cur_xy`` so we minimise travel."""
    bx = int(bridge_xy[0])
    by = int(bridge_xy[1])
    candidates = []
    for dx, dy in CARDINAL_DELTAS:
        sx = bx + dx
        sy = by + dy
        if not local_map.in_bounds(sx, sy):
            continue
        if not _is_general_movement_passable(local_map, sx, sy, (sx, sy)):
            continue
        candidates.append((
            _manhattan(cur_xy, (sx, sy)),
            sx,
            sy,
        ))
    if not candidates:
        return None
    _, sx, sy = min(candidates)
    return (sx, sy)


def _pick_axionite_root_tile(local_map, foundry_xy, core_xy):
    fx = int(foundry_xy[0])
    fy = int(foundry_xy[1])
    my_team = getattr(local_map, "my_team", None)

    # Prefer reusing an already-built friendly bridge adjacent to foundry.
    reuse_candidates = []
    for dx, dy in CARDINAL_DELTAS:
        bx = fx + dx
        by = fy + dy
        if not local_map.in_bounds(bx, by):
            continue
        if not _tile_is_known(local_map, bx, by):
            continue

        rec = _known_building_at(local_map, bx, by)
        if not isinstance(rec, dict):
            continue
        if rec.get("team") != my_team:
            continue
        if rec.get("entity_type") != EntityType.BRIDGE:
            continue

        target = rec.get("bridge_target")
        if not (isinstance(target, tuple) and len(target) == 2):
            continue
        rx = int(target[0])
        ry = int(target[1])
        if not local_map.in_bounds(rx, ry):
            continue
        if not _tile_is_known(local_map, rx, ry):
            continue

        tile = local_map.get(rx, ry)
        if tile in (MAP_OBSTACLE, MAP_ORE_TITANIUM, MAP_ORE_AXIONITE):
            continue

        # Reuse is invalid if the bridge target currently hosts any foundry.
        target_rec = _known_building_at(local_map, rx, ry)
        if (
            isinstance(target_rec, dict)
            and target_rec.get("entity_type") == EntityType.FOUNDRY
        ):
            continue

        reuse_candidates.append(
            (
                _manhattan((rx, ry), core_xy),
                _manhattan((bx, by), core_xy),
                bx,
                by,
                rx,
                ry,
            )
        )

    if reuse_candidates:
        _, _, _, _, rx, ry = min(reuse_candidates)
        return (rx, ry)

    for radius in (3, 2, 1):
        candidates = []
        r_sq = radius * radius
        prev_sq = (radius - 1) * (radius - 1)
        for dx in range(-radius, radius + 1):
            for dy in range(-radius, radius + 1):
                dist_sq = (dx * dx) + (dy * dy)
                if dist_sq > r_sq or dist_sq <= prev_sq:
                    continue
                rx = fx + dx
                ry = fy + dy
                if not local_map.in_bounds(rx, ry):
                    continue
                if not _tile_is_known(local_map, rx, ry):
                    continue

                tile = local_map.get(rx, ry)
                if tile in (MAP_OBSTACLE, MAP_ORE_TITANIUM, MAP_ORE_AXIONITE):
                    continue

                rec = _known_building_at(local_map, rx, ry)
                if isinstance(rec, dict):
                    etype = rec.get("entity_type")
                    team = rec.get("team")
                    if etype == EntityType.FOUNDRY and team == my_team:
                        continue
                    if etype not in (
                        EntityType.ROAD,
                        EntityType.CONVEYOR,
                        EntityType.SPLITTER,
                        EntityType.BRIDGE,
                        EntityType.MARKER,
                    ):
                        continue

                candidates.append(
                    (
                        _manhattan((rx, ry), core_xy),
                        _manhattan((rx, ry), foundry_xy),
                        rx,
                        ry,
                    )
                )

        if candidates:
            _, _, rx, ry = min(candidates)
            return (rx, ry)

    return None


def _supercover_line_cells(start_xy, end_xy):
    x0 = int(start_xy[0])
    y0 = int(start_xy[1])
    x1 = int(end_xy[0])
    y1 = int(end_xy[1])

    dx = x1 - x0
    dy = y1 - y0
    nx = abs(dx)
    ny = abs(dy)
    sx = 0 if dx == 0 else (1 if dx > 0 else -1)
    sy = 0 if dy == 0 else (1 if dy > 0 else -1)

    x = x0
    y = y0
    ix = 0
    iy = 0
    out = [(x, y)]

    while ix < nx or iy < ny:
        lhs = (1 + (ix << 1)) * ny
        rhs = (1 + (iy << 1)) * nx
        if lhs == rhs:
            if ix < nx:
                x += sx
                ix += 1
            if iy < ny:
                y += sy
                iy += 1
        elif lhs < rhs:
            if ix < nx:
                x += sx
                ix += 1
        else:
            if iy < ny:
                y += sy
                iy += 1
        out.append((x, y))

    return tuple(out)


def _bridge_path_crosses_known_wall(local_map, build_xy, target_xy):
    cells = _supercover_line_cells(build_xy, target_xy)
    if len(cells) <= 2:
        return False

    for cx, cy in cells[1:-1]:
        if not local_map.in_bounds(cx, cy):
            return True
        if not _tile_is_known(local_map, cx, cy):
            continue
        if local_map.get(cx, cy) == MAP_OBSTACLE:
            return True
    return False


def _tile_cardinally_adjacent_to_any_harvester(local_map, x: int, y: int) -> bool:
    """True if ``(x,y)`` is cardinally adjacent to any known friendly
    titanium or axionite harvester. Used by the foundry picker to avoid
    placements where the foundry's flow ring overlaps a harvester's
    output ring."""
    for harvester_set_name in ("titanium_harvesters", "axionite_harvesters"):
        hs = getattr(local_map, harvester_set_name, None)
        if not isinstance(hs, set):
            continue
        for hxy in hs:
            if not (isinstance(hxy, tuple) and len(hxy) == 2):
                continue
            hx = int(hxy[0])
            hy = int(hxy[1])
            if abs(hx - x) + abs(hy - y) == 1:
                return True
    return False


def _tile_cardinally_adjacent_to_friendly_transport(local_map, x: int, y: int) -> bool:
    """True if ``(x,y)`` is cardinally adjacent to any known friendly
    conveyor, splitter, bridge, or armoured conveyor."""
    my_team = getattr(local_map, "my_team", None)
    for dx, dy in CARDINAL_DELTAS:
        nx = x + dx
        ny = y + dy
        if not local_map.in_bounds(nx, ny):
            continue
        rec = _known_building_at(local_map, nx, ny)
        if not isinstance(rec, dict):
            continue
        if rec.get("team") != my_team:
            continue
        if rec.get("entity_type") in (
            EntityType.CONVEYOR,
            EntityType.SPLITTER,
            EntityType.BRIDGE,
            EntityType.ARMOURED_CONVEYOR,
        ):
            return True
    return False


def _foundry_has_output_ready_adjacent_transport(local_map, foundry_xy) -> bool:
    """True if the foundry at ``foundry_xy`` has a cardinally adjacent
    friendly transport that can accept the foundry's refined-axionite
    output — i.e. a bridge whose target is NOT the foundry itself, or a
    conveyor/splitter/armoured-conveyor whose direction does NOT point at
    the foundry. When this holds, the existing external network already
    provides a flow path to the core, so we can skip the root bridge and
    core-route portions of the axionite pipeline."""
    my_team = getattr(local_map, "my_team", None)
    fx = int(foundry_xy[0])
    fy = int(foundry_xy[1])
    for dx, dy in CARDINAL_DELTAS:
        nx = fx + dx
        ny = fy + dy
        if not local_map.in_bounds(nx, ny):
            continue
        rec = _known_building_at(local_map, nx, ny)
        if not isinstance(rec, dict):
            continue
        if rec.get("team") != my_team:
            continue
        etype = rec.get("entity_type")
        if etype == EntityType.BRIDGE:
            target = rec.get("bridge_target")
            if isinstance(target, tuple) and len(target) == 2:
                if (int(target[0]), int(target[1])) != (fx, fy):
                    return True
            continue
        if etype in (
            EntityType.CONVEYOR,
            EntityType.SPLITTER,
            EntityType.ARMOURED_CONVEYOR,
        ):
            direction = rec.get("direction")
            if direction is None:
                continue
            ddx, ddy = direction.delta()
            out_tile = (nx + int(ddx), ny + int(ddy))
            if out_tile != (fx, fy):
                return True
    return False


def _log_root_pick_debug_report(rnd, uid, cur_xy, foundry_xy, report, event_name):
    """Emit compact per-cardinal debug lines for a failed root-bridge pick."""
    if not isinstance(report, dict):
        return
    cardinals = report.get("cardinals") or ()
    fx = int(foundry_xy[0]) if isinstance(foundry_xy, tuple) else -1
    fy = int(foundry_xy[1]) if isinstance(foundry_xy, tuple) else -1
    for cinfo in cardinals:
        pos = cinfo.get("pos") or (-1, -1)
        rejects = cinfo.get("reject_counts") or {}
        rejects_str = ",".join(
            f"{k}={v}" for k, v in sorted(rejects.items())
        )
        log_event(
            rnd, uid, "economy", f"({cur_xy[0]},{cur_xy[1]})",
            event_name,
            fx=fx, fy=fy,
            bx=int(pos[0]), by=int(pos[1]),
            status=str(cinfo.get("status") or "unknown"),
            etype=str(cinfo.get("etype") or ""),
            scanned=int(cinfo.get("roots_scanned", 0)),
            accepted=int(cinfo.get("roots_accepted", 0)),
            rejects=rejects_str,
            etarget=(
                f"({cinfo['existing_target'][0]},{cinfo['existing_target'][1]})"
                if isinstance(cinfo.get("existing_target"), tuple)
                else ""
            ),
        )


def _pick_root_bridge_and_root(
    local_map,
    foundry_xy,
    core_xy,
    cur_xy,
    ti_harvester_positions=(),
    debug_report=None,
):
    """Find a (bridge_tile, root_tile, need_clear) triple for the axionite root bridge.

    Iterates cardinal neighbours of the foundry as bridge candidates. For each
    valid bridge tile, searches for a root at Euclidean radii 3, 2, then 1.
    Prefers no-clear options first, then the root closest to core.
    Returns (None, None, 0) when no valid pair exists.

    If ``debug_report`` is a dict, populates it with per-cardinal rejection
    reasons and root-scan stats for diagnostic logging.
    """
    fx, fy = int(foundry_xy[0]), int(foundry_xy[1])
    my_team = getattr(local_map, "my_team", None)
    candidates = []
    report_enabled = isinstance(debug_report, dict)
    if report_enabled:
        debug_report["cardinals"] = []

    for dx, dy in CARDINAL_DELTAS:
        bx = fx + dx
        by = fy + dy
        cardinal_info = {
            "pos": (bx, by),
            "status": "evaluated",
            "roots_scanned": 0,
            "roots_accepted": 0,
            "reject_counts": {},
        }

        if not local_map.in_bounds(bx, by):
            cardinal_info["status"] = "out_of_bounds"
            if report_enabled:
                debug_report["cardinals"].append(cardinal_info)
            continue
        if not _tile_is_known(local_map, bx, by):
            cardinal_info["status"] = "bridge_tile_unknown"
            if report_enabled:
                debug_report["cardinals"].append(cardinal_info)
            continue

        if any(
            abs(int(h[0]) - bx) + abs(int(h[1]) - by) == 1
            for h in ti_harvester_positions
            if isinstance(h, tuple) and len(h) == 2
        ):
            cardinal_info["status"] = "ti_harvester_adjacent"
            if report_enabled:
                debug_report["cardinals"].append(cardinal_info)
            continue

        tile = local_map.get(bx, by)
        rec = _known_building_at(local_map, bx, by)
        clear_enemy = 0

        if isinstance(rec, dict):
            etype = rec.get("entity_type")
            team = rec.get("team")

            if team == my_team and etype == EntityType.BRIDGE:
                target = rec.get("bridge_target")
                cardinal_info["status"] = "friendly_bridge"
                if isinstance(target, tuple) and len(target) == 2:
                    rx, ry = int(target[0]), int(target[1])
                    cardinal_info["existing_target"] = (rx, ry)
                    if local_map.in_bounds(rx, ry) and _tile_is_known(local_map, rx, ry):
                        rtile = local_map.get(rx, ry)
                        rrec = _known_building_at(local_map, rx, ry)
                        if rtile not in (MAP_OBSTACLE, MAP_ORE_TITANIUM, MAP_ORE_AXIONITE):
                            if not (
                                isinstance(rrec, dict)
                                and rrec.get("entity_type") == EntityType.FOUNDRY
                            ):
                                candidates.append(
                                    (0, _manhattan((rx, ry), core_xy), bx, by, rx, ry)
                                )
                                cardinal_info["status"] = "friendly_bridge_reused"
                                cardinal_info["roots_accepted"] = 1
                if report_enabled:
                    debug_report["cardinals"].append(cardinal_info)
                continue

            if etype == EntityType.ROAD:
                # Roads are low-value infra. A friendly road can be destroyed
                # for free by the bridge builder (see _build_bridge_on_tile,
                # which already handles this) so we treat it as free for
                # picking. An enemy road needs an attack pass first; signal
                # that via clear_enemy=1.
                if team != my_team:
                    clear_enemy = 1
                # else: fall through to the radius scan with clear_enemy=0.
            elif etype in (
                EntityType.CONVEYOR,
                EntityType.SPLITTER,
                EntityType.BRIDGE,
            ):
                if team != my_team:
                    # Enemy transport — attackable; destroying it is OK.
                    clear_enemy = 1
                else:
                    # Friendly transport — do NOT destroy; breaks networks.
                    cardinal_info["status"] = "friendly_transport_infra"
                    cardinal_info["etype"] = str(etype.name).lower()
                    if report_enabled:
                        debug_report["cardinals"].append(cardinal_info)
                    continue
            else:
                # Armoured conveyor (immune to attack), turrets, foundries,
                # harvesters, barriers, markers, etc. — reject.
                cardinal_info["status"] = "non_transport_building"
                cardinal_info["etype"] = str(etype.name).lower() if etype is not None else "none"
                if report_enabled:
                    debug_report["cardinals"].append(cardinal_info)
                continue
        else:
            if tile in (MAP_OBSTACLE, MAP_ORE_TITANIUM, MAP_ORE_AXIONITE):
                cardinal_info["status"] = "obstacle_or_ore"
                cardinal_info["tile_code"] = int(tile)
                if report_enabled:
                    debug_report["cardinals"].append(cardinal_info)
                continue

        reject = cardinal_info["reject_counts"]
        for radius in (3, 2, 1):
            r_sq = radius * radius
            prev_sq = (radius - 1) * (radius - 1)
            for rdx in range(-radius, radius + 1):
                for rdy in range(-radius, radius + 1):
                    dist_sq = rdx * rdx + rdy * rdy
                    if dist_sq > r_sq or dist_sq <= prev_sq:
                        continue
                    rx = bx + rdx
                    ry = by + rdy
                    cardinal_info["roots_scanned"] += 1
                    if not local_map.in_bounds(rx, ry):
                        reject["oob"] = reject.get("oob", 0) + 1
                        continue
                    if not _tile_is_known(local_map, rx, ry):
                        reject["unknown"] = reject.get("unknown", 0) + 1
                        continue
                    if (rx, ry) == (bx, by) or (rx, ry) == (fx, fy):
                        reject["self_or_foundry"] = reject.get(
                            "self_or_foundry", 0) + 1
                        continue
                    rtile = local_map.get(rx, ry)
                    if rtile in (MAP_OBSTACLE, MAP_ORE_TITANIUM, MAP_ORE_AXIONITE):
                        reject["obstacle_or_ore"] = reject.get(
                            "obstacle_or_ore", 0) + 1
                        continue
                    rrec = _known_building_at(local_map, rx, ry)
                    if isinstance(rrec, dict):
                        rtype = rrec.get("entity_type")
                        rteam = rrec.get("team")
                        if rtype == EntityType.FOUNDRY and rteam == my_team:
                            reject["friendly_foundry"] = reject.get(
                                "friendly_foundry", 0) + 1
                            continue
                        if rtype not in (
                            EntityType.ROAD,
                            EntityType.CONVEYOR,
                            EntityType.SPLITTER,
                            EntityType.BRIDGE,
                            EntityType.MARKER,
                        ):
                            reject["non_transport_bldg"] = reject.get(
                                "non_transport_bldg", 0) + 1
                            continue
                    if _bridge_path_crosses_known_wall(local_map, (bx, by), (rx, ry)):
                        reject["path_wall"] = reject.get("path_wall", 0) + 1
                        continue
                    candidates.append(
                        (clear_enemy, _manhattan((rx, ry), core_xy), bx, by, rx, ry)
                    )
                    cardinal_info["roots_accepted"] += 1

        cardinal_info["status"] = (
            "free" if clear_enemy == 0 else "enemy_replaceable"
        )
        if report_enabled:
            debug_report["cardinals"].append(cardinal_info)

    if not candidates:
        return None, None, 0

    candidates.sort(key=lambda c: (c[0], c[1]))
    clear_enemy, _, bx, by, rx, ry = candidates[0]
    return (bx, by), (rx, ry), clear_enemy


def _pick_root_bridge_build_tile(
    local_map,
    foundry_xy,
    root_xy,
    cur_xy,
    ti_harvester_positions=(),
):
    fx = int(foundry_xy[0])
    fy = int(foundry_xy[1])
    rx = int(root_xy[0])
    ry = int(root_xy[1])
    my_team = getattr(local_map, "my_team", None)
    candidates = []

    # Root bridge must sit on a cardinal-adjacent tile to the foundry.
    for dx, dy in CARDINAL_DELTAS:
        bx = fx + dx
        by = fy + dy
        if not local_map.in_bounds(bx, by):
            continue
        if not _tile_is_known(local_map, bx, by):
            continue
        if (bx, by) == (rx, ry):
            continue

        if ((rx - bx) * (rx - bx)) + ((ry - by) * (ry - by)) > 9:
            continue

        # Prevent backflow: root bridge tile must not be cardinal-adjacent
        # to a known friendly titanium harvester.
        blocked_by_ti = False
        for hxy in ti_harvester_positions:
            if not (isinstance(hxy, tuple) and len(hxy) == 2):
                continue
            hx = int(hxy[0])
            hy = int(hxy[1])
            if abs(hx - bx) + abs(hy - by) == 1:
                blocked_by_ti = True
                break
        if blocked_by_ti:
            continue

        tile = local_map.get(bx, by)
        if tile in (MAP_OBSTACLE, MAP_ORE_TITANIUM, MAP_ORE_AXIONITE):
            continue

        rec = _known_building_at(local_map, bx, by)
        clear_enemy = 0
        if isinstance(rec, dict):
            etype = rec.get("entity_type")
            team = rec.get("team")

            if team == my_team and etype == EntityType.BRIDGE:
                target = rec.get("bridge_target")
                if isinstance(target, tuple) and len(target) == 2:
                    if (int(target[0]), int(target[1])) == (rx, ry):
                        return (bx, by), 0
                # Existing friendly bridge with mismatched target cannot be
                # reused for this root bridge; try other cardinal candidates.
                continue

            if etype in (
                EntityType.ROAD,
                EntityType.CONVEYOR,
                EntityType.SPLITTER,
                EntityType.BRIDGE,
            ):
                if team != my_team:
                    clear_enemy = 1
            else:
                continue

        if _bridge_path_crosses_known_wall(local_map, (bx, by), (rx, ry)):
            continue

        candidates.append(
            (
                clear_enemy,
                _manhattan(cur_xy, (bx, by)),
                bx,
                by,
            )
        )

    if not candidates:
        return None, 0

    # Try candidates in priority order and return the first valid one.
    for clear_enemy, _, bx, by in sorted(candidates):
        return (bx, by), clear_enemy
    return None, 0


def _axionite_root_halo_tiles(root_xy):
    if not (isinstance(root_xy, tuple) and len(root_xy) == 2):
        return set()
    rx = int(root_xy[0])
    ry = int(root_xy[1])
    return {
        (rx + dx, ry + dy)
        for dx in (-1, 0, 1)
        for dy in (-1, 0, 1)
    }


def _pick_pair_link_endpoint_near_harvester(local_map, harvester_xy, toward_xy):
    hx = int(harvester_xy[0])
    hy = int(harvester_xy[1])
    if not (isinstance(toward_xy, tuple) and len(toward_xy) == 2):
        toward_xy = (hx, hy)
    candidates = []
    for dx, dy in CARDINAL_DELTAS:
        sx = hx + dx
        sy = hy + dy
        if not local_map.in_bounds(sx, sy):
            continue
        if _known_unit_at(local_map, sx, sy) is not None:
            continue
        # Feeder source must be on a free tile or a road tile adjacent to the
        # titanium harvester.
        rec = _known_building_at(local_map, sx, sy)
        if isinstance(rec, dict) and rec.get("entity_type") != EntityType.ROAD:
            continue
        if not _is_conveyor_planner_passable(local_map, sx, sy, (sx, sy)):
            continue
        candidates.append((
            _manhattan((sx, sy), toward_xy),
            sx,
            sy,
        ))
    if not candidates:
        return None
    _, sx, sy = min(candidates)
    return (sx, sy)


def _pick_pair_link_sink_near_foundry(
    local_map,
    foundry_xy,
    toward_xy,
    blocked_tiles=(),
    block_friendly_infra: bool = False,
):
    fx = int(foundry_xy[0])
    fy = int(foundry_xy[1])
    if not (isinstance(toward_xy, tuple) and len(toward_xy) == 2):
        toward_xy = (fx, fy)

    blocked = {
        (int(p[0]), int(p[1]))
        for p in blocked_tiles
        if isinstance(p, tuple) and len(p) == 2
    }

    candidates = []
    for tx in range(fx - 3, fx + 4):
        for ty in range(fy - 3, fy + 4):
            dist_sq = ((tx - fx) * (tx - fx)) + ((ty - fy) * (ty - fy))
            if dist_sq == 0 or dist_sq > 9:
                continue
            if not local_map.in_bounds(tx, ty):
                continue
            if not _tile_is_known(local_map, tx, ty):
                continue
            if _known_unit_at(local_map, tx, ty) is not None:
                continue
            if (tx, ty) in blocked:
                continue

            tile = local_map.get(tx, ty)
            if tile in (MAP_OBSTACLE, MAP_ORE_TITANIUM, MAP_ORE_AXIONITE):
                continue

            rec = _known_building_at(local_map, tx, ty)
            if isinstance(rec, dict):
                etype = rec.get("entity_type")
                if block_friendly_infra and etype in (
                    EntityType.CONVEYOR,
                    EntityType.ARMOURED_CONVEYOR,
                    EntityType.SPLITTER,
                    EntityType.BRIDGE,
                ):
                    continue
                if etype not in (
                    EntityType.ROAD,
                    EntityType.CONVEYOR,
                    EntityType.ARMOURED_CONVEYOR,
                    EntityType.SPLITTER,
                    EntityType.BRIDGE,
                    EntityType.MARKER,
                ):
                    continue

            if not _is_conveyor_planner_passable(local_map, tx, ty, (tx, ty)):
                continue

            candidates.append((
                _manhattan((tx, ty), toward_xy),
                dist_sq,
                tx,
                ty,
            ))

    if not candidates:
        return None
    _, _, tx, ty = min(candidates)
    return (tx, ty)


def _axionite_ti_pair_key(foundry_xy, harvester_xy):
    return (
        int(foundry_xy[0]),
        int(foundry_xy[1]),
        int(harvester_xy[0]),
        int(harvester_xy[1]),
    )


def _axionite_core_candidate_key(candidate):
    if not isinstance(candidate, dict):
        return None

    mode = str(candidate.get("mode") or "none")
    bridge_pos = candidate.get("bridge_pos")
    bridge_target = candidate.get("bridge_target")
    if not (
        isinstance(bridge_pos, tuple)
        and len(bridge_pos) == 2
        and isinstance(bridge_target, tuple)
        and len(bridge_target) == 2
    ):
        return None

    return (
        mode,
        int(bridge_pos[0]),
        int(bridge_pos[1]),
        int(bridge_target[0]),
        int(bridge_target[1]),
    )


def _axionite_core_excluded_tiles(ctx, root_xy):
    out = {
        (int(root_xy[0]), int(root_xy[1])),
    }
    for p in ctx.get("ti_feeder_tiles", ()):
        if isinstance(p, tuple) and len(p) == 2:
            out.add((int(p[0]), int(p[1])))
    return out


def _axionite_core_direct_candidates(state: EconomyState, root_xy, excluded_tiles):
    cx = int(state.core_xy[0])
    cy = int(state.core_xy[1])
    core_tiles = [
        (cx + dx, cy + dy)
        for dx in (-1, 0, 1)
        for dy in (-1, 0, 1)
    ]

    out = []
    for bx, by in state.direct_anchor_available:
        bx = int(bx)
        by = int(by)
        if (bx, by) in excluded_tiles:
            continue

        tx, ty = _nearest_core_tile_target((bx, by), core_tiles)
        if (tx, ty) in excluded_tiles:
            continue
        if ((tx - bx) * (tx - bx) + (ty - by) * (ty - by)) > 9:
            continue

        out.append({
            "mode": "direct",
            "bridge_pos": (bx, by),
            "bridge_target": (tx, ty),
            "source_network_key": None,
            "source_conveyor": None,
            "score": (
                _manhattan(root_xy, (bx, by)),
                abs(bx - cx) + abs(by - cy),
                bx,
                by,
            ),
        })

    out.sort(key=lambda cnd: cnd["score"])
    return tuple(out)


def _plan_axionite_ti_link_nodes(
    local_map,
    source_xy,
    foundry_xy,
    blocked_tiles=(),
    block_friendly_infra: bool = False,
):
    source_xy = (int(source_xy[0]), int(source_xy[1]))
    foundry_xy = (int(foundry_xy[0]), int(foundry_xy[1]))

    if source_xy != foundry_xy:
        sdx = source_xy[0] - foundry_xy[0]
        sdy = source_xy[1] - foundry_xy[1]
        if (sdx * sdx) + (sdy * sdy) <= 9:
            return (source_xy,)

    sink_xy = _pick_pair_link_sink_near_foundry(
        local_map,
        foundry_xy,
        source_xy,
        blocked_tiles=blocked_tiles,
        block_friendly_infra=block_friendly_infra,
    )
    if not (isinstance(sink_xy, tuple) and len(sink_xy) == 2):
        return ()

    nodes = _plan_axionite_link_nodes(
        local_map,
        source_xy,
        sink_xy,
        blocked_tiles=blocked_tiles,
        block_friendly_infra=block_friendly_infra,
    )
    if not nodes:
        return ()

    sink_xy = (int(nodes[-1][0]), int(nodes[-1][1]))
    dx = sink_xy[0] - foundry_xy[0]
    dy = sink_xy[1] - foundry_xy[1]
    if (dx * dx) + (dy * dy) > 9:
        return ()
    return nodes


def _expand_diagonal_path_nodes(nodes, passable_fn):
    """Expand diagonal steps into two cardinal steps by inserting an intermediate node.
    Returns empty tuple if expansion is impossible (both intermediates are impassable)."""
    if len(nodes) < 2:
        return nodes
    result = [nodes[0]]
    for i in range(1, len(nodes)):
        prev = result[-1]
        curr = nodes[i]
        dx = int(curr[0]) - int(prev[0])
        dy = int(curr[1]) - int(prev[1])
        if abs(dx) == 1 and abs(dy) == 1:
            mid_h = (int(prev[0]) + dx, int(prev[1]))
            mid_v = (int(prev[0]), int(prev[1]) + dy)
            if passable_fn(mid_h[0], mid_h[1]):
                result.append(mid_h)
            elif passable_fn(mid_v[0], mid_v[1]):
                result.append(mid_v)
            else:
                return ()
        result.append(curr)
    return tuple(result)


def _plan_axionite_link_nodes(
    local_map,
    start_xy,
    sink_xy,
    blocked_tiles=(),
    block_friendly_infra: bool = False,
):
    blocked = {
        (int(p[0]), int(p[1]))
        for p in blocked_tiles
        if isinstance(p, tuple) and len(p) == 2
    }
    blocked.discard((int(start_xy[0]), int(start_xy[1])))
    blocked.discard((int(sink_xy[0]), int(sink_xy[1])))

    if start_xy == sink_xy:
        return (start_xy,)

    passable_fn = lambda x, y: _is_conveyor_planner_passable(
        local_map, x, y, sink_xy,
        exclude_positions=blocked,
        block_friendly_infra=block_friendly_infra,
    )
    steps = _astar_cardinal_plan(
        local_map,
        start_xy,
        sink_xy,
        max_expansions=1536,
        tile_passable_fn=passable_fn,
    )
    if not steps:
        return ()

    nodes = (start_xy, *steps)
    if nodes[-1] != sink_xy:
        return ()

    # Do NOT expand diagonals into cardinal pairs. Mirroring the main
    # titanium network pipeline (see _build_network_segment_transport), the
    # compose step below emits a bridge for each diagonal segment — which
    # saves a tile per diagonal and matches the behavior the user observed
    # in titanium network building.
    return nodes


def _compose_axionite_link_sequence(
    nodes,
    terminal_xy,
    build_order: str = "reverse",
    terminal_mode: str = "conveyor",
):
    if not nodes:
        return ()

    out = []
    if str(build_order).lower() == "forward":
        order = range(0, len(nodes))
    else:
        order = range(len(nodes) - 1, -1, -1)

    last_idx = len(nodes) - 1
    for idx in order:
        tx, ty = nodes[idx]
        if idx < len(nodes) - 1:
            nx, ny = nodes[idx + 1]
        else:
            nx, ny = terminal_xy

        if idx == last_idx and str(terminal_mode).lower() == "bridge":
            out.append((
                "bridge",
                int(tx),
                int(ty),
                (int(terminal_xy[0]), int(terminal_xy[1])),
            ))
            continue

        dx = int(nx) - int(tx)
        dy = int(ny) - int(ty)
        # Diagonal segment: place a bridge at the current node targeting the
        # next node. Bridges can accept input from any direction and deliver
        # to their target tile, which is how the main titanium network
        # pipeline handles diagonal path segments (see
        # _build_network_segment_transport).
        if abs(dx) == 1 and abs(dy) == 1:
            out.append((
                "bridge",
                int(tx),
                int(ty),
                (int(nx), int(ny)),
            ))
            continue

        out_dir = _CARDINAL_DIRECTION_BY_DELTA.get((dx, dy))
        if out_dir is None:
            return ()
        out.append(("conveyor", int(tx), int(ty), out_dir.name))
    return tuple(out)


# -----------------------------------------------------------------------------
# Splitter-splice fallback (Item C) — revised rules
# -----------------------------------------------------------------------------
# Used when every ti→foundry plan over existing titanium harvesters has
# failed: we splice into a nearby *titanium* conveyor by replacing it with
# a splitter and tapping a perpendicular conveyor chain toward the foundry.
#
# Revised rules (latest user spec):
#   1. Pick the conveyor closest (Chebyshev → Manhattan) to the foundry.
#   2. It must be a titanium conveyor — i.e. tracing upstream reaches a
#      titanium harvester (a conveyor in the ti delivery network).
#   3. No downstream check.
#   4. Splitter direction:
#        - Let d₁ be the original conveyor's direction at (cx,cy).
#        - Upstream tile u = (cx,cy) − d₁.
#        - If u is a harvester or bridge → splitter direction = d₁.
#        - If u is a conveyor with direction d₂ → splitter direction = d₂.
#        - Otherwise (splitter, armoured, or non-transport upstream) the
#          candidate is rejected.
#   5. The splitter still needs a perpendicular cardinal free/road neighbour
#      (side-tap) so the new ti-route chain has somewhere to start from.
#
# Replacing the conveyor with a splitter preserves its primary flow while
# opening a side-tap on the chosen perpendicular tile.


def _friendly_building_rec(local_map, x: int, y: int):
    rec = _known_building_at(local_map, x, y)
    if not isinstance(rec, dict):
        return None
    my_team = getattr(local_map, "my_team", None)
    if rec.get("team") != my_team:
        return None
    return rec


def _conveyor_traces_to_titanium_harvester(
    local_map, conveyor_xy, max_steps: int = 96,
) -> bool:
    """Walk upstream from a friendly conveyor one cardinal step at a time
    (following the reverse of each conveyor's direction). Return True if
    the chain terminates at a known titanium harvester; False otherwise.

    Bridges, splitters, armoured conveyors, and unknown upstream tiles
    terminate the trace with False — the splitter-splice replace step is
    only safe on a pure conveyor chain whose titanium provenance we can
    observe from local vision."""
    my_team = getattr(local_map, "my_team", None)
    ti_harvesters = getattr(local_map, "titanium_harvesters", None)
    if not isinstance(ti_harvesters, set):
        return False
    visited = set()
    cx, cy = int(conveyor_xy[0]), int(conveyor_xy[1])
    for _ in range(max_steps):
        if (cx, cy) in visited:
            return False
        visited.add((cx, cy))
        rec = _known_building_at(local_map, cx, cy)
        if not isinstance(rec, dict):
            return False
        if rec.get("team") != my_team:
            return False
        etype = rec.get("entity_type")
        if etype == EntityType.HARVESTER:
            return (cx, cy) in ti_harvesters
        if etype != EntityType.CONVEYOR:
            return False
        direction = rec.get("direction")
        if direction is None:
            return False
        ddx, ddy = direction.delta()
        cx, cy = cx - int(ddx), cy - int(ddy)
    return False


def _splitter_side_tap_candidates(local_map, x: int, y: int, dx: int, dy: int):
    # Perpendicular cardinal offsets relative to direction (dx,dy). Direction
    # is always axial (dx*dy == 0), so the axis (horizontal vs vertical)
    # determines which perpendicular pair we consider.
    if dx != 0 and dy == 0:
        perp = ((0, -1), (0, 1))
    else:
        perp = ((-1, 0), (1, 0))
    out = []
    for pdx, pdy in perp:
        tx = x + pdx
        ty = y + pdy
        if not local_map.in_bounds(tx, ty):
            continue
        # Must be MAP_FREE or MAP_ROAD (per Item C spec — cardinals only).
        tile = local_map.get(tx, ty)
        rec = _known_building_at(local_map, tx, ty)
        if isinstance(rec, dict):
            etype = rec.get("entity_type")
            if etype == EntityType.ROAD:
                out.append((tx, ty))
            # Anything else on the tile disqualifies it (we will not bulldoze
            # a non-road building to clear the tap path).
            continue
        if tile == MAP_FREE or tile == MAP_ROAD:
            out.append((tx, ty))
    return tuple(out)


def _find_consistent_conveyor_near_foundry(local_map, foundry_xy):
    """Scan friendly conveyors visible on the map and return the closest one
    (Chebyshev-to-foundry, tiebreak by Manhattan then position) that is part
    of a titanium delivery chain and has a valid splitter direction derived
    from its upstream neighbour. Returns a dict describing the splice site
    or ``None``. See the header comment for the revised rules."""
    fx = int(foundry_xy[0])
    fy = int(foundry_xy[1])
    my_team = getattr(local_map, "my_team", None)
    entities = getattr(local_map, "entities", None)
    if not isinstance(entities, dict):
        return None

    best = None
    best_key = None

    for rec in entities.values():
        if not isinstance(rec, dict):
            continue
        if not rec.get("alive", False):
            continue
        if rec.get("team") != my_team:
            continue
        if rec.get("entity_type") != EntityType.CONVEYOR:
            continue
        pos = rec.get("position")
        if not (isinstance(pos, tuple) and len(pos) == 2):
            continue
        cx = int(pos[0])
        cy = int(pos[1])
        direction = rec.get("direction")
        if direction is None:
            continue
        ddx, ddy = direction.delta()
        ddx = int(ddx)
        ddy = int(ddy)

        # Rule 2: must be part of a titanium delivery network.
        if not _conveyor_traces_to_titanium_harvester(local_map, (cx, cy)):
            continue

        # Rule 4: derive splitter direction from the upstream tile.
        up_xy = (cx - ddx, cy - ddy)
        if not local_map.in_bounds(up_xy[0], up_xy[1]):
            continue
        up_rec = _friendly_building_rec(local_map, up_xy[0], up_xy[1])
        if up_rec is None:
            continue
        up_etype = up_rec.get("entity_type")
        if up_etype in (EntityType.HARVESTER, EntityType.BRIDGE):
            splitter_dir_name = direction.name
        elif up_etype == EntityType.CONVEYOR:
            up_dir = up_rec.get("direction")
            if up_dir is None:
                continue
            splitter_dir_name = up_dir.name
        else:
            # Splitters, armoured conveyors, non-transport buildings, or any
            # other unexpected entity type are not valid upstream sources.
            continue

        # Rule 5: side-tap perpendicular to the splitter's primary direction.
        splitter_dir_obj = _CARDINAL_DIRECTION_BY_NAME.get(splitter_dir_name)
        if splitter_dir_obj is None:
            continue
        sdx, sdy = splitter_dir_obj.delta()
        sdx = int(sdx)
        sdy = int(sdy)
        tap_candidates = _splitter_side_tap_candidates(
            local_map, cx, cy, sdx, sdy)
        if not tap_candidates:
            continue

        # Pick the side-tap closest to the foundry (Chebyshev then Manhattan).
        tap_xy = min(
            tap_candidates,
            key=lambda t: (
                max(abs(t[0] - fx), abs(t[1] - fy)),
                abs(t[0] - fx) + abs(t[1] - fy),
                t[0],
                t[1],
            ),
        )

        key = (
            max(abs(cx - fx), abs(cy - fy)),
            abs(cx - fx) + abs(cy - fy),
            cx,
            cy,
        )
        if best_key is None or key < best_key:
            best_key = key
            best = {
                "splitter_xy": (cx, cy),
                "splitter_dir_name": splitter_dir_name,
                "side_tap_xy": tap_xy,
            }

    return best


def _pick_core_link_sinks(local_map, core_xy, toward_xy, blocked_tiles=()):
    cx = int(core_xy[0])
    cy = int(core_xy[1])
    if not (isinstance(toward_xy, tuple) and len(toward_xy) == 2):
        toward_xy = (cx, cy)

    blocked = {
        (int(p[0]), int(p[1]))
        for p in blocked_tiles
        if isinstance(p, tuple) and len(p) == 2
    }

    candidates = []
    for dx, dy in CARDINAL_DELTAS:
        sx = cx + dx
        sy = cy + dy
        if not local_map.in_bounds(sx, sy):
            continue
        if (sx, sy) in blocked:
            continue
        if not _is_conveyor_planner_passable(local_map, sx, sy, (sx, sy)):
            continue
        candidates.append((
            _manhattan((sx, sy), toward_xy),
            sx,
            sy,
        ))
    if not candidates:
        return ()
    candidates.sort()
    return tuple((sx, sy) for _, sx, sy in candidates)
def _collect_friendly_titanium_harvesters(local_map, state: EconomyState):
    my_team = getattr(local_map, "my_team", None)
    out = []
    for hxy in getattr(local_map, "titanium_harvesters", set()):
        if not (isinstance(hxy, tuple) and len(hxy) == 2):
            continue
        hx = int(hxy[0])
        hy = int(hxy[1])
        rec = _known_building_at(local_map, hx, hy)
        if not isinstance(rec, dict):
            continue
        if rec.get("entity_type") != EntityType.HARVESTER:
            continue

        if rec.get("team") == my_team:
            out.append((hx, hy))
            continue

        hid = _entity_id_from_rec(rec)
        trusted_stolen = (
            (hx, hy) in state.harvester_stolen_positions
            or (isinstance(hid, int) and hid in state.harvester_ids_stolen)
        )
        if trusted_stolen:
            out.append((hx, hy))

    out.sort(key=lambda p: (p[0], p[1]))
    return out


def _run_axionite_link_build_sequence(
    c: Controller,
    state: EconomyState,
    local_map,
    cur_xy,
    rnd: int,
    uid: int,
    ctx,
    phase_name: str,
) -> str:
    seq = tuple(ctx.get("link_sequence", ()))
    idx = int(ctx.get("link_sequence_index", 0))
    if not seq:
        return "failed"
    if idx >= len(seq):
        return "done"

    entry = seq[idx]
    step_mode = "conveyor"
    direction_name = None
    bridge_target_xy = None

    if isinstance(entry, tuple) and len(entry) == 3:
        tx, ty, direction_name = entry
        tile_xy = (int(tx), int(ty))
    elif isinstance(entry, tuple) and len(entry) == 4 and isinstance(entry[0], str):
        step_mode = str(entry[0]).lower()
        tile_xy = (int(entry[1]), int(entry[2]))
        if step_mode == "conveyor":
            direction_name = entry[3]
        elif step_mode == "bridge":
            bridge_target_xy = entry[3]
        else:
            return "failed"
    else:
        return "failed"

    # For bridge steps, the bot CANNOT stand on tile_xy when building —
    # bridges are non-walkable and the engine rejects "build non-walkable on
    # a tile the bot occupies" per the core rules. We ALSO need the bot to be
    # cardinally adjacent to tile_xy (dist²=1): the main titanium pipeline
    # never uses diagonal stands for bridge builds (the bot is always on
    # the previously-built conveyor, one cardinal step from the next node)
    # and diagonal stands appear to be rejected by the engine's build
    # eligibility check in practice. Conveyor steps can still build from
    # tile_xy itself since conveyors are walkable.
    bridge_stand_xy = None
    if step_mode == "bridge":
        dx_br = cur_xy[0] - int(tile_xy[0])
        dy_br = cur_xy[1] - int(tile_xy[1])
        cardinally_adjacent = (abs(dx_br) + abs(dy_br)) == 1
        if cardinally_adjacent:
            bridge_stand_xy = cur_xy
        else:
            bridge_stand_xy = _pick_cardinal_bridge_stand(
                local_map, tile_xy, cur_xy,
            )
            if bridge_stand_xy is None:
                # Last resort: accept any in-radius stand (including diagonals).
                bridge_stand_xy = _pick_build_stand_for_target(
                    local_map, tile_xy, cur_xy, disallow_xy=tile_xy,
                )
            if bridge_stand_xy is None:
                return "failed"

    move_target_xy = bridge_stand_xy if bridge_stand_xy is not None else tile_xy

    if cur_xy != move_target_xy:
        if state.plan_index >= len(state.plan_steps):
            steps = _astar_cardinal_plan(
                local_map,
                cur_xy,
                move_target_xy,
                max_expansions=640,
                tile_passable_fn=lambda x, y: _is_general_movement_passable(
                    local_map,
                    x,
                    y,
                    move_target_xy,
                ),
            )
            if not steps:
                steps = _astar_cardinal_plan(
                    local_map,
                    cur_xy,
                    move_target_xy,
                    max_expansions=640,
                    tile_passable_fn=lambda x, y: _is_general_movement_passable(
                        local_map,
                        x,
                        y,
                        move_target_xy,
                        respect_halo=False,
                    ),
                )
            if not steps:
                return "failed"
            state.plan_steps = steps
            state.plan_index = 0

        nxt = state.plan_steps[state.plan_index]
        if not _is_adjacent_step(cur_xy, nxt):
            state.plan_steps = ()
            state.plan_index = 0
            return "pending"

        move_result = _execute_step_toward(c, local_map, cur_xy, nxt, rnd, uid)
        if move_result == "moved":
            state.issued_move_last_tick = True
            state.expected_xy_after_move = nxt
            state.plan_index += 1
            return "pending"
        if move_result in ("built", "wait_cd"):
            return "pending"
        state.plan_steps = ()
        state.plan_index = 0
        return "pending"

    state.plan_steps = ()
    state.plan_index = 0
    if step_mode == "conveyor":
        matched, _seen_id, _seen_dir_name, checked = _visible_friendly_conveyor_match(
            local_map,
            tile_xy,
            direction_name=direction_name,
            require_direction_match=True,
        )
        if checked and matched:
            ctx["link_sequence_index"] = idx + 1
            return "pending"

    blocker_kind = _enemy_repair_blocker_kind_from_map(local_map, tile_xy)
    if blocker_kind is None:
        blocker_kind = _enemy_repair_blocker_kind_from_controller(c, tile_xy)

    if blocker_kind == "armoured":
        _log_axionite_build_failure_diag(
            c, local_map, tile_xy, cur_xy, phase_name,
            step_mode, direction_name, bridge_target_xy,
            rnd, uid, reason="enemy_armoured_on_tile",
            build_result=None, blocker_kind=blocker_kind,
        )
        return "failed"

    if blocker_kind is not None:
        _start_network_attack(
            state,
            tile_xy,
            phase_name,
            f"axionite_link_enemy_{blocker_kind}",
        )
        return "pending"

    if step_mode == "bridge":
        if not (
            isinstance(bridge_target_xy, tuple)
            and len(bridge_target_xy) == 2
        ):
            _log_axionite_build_failure_diag(
                c, local_map, tile_xy, cur_xy, phase_name,
                step_mode, direction_name, bridge_target_xy,
                rnd, uid, reason="bridge_target_missing",
                build_result=None, blocker_kind=None,
            )
            return "failed"
        bridge_target_xy = (int(bridge_target_xy[0]), int(bridge_target_xy[1]))
        build_result = _build_bridge_on_tile(
            c,
            tile_xy,
            bridge_target_xy,
            rnd,
            uid,
            state=state,
        )
        if build_result == "wait_cd":
            return "pending"
        if build_result in ("built", "already_built"):
            ctx["link_sequence_index"] = idx + 1
            return "pending"
        if _enter_resource_wait_or_fail(
            c, state, step_mode="bridge",
            resume_phase=phase_name, owner="axionite",
            rnd=rnd, uid=uid, cur_xy=cur_xy,
        ):
            return "pending"
        _log_axionite_build_failure_diag(
            c, local_map, tile_xy, cur_xy, phase_name,
            step_mode, direction_name, bridge_target_xy,
            rnd, uid, reason="bridge_build_not_built",
            build_result=build_result, blocker_kind=None,
        )
        return "failed"

    out_dir = _CARDINAL_DIRECTION_BY_NAME.get(direction_name)
    if out_dir is None:
        _log_axionite_build_failure_diag(
            c, local_map, tile_xy, cur_xy, phase_name,
            step_mode, direction_name, bridge_target_xy,
            rnd, uid, reason="invalid_direction_name",
            build_result=None, blocker_kind=None,
        )
        return "failed"

    build_result = _build_conveyor_on_tile(
        c,
        tile_xy,
        out_dir,
        rnd,
        uid,
        "economy_axionite_link_conveyor_built",
        state=state,
    )
    if build_result == "wait_cd":
        return "pending"
    if build_result == "built":
        ctx["link_sequence_index"] = idx + 1
        return "pending"
    if _enter_resource_wait_or_fail(
        c, state, step_mode="conveyor",
        resume_phase=phase_name, owner="axionite",
        rnd=rnd, uid=uid, cur_xy=cur_xy,
    ):
        return "pending"
    _log_axionite_build_failure_diag(
        c, local_map, tile_xy, cur_xy, phase_name,
        step_mode, direction_name, bridge_target_xy,
        rnd, uid, reason="conveyor_build_not_built",
        build_result=build_result, blocker_kind=None,
    )
    return "failed"


def _log_axionite_build_failure_diag(
    c: Controller, local_map, tile_xy, cur_xy, phase_name,
    step_mode, direction_name, bridge_target_xy,
    rnd, uid, reason, build_result, blocker_kind,
):
    """Emit a comprehensive one-shot log whenever _run_axionite_link_build_sequence
    returns "failed". Captures everything needed to diagnose why a conveyor/bridge
    build was rejected despite a clear tile and sufficient funds.
    Gated on COMPETITION_MODE."""
    if COMPETITION_MODE:
        return
    tx = int(tile_xy[0])
    ty = int(tile_xy[1])

    # Local-map view of the tile.
    env_code = -1
    try:
        env_code = int(local_map.get(tx, ty))
    except (TypeError, ValueError):
        pass
    rec = _known_building_at(local_map, tx, ty)
    rec_etype = "none"
    rec_team = "none"
    rec_dir = "none"
    if isinstance(rec, dict):
        etype_obj = rec.get("entity_type")
        rec_etype = str(etype_obj.name).lower() if etype_obj is not None else "none"
        my_team = getattr(local_map, "my_team", None)
        rec_team = "mine" if rec.get("team") == my_team else "enemy"
        dir_obj = rec.get("direction")
        if dir_obj is not None:
            rec_dir = dir_obj.name

    # Live controller view — the actual source of truth for can_build_*.
    ctrl_bid = -1
    ctrl_etype = "none"
    ctrl_team = "none"
    tp = Position(tx, ty)
    try:
        bid = c.get_tile_building_id(tp)
        if bid is not None:
            ctrl_bid = int(bid)
            try:
                ctrl_etype = str(c.get_entity_type(bid).name).lower()
            except GameError:
                pass
            try:
                ctrl_team = "mine" if c.get_team(bid) == c.get_team() else "enemy"
            except GameError:
                pass
    except GameError:
        pass

    ctrl_env = "?"
    try:
        ctrl_env = c.get_tile_env(tp).name
    except GameError:
        pass

    can_build = "?"
    if step_mode == "conveyor":
        d_obj = _CARDINAL_DIRECTION_BY_NAME.get(direction_name)
        if d_obj is not None:
            try:
                can_build = "1" if c.can_build_conveyor(tp, d_obj) else "0"
            except GameError:
                can_build = "error"
    elif step_mode == "bridge" and isinstance(bridge_target_xy, tuple):
        try:
            can_build = "1" if c.can_build_bridge(
                tp, Position(int(bridge_target_xy[0]), int(bridge_target_xy[1]))
            ) else "0"
        except GameError:
            can_build = "error"

    try:
        resources = c.get_global_resources()
        ti_res = int(resources[0])
        ax_res = int(resources[1])
    except GameError:
        ti_res = -1
        ax_res = -1
    try:
        acd = c.get_action_cooldown()
    except GameError:
        acd = -1
    try:
        mcd = c.get_move_cooldown()
    except GameError:
        mcd = -1

    # Scaled build cost — the live price the engine charges right now. If
    # this exceeds available resources, can_build_* returns False silently.
    cost_ti = -1
    cost_ax = -1
    try:
        if step_mode == "bridge":
            bc = c.get_bridge_cost()
            cost_ti = int(bc[0])
            cost_ax = int(bc[1])
        elif step_mode == "conveyor":
            bc = c.get_conveyor_cost()
            cost_ti = int(bc[0])
            cost_ax = int(bc[1])
    except GameError:
        pass

    try:
        scale_pct = float(c.get_scale_percent())
    except GameError:
        scale_pct = -1.0

    # Are resources actually short? (informational so a grep is immediate.)
    afford = "?"
    if cost_ti >= 0 and cost_ax >= 0 and ti_res >= 0 and ax_res >= 0:
        afford = "1" if (ti_res >= cost_ti and ax_res >= cost_ax) else "0"

    # Unit occupancy on the build tile (a unit there would also block a
    # non-walkable build even if the building slot is empty).
    unit_on_tile = "none"
    try:
        uid_on = local_map.get_known_unit(tx, ty)
        if uid_on is not None:
            unit_on_tile = str(int(uid_on))
    except (TypeError, AttributeError):
        pass

    # Bridge target tile info — the engine may reject a bridge whose target
    # is an obstacle / wall / ore (specific to bridges, not diagnosed above).
    target_env = "?"
    target_bid = -1
    target_etype = "none"
    if step_mode == "bridge" and isinstance(bridge_target_xy, tuple) and len(bridge_target_xy) == 2:
        ttp = Position(int(bridge_target_xy[0]), int(bridge_target_xy[1]))
        try:
            target_env = c.get_tile_env(ttp).name
        except GameError:
            pass
        try:
            tbid = c.get_tile_building_id(ttp)
            if tbid is not None:
                target_bid = int(tbid)
                try:
                    target_etype = str(c.get_entity_type(tbid).name).lower()
                except GameError:
                    pass
        except GameError:
            pass

    log_event(
        rnd, uid, "economy", f"({cur_xy[0]},{cur_xy[1]})",
        "economy_axionite_link_build_failed_diag",
        phase=str(phase_name),
        reason=str(reason),
        tx=tx, ty=ty,
        step=str(step_mode),
        dir=str(direction_name or "none"),
        btgt=(
            f"({int(bridge_target_xy[0])},{int(bridge_target_xy[1])})"
            if isinstance(bridge_target_xy, tuple) and len(bridge_target_xy) == 2
            else "none"
        ),
        build_result=str(build_result or "none"),
        blocker=str(blocker_kind or "none"),
        map_env=env_code,
        map_etype=rec_etype,
        map_team=rec_team,
        map_dir=rec_dir,
        ctrl_bid=ctrl_bid,
        ctrl_etype=ctrl_etype,
        ctrl_team=ctrl_team,
        ctrl_env=ctrl_env,
        can_build=can_build,
        ti=ti_res,
        ax=ax_res,
        cost_ti=cost_ti,
        cost_ax=cost_ax,
        scale_pct=scale_pct,
        afford=afford,
        unit=unit_on_tile,
        acd=acd,
        mcd=mcd,
        tgt_env=target_env,
        tgt_bid=target_bid,
        tgt_etype=target_etype,
        same_tile_as_bot=1 if (tx, ty) == tuple(cur_xy) else 0,
    )


def _run_axionite_refinement(
    c: Controller,
    state: EconomyState,
    local_map,
    cur_xy,
    rnd: int,
    uid: int,
):
    ctx = _ensure_axionite_ctx(state)
    last_phase = ctx.get("debug_last_phase")
    if last_phase != state.phase:
        log_event(
            rnd,
            uid,
            "economy",
            f"({cur_xy[0]},{cur_xy[1]})",
            "economy_axionite_phase_change",
            prev=(str(last_phase) if last_phase is not None else "none"),
            phase=str(state.phase),
        )
        ctx["debug_last_phase"] = state.phase

    if state.phase not in _HARVEST_AXIONITE_PHASES:
        state.phase = "axionite_enter"
        state.plan_steps = ()
        state.plan_index = 0
        state.defer_step_once = False
        ctx["active"] = True
        ctx["ore_xy"] = None
        ctx["ore_goal_xy"] = None
        ctx["foundry_goal_xy"] = None
        ctx["pair_mode"] = None
        ctx["ti_pair_key"] = None
        ctx["ti_feeder_tiles"] = ()
        ctx["link_nodes"] = ()
        ctx["link_sequence"] = ()
        ctx["link_sequence_index"] = 0
        ctx["link_terminal_xy"] = None
        ctx["core_route_candidate"] = None
        ctx["blocked_core_link_candidates"] = set()
        ctx["core_no_candidate_streak"] = 0
        ctx["debug_last_phase"] = "axionite_enter"
        log_event(
            rnd,
            uid,
            "economy",
            f"({cur_xy[0]},{cur_xy[1]})",
            "economy_axionite_pipeline_start",
            titanium=_team_titanium(c),
        )
        return

    if state.phase == "axionite_wait_resources":
        _run_resource_wait_phase(c, state, cur_xy, rnd, uid, "axionite")
        return

    known_ax_all = _known_unharvested_axionite(
        local_map,
        core_xy=state.core_xy,
        min_core_cheb=_ORE_CORE_EXCLUSION_MIN_CHEB,
    )
    known_ax_tiles = _known_axionite_tiles(local_map)
    blocked_ax = _axionite_blocked_ores(state)
    blocked_ax.intersection_update(set(known_ax_all))
    known_ax = [ore for ore in known_ax_all if ore not in blocked_ax]

    if state.phase == "axionite_enter":
        if not COMPETITION_MODE:
            log_event(rnd, uid, "economy", f"({cur_xy[0]},{cur_xy[1]})",
                      "economy_axionite_enter_start",
                      known_ax=len(known_ax), blocked_ax=len(blocked_ax))
        state.phase = "axionite_pick_ore"
        return

    if state.phase == "axionite_pick_ore":
        if not COMPETITION_MODE:
            log_event(rnd, uid, "economy", f"({cur_xy[0]},{cur_xy[1]})",
                      "economy_axionite_pick_ore_entry",
                      known_ax=len(known_ax), blocked_ax=len(blocked_ax))
        if not known_ax:
            # No remaining unharvested axionite target to start/continue.
            if not COMPETITION_MODE:
                log_event(rnd, uid, "economy", f"({cur_xy[0]},{cur_xy[1]})",
                          "economy_axionite_pick_ore_no_targets")
            ctx["active"] = False
            _resume_exploration_after_harvest(state)
            return

        ore_xy = _pick_nearest_ore(cur_xy, known_ax)
        if not COMPETITION_MODE:
            log_event(rnd, uid, "economy", f"({cur_xy[0]},{cur_xy[1]})",
                      "economy_axionite_pick_ore_selected",
                      ore_x=int(ore_xy[0]), ore_y=int(ore_xy[1]))
        ctx["ore_xy"] = ore_xy
        ctx["ore_goal_xy"] = None
        state.plan_steps = ()
        state.plan_index = 0
        state.defer_step_once = False
        state.phase = "axionite_pick_goal"
        return

    ore_xy = ctx.get("ore_xy")
    if not (isinstance(ore_xy, tuple) and len(ore_xy) == 2):
        if not COMPETITION_MODE:
            log_event(rnd, uid, "economy", f"({cur_xy[0]},{cur_xy[1]})",
                      "economy_axionite_pick_ore_invalid_ctx",
                      phase=state.phase)
        state.phase = "axionite_pick_ore"
        return

    ore_xy = (int(ore_xy[0]), int(ore_xy[1]))
    if ore_xy not in known_ax_tiles:
        if not COMPETITION_MODE:
            log_event(rnd, uid, "economy", f"({cur_xy[0]},{cur_xy[1]})",
                      "economy_axionite_pick_ore_tile_gone",
                      ore_x=ore_xy[0], ore_y=ore_xy[1])
        state.phase = "axionite_pick_ore"
        ctx["ore_xy"] = None
        ctx["ore_goal_xy"] = None
        return

    if state.phase == "axionite_pick_goal":
        if not COMPETITION_MODE:
            log_event(rnd, uid, "economy", f"({cur_xy[0]},{cur_xy[1]})",
                      "economy_axionite_pick_goal_entry",
                      ore_x=ore_xy[0], ore_y=ore_xy[1])
        goal_xy = _pick_axionite_adjacent_goal(local_map, ore_xy, cur_xy)
        if goal_xy is None:
            if not COMPETITION_MODE:
                log_event(rnd, uid, "economy", f"({cur_xy[0]},{cur_xy[1]})",
                          "economy_axionite_pick_goal_no_adjacent",
                          ore_x=ore_xy[0], ore_y=ore_xy[1])
            blocked_ax.add(ore_xy)
            ctx["ore_xy"] = None
            state.phase = "axionite_pick_ore"
            return
        if not COMPETITION_MODE:
            log_event(rnd, uid, "economy", f"({cur_xy[0]},{cur_xy[1]})",
                      "economy_axionite_pick_goal_selected",
                      ore_x=ore_xy[0], ore_y=ore_xy[1],
                      goal_x=int(goal_xy[0]), goal_y=int(goal_xy[1]))
        ctx["ore_goal_xy"] = goal_xy
        state.plan_steps = ()
        state.plan_index = 0
        state.defer_step_once = False
        state.phase = "axionite_plan_goal"
        return

    goal_xy = ctx.get("ore_goal_xy")
    if not (isinstance(goal_xy, tuple) and len(goal_xy) == 2):
        state.phase = "axionite_pick_goal"
        return

    goal_xy = (int(goal_xy[0]), int(goal_xy[1]))
    if state.phase == "axionite_plan_goal":
        if not COMPETITION_MODE:
            log_event(rnd, uid, "economy", f"({cur_xy[0]},{cur_xy[1]})",
                      "economy_axionite_plan_goal_entry",
                      ore_x=ore_xy[0], ore_y=ore_xy[1],
                      goal_x=goal_xy[0], goal_y=goal_xy[1])
        if cur_xy == goal_xy:
            if not COMPETITION_MODE:
                log_event(rnd, uid, "economy", f"({cur_xy[0]},{cur_xy[1]})",
                          "economy_axionite_plan_goal_already_at_goal",
                          ore_x=ore_xy[0], ore_y=ore_xy[1])
            state.phase = "axionite_build_harvester"
            return
        if not _is_axionite_goal_valid(local_map, ore_xy, goal_xy):
            if not COMPETITION_MODE:
                log_event(rnd, uid, "economy", f"({cur_xy[0]},{cur_xy[1]})",
                          "economy_axionite_plan_goal_invalid",
                          ore_x=ore_xy[0], ore_y=ore_xy[1],
                          goal_x=goal_xy[0], goal_y=goal_xy[1])
            ctx["ore_goal_xy"] = None
            state.phase = "axionite_pick_goal"
            return

        steps = _astar_cardinal_plan(
            local_map,
            cur_xy,
            goal_xy,
            max_expansions=768,
            tile_passable_fn=lambda x, y: _is_general_movement_passable(
                local_map,
                x,
                y,
                goal_xy,
            ),
        )
        if not steps:
            if not COMPETITION_MODE:
                log_event(rnd, uid, "economy", f"({cur_xy[0]},{cur_xy[1]})",
                          "economy_axionite_plan_goal_no_path",
                          ore_x=ore_xy[0], ore_y=ore_xy[1],
                          goal_x=goal_xy[0], goal_y=goal_xy[1])
            blocked_ax.add(ore_xy)
            ctx["ore_xy"] = None
            ctx["ore_goal_xy"] = None
            state.phase = "axionite_pick_ore"
            return

        if not COMPETITION_MODE:
            log_event(rnd, uid, "economy", f"({cur_xy[0]},{cur_xy[1]})",
                      "economy_axionite_plan_goal_path_found",
                      ore_x=ore_xy[0], ore_y=ore_xy[1],
                      goal_x=goal_xy[0], goal_y=goal_xy[1],
                      steps=len(steps))
        state.plan_steps = steps
        state.plan_index = 0
        state.defer_step_once = True
        state.phase = "axionite_follow_goal"
        return

    if state.phase == "axionite_follow_goal":
        if not COMPETITION_MODE:
            log_event(rnd, uid, "economy", f"({cur_xy[0]},{cur_xy[1]})",
                      "economy_axionite_follow_goal_entry",
                      ore_x=ore_xy[0], ore_y=ore_xy[1],
                      goal_x=goal_xy[0], goal_y=goal_xy[1],
                      plan_index=state.plan_index)
        if cur_xy == goal_xy:
            state.plan_steps = ()
            state.plan_index = 0
            state.defer_step_once = False
            state.phase = "axionite_build_harvester"
            return

        if state.defer_step_once:
            state.defer_step_once = False
            return

        if state.plan_index >= len(state.plan_steps):
            if not COMPETITION_MODE:
                log_event(rnd, uid, "economy", f"({cur_xy[0]},{cur_xy[1]})",
                          "economy_axionite_follow_goal_replan",
                          ore_x=ore_xy[0], ore_y=ore_xy[1],
                          goal_x=goal_xy[0], goal_y=goal_xy[1])
            state.phase = "axionite_plan_goal"
            return

        nxt = state.plan_steps[state.plan_index]
        if not _is_adjacent_step(cur_xy, nxt):
            if not COMPETITION_MODE:
                log_event(rnd, uid, "economy", f"({cur_xy[0]},{cur_xy[1]})",
                          "economy_axionite_follow_goal_step_invalid",
                          ore_x=ore_xy[0], ore_y=ore_xy[1])
            state.phase = "axionite_plan_goal"
            return

        move_result = _execute_step_toward(c, local_map, cur_xy, nxt, rnd, uid)
        if move_result == "moved":
            state.issued_move_last_tick = True
            state.expected_xy_after_move = nxt
            state.plan_index += 1
            return
        if move_result in ("built", "wait_cd"):
            return
        if not COMPETITION_MODE:
            log_event(rnd, uid, "economy", f"({cur_xy[0]},{cur_xy[1]})",
                      "economy_axionite_follow_goal_move_failed",
                      ore_x=ore_xy[0], ore_y=ore_xy[1],
                      move_result=str(move_result))
        state.phase = "axionite_plan_goal"
        return

    if state.phase == "axionite_build_harvester":
        if not COMPETITION_MODE:
            log_event(rnd, uid, "economy", f"({cur_xy[0]},{cur_xy[1]})",
                      "economy_axionite_build_harvester_entry",
                      ore_x=ore_xy[0], ore_y=ore_xy[1],
                      goal_x=goal_xy[0], goal_y=goal_xy[1])
        if cur_xy != goal_xy:
            if not COMPETITION_MODE:
                log_event(rnd, uid, "economy", f"({cur_xy[0]},{cur_xy[1]})",
                          "economy_axionite_build_harvester_not_at_goal",
                          ore_x=ore_xy[0], ore_y=ore_xy[1])
            state.phase = "axionite_plan_goal"
            return
        if c.get_action_cooldown() > 0:
            return

        ore_pos = Position(ore_xy[0], ore_xy[1])
        try:
            can_build = c.can_build_harvester(ore_pos)
        except GameError:
            can_build = False

        if can_build:
            try:
                new_id = c.build_harvester(ore_pos)
                if isinstance(new_id, int):
                    state.built_entity_ids.add(new_id)
                state.built_harvester_positions.add(ore_xy)
                log_event(
                    rnd,
                    uid,
                    "economy",
                    f"({cur_xy[0]},{cur_xy[1]})",
                    "economy_axionite_harvester_built",
                    ox=ore_xy[0],
                    oy=ore_xy[1],
                )
                state.phase = "axionite_pick_foundry"
                return
            except GameError:
                # Contention: another bot likely built on this tile this tick.
                # Block the ore and move on — axionite ores are rare but the
                # cost of false-blocking an ore is low vs an infinite retry.
                rec_ax = _known_building_at(local_map, ore_xy[0], ore_xy[1])
                if isinstance(rec_ax, dict) and rec_ax.get("entity_type") == EntityType.HARVESTER:
                    blocked_ax.add((int(ore_xy[0]), int(ore_xy[1])))
                log_event(
                    rnd,
                    uid,
                    "economy",
                    f"({cur_xy[0]},{cur_xy[1]})",
                    "economy_axionite_harvester_build_contention",
                    ox=ore_xy[0],
                    oy=ore_xy[1],
                )
                _reset_axionite_pipeline_for_next_ore(state, ctx)
                return
            return

        # can_build returned False for a non-cooldown reason. If we have
        # resources, assume contention and abandon this ore.
        if _team_titanium(c) < int(HARVESTER_BASE_COST[0]):
            return

        blocked_ax.add((int(ore_xy[0]), int(ore_xy[1])))
        log_event(
            rnd,
            uid,
            "economy",
            f"({cur_xy[0]},{cur_xy[1]})",
            "economy_axionite_harvester_build_rejected",
            ox=ore_xy[0],
            oy=ore_xy[1],
        )
        _reset_axionite_pipeline_for_next_ore(state, ctx)
        return

    if state.phase == "axionite_pick_foundry":
        if not COMPETITION_MODE:
            log_event(rnd, uid, "economy", f"({cur_xy[0]},{cur_xy[1]})",
                      "economy_axionite_pick_foundry_entry",
                      ore_x=ore_xy[0], ore_y=ore_xy[1])
        foundry_xy, need_clear, need_self_move = _pick_foundry_tile_for_ore(
            local_map, ore_xy, cur_xy)
        if isinstance(foundry_xy, tuple) and len(foundry_xy) == 2 and need_self_move:
            # The only (or best) valid foundry tile is the one the bot is
            # standing on. Record it and step off before building.
            ctx["foundry_xy"] = (int(foundry_xy[0]), int(foundry_xy[1]))
            if not COMPETITION_MODE:
                log_event(rnd, uid, "economy", f"({cur_xy[0]},{cur_xy[1]})",
                          "economy_axionite_pick_foundry_self_stand",
                          fx=int(foundry_xy[0]), fy=int(foundry_xy[1]))
            state.plan_steps = ()
            state.plan_index = 0
            state.defer_step_once = False
            state.phase = "axionite_step_off_foundry_tile"
            return
        if not (isinstance(foundry_xy, tuple) and len(foundry_xy) == 2):
            blocked_ax.add((int(ore_xy[0]), int(ore_xy[1])))
            log_event(
                rnd,
                uid,
                "economy",
                f"({cur_xy[0]},{cur_xy[1]})",
                "economy_axionite_ore_blocked_no_foundry_tile",
                ox=int(ore_xy[0]),
                oy=int(ore_xy[1]),
            )
            ctx["ore_xy"] = None
            ctx["ore_goal_xy"] = None
            ctx["foundry_xy"] = None
            ctx["foundry_goal_xy"] = None
            ctx["root_xy"] = None
            ctx["root_bridge_xy"] = None
            ctx["ti_pair_xy"] = None
            ctx["ti_pair_key"] = None
            ctx["pair_mode"] = None
            ctx["fallback_ti_ore_xy"] = None
            ctx["fallback_ti_goal_xy"] = None
            ctx["ti_feeder_tiles"] = ()
            ctx["link_nodes"] = ()
            ctx["link_sequence"] = ()
            ctx["link_sequence_index"] = 0
            ctx["link_terminal_xy"] = None
            ctx["core_route_candidate"] = None
            ctx["blocked_core_link_candidates"] = set()
            state.plan_steps = ()
            state.plan_index = 0
            state.defer_step_once = False
            state.phase = "axionite_pick_ore"
            return

        if need_clear:
            if not COMPETITION_MODE:
                log_event(rnd, uid, "economy", f"({cur_xy[0]},{cur_xy[1]})",
                          "economy_axionite_pick_foundry_need_clear",
                          fx=int(foundry_xy[0]), fy=int(foundry_xy[1]))
            _start_network_attack(
                state,
                foundry_xy,
                "axionite_pick_foundry",
                "axionite_clear_foundry_tile",
            )
            return

        ctx["foundry_xy"] = (int(foundry_xy[0]), int(foundry_xy[1]))
        log_event(
            rnd,
            uid,
            "economy",
            f"({cur_xy[0]},{cur_xy[1]})",
            "economy_axionite_foundry_target_selected",
            fx=ctx["foundry_xy"][0],
            fy=ctx["foundry_xy"][1],
        )
        state.phase = "axionite_pick_foundry_goal"
        return

    if state.phase == "axionite_step_off_foundry_tile":
        target_xy = ctx.get("foundry_xy")
        if not (isinstance(target_xy, tuple) and len(target_xy) == 2):
            # Lost the intended foundry tile — re-enter selection.
            state.phase = "axionite_pick_foundry"
            return
        target_xy = (int(target_xy[0]), int(target_xy[1]))

        if cur_xy != target_xy:
            # Already off the foundry tile — re-verify with the picker. If
            # the tile is still valid (nothing else moved onto it) the picker
            # will pick it again; otherwise it will pick a better candidate.
            state.phase = "axionite_pick_foundry"
            return

        # Bot still standing on the target. Step to an adjacent walkable tile
        # that is NOT the ore. Prefer diagonal steps to maintain ore adjacency
        # for the foundry's input side, but any walkable neighbour works.
        ore_xy_local = ctx.get("ore_xy")
        if isinstance(ore_xy_local, tuple) and len(ore_xy_local) == 2:
            ore_xy_local = (int(ore_xy_local[0]), int(ore_xy_local[1]))
        else:
            ore_xy_local = None

        best_step = None
        best_score = None
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                if dx == 0 and dy == 0:
                    continue
                nx = cur_xy[0] + dx
                ny = cur_xy[1] + dy
                if not local_map.in_bounds(nx, ny):
                    continue
                if ore_xy_local is not None and (nx, ny) == ore_xy_local:
                    continue
                if not _is_builder_directly_walkable_tile(
                    local_map, nx, ny, respect_halo=True,
                ):
                    continue
                # Prefer steps that keep us adjacent (r²≤2) to the foundry
                # target so the follow-up build stand check can succeed
                # cheaply, and closer to the ore so we remain useful.
                ddx_f = nx - target_xy[0]
                ddy_f = ny - target_xy[1]
                dist_sq_f = ddx_f * ddx_f + ddy_f * ddy_f
                score = (
                    0 if dist_sq_f <= ACTION_RADIUS_SQ else 1,
                    dist_sq_f,
                    nx,
                    ny,
                )
                if best_score is None or score < best_score:
                    best_score = score
                    best_step = (nx, ny)

        if best_step is None:
            # No walkable neighbour — fall back to halo-off movement.
            for dx in (-1, 0, 1):
                for dy in (-1, 0, 1):
                    if dx == 0 and dy == 0:
                        continue
                    nx = cur_xy[0] + dx
                    ny = cur_xy[1] + dy
                    if not local_map.in_bounds(nx, ny):
                        continue
                    if ore_xy_local is not None and (nx, ny) == ore_xy_local:
                        continue
                    if _is_builder_directly_walkable_tile(
                        local_map, nx, ny, respect_halo=False,
                    ):
                        best_step = (nx, ny)
                        break
                if best_step is not None:
                    break

        if best_step is None:
            # Genuinely stuck — abandon this foundry pick and try a different
            # tile next round.
            log_event(
                rnd, uid, "economy", f"({cur_xy[0]},{cur_xy[1]})",
                "economy_axionite_step_off_foundry_no_step",
                fx=target_xy[0], fy=target_xy[1],
            )
            ctx["foundry_xy"] = None
            state.phase = "axionite_pick_foundry"
            return

        move_result = _execute_step_toward(
            c, local_map, cur_xy, best_step, rnd, uid)
        if move_result == "moved":
            state.issued_move_last_tick = True
            state.expected_xy_after_move = best_step
            if not COMPETITION_MODE:
                log_event(
                    rnd, uid, "economy", f"({cur_xy[0]},{cur_xy[1]})",
                    "economy_axionite_step_off_foundry_moved",
                    fx=target_xy[0], fy=target_xy[1],
                    nx=best_step[0], ny=best_step[1],
                )
            # Re-verify next round — leaves phase as axionite_step_off_foundry_tile
            # so if the step somehow didn't land us off the target we try again.
            return
        if move_result in ("built", "wait_cd"):
            return
        # Move failed — retry next round.
        return

    foundry_xy = ctx.get("foundry_xy")
    if not (isinstance(foundry_xy, tuple) and len(foundry_xy) == 2):
        if state.phase in (
            "axionite_pick_foundry_goal",
            "axionite_plan_foundry_goal",
            "axionite_follow_foundry_goal",
            "axionite_build_foundry",
        ):
            state.phase = "axionite_pick_foundry"
            return

    if state.phase == "axionite_pick_foundry_goal":
        if not COMPETITION_MODE:
            _fxy = ctx.get("foundry_xy")
            log_event(rnd, uid, "economy", f"({cur_xy[0]},{cur_xy[1]})",
                      "economy_axionite_pick_foundry_goal_entry",
                      fx=int(_fxy[0]) if isinstance(_fxy, tuple) else -1,
                      fy=int(_fxy[1]) if isinstance(_fxy, tuple) else -1)
        stand_xy = _pick_build_stand_for_target(local_map, foundry_xy, cur_xy)
        if (
            isinstance(stand_xy, tuple)
            and len(stand_xy) == 2
            and (int(stand_xy[0]), int(stand_xy[1])) == cur_xy
        ):
            alt_xy = _pick_build_stand_for_target(
                local_map,
                foundry_xy,
                cur_xy,
                disallow_xy=cur_xy,
            )
            if isinstance(alt_xy, tuple) and len(alt_xy) == 2:
                stand_xy = alt_xy
        if not (isinstance(stand_xy, tuple) and len(stand_xy) == 2):
            if not COMPETITION_MODE:
                log_event(rnd, uid, "economy", f"({cur_xy[0]},{cur_xy[1]})",
                          "economy_axionite_pick_foundry_goal_no_stand",
                          fx=int(foundry_xy[0]) if isinstance(foundry_xy, tuple) else -1,
                          fy=int(foundry_xy[1]) if isinstance(foundry_xy, tuple) else -1)
            state.phase = "axionite_pick_foundry"
            return
        ctx["foundry_goal_xy"] = stand_xy
        log_event(
            rnd,
            uid,
            "economy",
            f"({cur_xy[0]},{cur_xy[1]})",
            "economy_axionite_foundry_goal_selected",
            gx=stand_xy[0],
            gy=stand_xy[1],
            fx=foundry_xy[0],
            fy=foundry_xy[1],
        )
        state.plan_steps = ()
        state.plan_index = 0
        state.defer_step_once = False
        state.phase = "axionite_plan_foundry_goal"
        return

    foundry_goal_xy = ctx.get("foundry_goal_xy")
    if state.phase == "axionite_plan_foundry_goal":
        if not (isinstance(foundry_goal_xy, tuple) and len(foundry_goal_xy) == 2):
            if not COMPETITION_MODE:
                log_event(rnd, uid, "economy", f"({cur_xy[0]},{cur_xy[1]})",
                          "economy_axionite_plan_foundry_goal_no_goal")
            state.phase = "axionite_pick_foundry_goal"
            return
        foundry_goal_xy = (int(foundry_goal_xy[0]), int(foundry_goal_xy[1]))
        if not COMPETITION_MODE:
            log_event(rnd, uid, "economy", f"({cur_xy[0]},{cur_xy[1]})",
                      "economy_axionite_plan_foundry_goal_entry",
                      gx=foundry_goal_xy[0], gy=foundry_goal_xy[1])
        if cur_xy == foundry_goal_xy:
            if not COMPETITION_MODE:
                log_event(rnd, uid, "economy", f"({cur_xy[0]},{cur_xy[1]})",
                          "economy_axionite_plan_foundry_goal_already_at_goal")
            state.phase = "axionite_build_foundry"
            return

        steps = _astar_cardinal_plan(
            local_map,
            cur_xy,
            foundry_goal_xy,
            max_expansions=640,
            tile_passable_fn=lambda x, y: _is_general_movement_passable(
                local_map,
                x,
                y,
                foundry_goal_xy,
            ),
        )
        if not steps:
            if not COMPETITION_MODE:
                log_event(rnd, uid, "economy", f"({cur_xy[0]},{cur_xy[1]})",
                          "economy_axionite_plan_foundry_goal_no_path",
                          gx=foundry_goal_xy[0], gy=foundry_goal_xy[1])
            state.phase = "axionite_pick_foundry_goal"
            return
        if not COMPETITION_MODE:
            log_event(rnd, uid, "economy", f"({cur_xy[0]},{cur_xy[1]})",
                      "economy_axionite_plan_foundry_goal_path_found",
                      gx=foundry_goal_xy[0], gy=foundry_goal_xy[1],
                      steps=len(steps))
        state.plan_steps = steps
        state.plan_index = 0
        state.defer_step_once = True
        state.phase = "axionite_follow_foundry_goal"
        return

    if state.phase == "axionite_follow_foundry_goal":
        if not (isinstance(foundry_goal_xy, tuple) and len(foundry_goal_xy) == 2):
            if not COMPETITION_MODE:
                log_event(rnd, uid, "economy", f"({cur_xy[0]},{cur_xy[1]})",
                          "economy_axionite_follow_foundry_goal_no_goal")
            state.phase = "axionite_pick_foundry_goal"
            return
        foundry_goal_xy = (int(foundry_goal_xy[0]), int(foundry_goal_xy[1]))
        if not COMPETITION_MODE:
            log_event(rnd, uid, "economy", f"({cur_xy[0]},{cur_xy[1]})",
                      "economy_axionite_follow_foundry_goal_entry",
                      gx=foundry_goal_xy[0], gy=foundry_goal_xy[1],
                      plan_index=state.plan_index)
        if cur_xy == foundry_goal_xy:
            state.plan_steps = ()
            state.plan_index = 0
            state.defer_step_once = False
            state.phase = "axionite_build_foundry"
            return

        if state.defer_step_once:
            state.defer_step_once = False
            return
        if state.plan_index >= len(state.plan_steps):
            if not COMPETITION_MODE:
                log_event(rnd, uid, "economy", f"({cur_xy[0]},{cur_xy[1]})",
                          "economy_axionite_follow_foundry_goal_replan",
                          gx=foundry_goal_xy[0], gy=foundry_goal_xy[1])
            state.phase = "axionite_plan_foundry_goal"
            return

        nxt = state.plan_steps[state.plan_index]
        if not _is_adjacent_step(cur_xy, nxt):
            if not COMPETITION_MODE:
                log_event(rnd, uid, "economy", f"({cur_xy[0]},{cur_xy[1]})",
                          "economy_axionite_follow_foundry_goal_step_invalid")
            state.phase = "axionite_plan_foundry_goal"
            return
        move_result = _execute_step_toward(c, local_map, cur_xy, nxt, rnd, uid)
        if move_result == "moved":
            state.issued_move_last_tick = True
            state.expected_xy_after_move = nxt
            state.plan_index += 1
            return
        if move_result in ("built", "wait_cd"):
            return
        if not COMPETITION_MODE:
            log_event(rnd, uid, "economy", f"({cur_xy[0]},{cur_xy[1]})",
                      "economy_axionite_follow_foundry_goal_move_failed",
                      move_result=str(move_result))
        state.phase = "axionite_plan_foundry_goal"
        return

    if state.phase == "axionite_build_foundry":
        if not COMPETITION_MODE:
            log_event(rnd, uid, "economy", f"({cur_xy[0]},{cur_xy[1]})",
                      "economy_axionite_build_foundry_entry",
                      fx=int(foundry_xy[0]) if isinstance(foundry_xy, tuple) else -1,
                      fy=int(foundry_xy[1]) if isinstance(foundry_xy, tuple) else -1)
        if not (isinstance(foundry_goal_xy, tuple) and len(foundry_goal_xy) == 2):
            state.phase = "axionite_pick_foundry_goal"
            return
        if not (isinstance(foundry_xy, tuple) and len(foundry_xy) == 2):
            state.phase = "axionite_pick_foundry"
            return
        foundry_goal_xy = (int(foundry_goal_xy[0]), int(foundry_goal_xy[1]))
        foundry_xy = (int(foundry_xy[0]), int(foundry_xy[1]))
        if cur_xy != foundry_goal_xy:
            if not COMPETITION_MODE:
                log_event(rnd, uid, "economy", f"({cur_xy[0]},{cur_xy[1]})",
                          "economy_axionite_build_foundry_not_at_goal",
                          fx=foundry_xy[0], fy=foundry_xy[1],
                          gx=foundry_goal_xy[0], gy=foundry_goal_xy[1])
            state.phase = "axionite_plan_foundry_goal"
            return
        if c.get_action_cooldown() > 0:
            return

        fp = Position(foundry_xy[0], foundry_xy[1])
        try:
            if c.can_build_foundry(fp):
                new_id = c.build_foundry(fp)
                if isinstance(new_id, int):
                    state.built_entity_ids.add(new_id)
                log_event(
                    rnd,
                    uid,
                    "economy",
                    f"({cur_xy[0]},{cur_xy[1]})",
                    "economy_axionite_foundry_built",
                    fx=foundry_xy[0],
                    fy=foundry_xy[1],
                )
                # Root bridge placement is deferred until AFTER the titanium
                # feeder is connected to this foundry — see the ordering
                # comment in axionite_pick_ti_route.
                state.phase = "axionite_pick_titanium_pair"
                return
        except GameError:
            return

        # Reuse an already existing friendly foundry at this tile.
        try:
            existing = c.get_tile_building_id(fp)
        except GameError:
            existing = None

        if existing is not None:
            try:
                same_team = c.get_team(existing) == c.get_team()
                etype = c.get_entity_type(existing)
                if same_team and etype == EntityType.FOUNDRY:
                    log_event(
                        rnd,
                        uid,
                        "economy",
                        f"({cur_xy[0]},{cur_xy[1]})",
                        "economy_axionite_foundry_reused",
                        fx=foundry_xy[0],
                        fy=foundry_xy[1],
                    )
                    # Reordering: connect titanium feeder first, then root.
                    state.phase = "axionite_pick_titanium_pair"
                    return
            except GameError:
                pass

        # If a friendly replaceable transport/road blocks the target, destroy
        # then retry foundry build in-place.
        if existing is not None:
            try:
                same_team = c.get_team(existing) == c.get_team()
                etype = c.get_entity_type(existing)
                if (
                    same_team
                    and etype in (
                        EntityType.ROAD,
                        EntityType.CONVEYOR,
                        EntityType.ARMOURED_CONVEYOR,
                        EntityType.SPLITTER,
                        EntityType.BRIDGE,
                    )
                    and c.can_destroy(fp)
                ):
                    c.destroy(fp)
                    if c.can_build_foundry(fp):
                        new_id = c.build_foundry(fp)
                        if isinstance(new_id, int):
                            state.built_entity_ids.add(new_id)
                        log_event(
                            rnd,
                            uid,
                            "economy",
                            f"({cur_xy[0]},{cur_xy[1]})",
                            "economy_axionite_foundry_built",
                            fx=foundry_xy[0],
                            fy=foundry_xy[1],
                        )
                        # Reordering: connect titanium feeder first, then root.
                        state.phase = "axionite_pick_titanium_pair"
                        return
            except GameError:
                pass

        # Friendly occupied tiles are not valid foundry targets; reselect.
        rec = _known_building_at(local_map, foundry_xy[0], foundry_xy[1])
        my_team = getattr(local_map, "my_team", None)
        if (
            isinstance(rec, dict)
            and rec.get("team") == my_team
            and rec.get("entity_type") != EntityType.FOUNDRY
        ):
            ctx["foundry_xy"] = None
            ctx["foundry_goal_xy"] = None
            state.phase = "axionite_pick_foundry"
            return

        blocker_kind = _enemy_repair_blocker_kind_from_map(
            local_map, foundry_xy)
        if blocker_kind is None:
            blocker_kind = _enemy_repair_blocker_kind_from_controller(
                c, foundry_xy)
        if blocker_kind is not None and blocker_kind != "armoured":
            _start_network_attack(
                state,
                foundry_xy,
                "axionite_build_foundry",
                "axionite_foundry_tile_blocked",
            )
            return

        # If build still fails while standing on the foundry goal tile,
        # reselect a different adjacent stand tile and move there first.
        if cur_xy == foundry_goal_xy:
            alt_xy = _pick_build_stand_for_target(
                local_map,
                foundry_xy,
                cur_xy,
                disallow_xy=cur_xy,
            )
            if isinstance(alt_xy, tuple) and len(alt_xy) == 2:
                ctx["foundry_goal_xy"] = (int(alt_xy[0]), int(alt_xy[1]))
                state.plan_steps = ()
                state.plan_index = 0
                state.defer_step_once = False
                state.phase = "axionite_plan_foundry_goal"
                log_event(
                    rnd,
                    uid,
                    "economy",
                    f"({cur_xy[0]},{cur_xy[1]})",
                    "economy_axionite_foundry_goal_reselected",
                    gx=ctx["foundry_goal_xy"][0],
                    gy=ctx["foundry_goal_xy"][1],
                    fx=foundry_xy[0],
                    fy=foundry_xy[1],
                    reason="build_from_current_failed",
                )
                return
            state.phase = "axionite_pick_foundry_goal"
        return

    if state.phase == "axionite_pick_root":
        if not COMPETITION_MODE:
            log_event(rnd, uid, "economy", f"({cur_xy[0]},{cur_xy[1]})",
                      "economy_axionite_pick_root_entry",
                      fx=int(foundry_xy[0]) if isinstance(foundry_xy, tuple) else -1,
                      fy=int(foundry_xy[1]) if isinstance(foundry_xy, tuple) else -1)
        ti_harvesters = _collect_friendly_titanium_harvesters(local_map, state)
        debug_report = {}
        bridge_xy, root_xy, need_clear = _pick_root_bridge_and_root(
            local_map, foundry_xy, state.core_xy, cur_xy,
            ti_harvester_positions=ti_harvesters,
            debug_report=debug_report,
        )
        if bridge_xy is None or root_xy is None:
            _log_root_pick_debug_report(
                rnd, uid, cur_xy, foundry_xy, debug_report,
                "economy_axionite_pick_root_no_candidates",
            )
            # Defer instead of abandoning immediately — if the map is merely
            # under-explored on the foundry's far side, future rounds will
            # fill in vision and the pick may succeed. We only abandon after
            # a conservative retry budget.
            retries = int(ctx.get("root_pick_retry_count", 0)) + 1
            ctx["root_pick_retry_count"] = retries
            if retries < 32:
                # Stay in this phase; tick will naturally retry next round.
                return

            log_event(
                rnd, uid, "economy", f"({cur_xy[0]},{cur_xy[1]})",
                "economy_axionite_ore_blocked_no_root_bridge",
                ore=(f"({ore_xy[0]},{ore_xy[1]})" if isinstance(ore_xy, tuple) else "none"),
                foundry=(f"({foundry_xy[0]},{foundry_xy[1]})" if isinstance(foundry_xy, tuple) else "none"),
                root="none",
                retries=retries,
            )
            _mark_axionite_ore_inaccessible(state, ctx, ore_xy)
            return
        # Successful pick — reset the retry counter for the next ore.
        ctx["root_pick_retry_count"] = 0
        ctx["root_xy"] = root_xy
        ctx["root_bridge_xy"] = bridge_xy
        if not COMPETITION_MODE:
            log_event(rnd, uid, "economy", f"({cur_xy[0]},{cur_xy[1]})",
                      "economy_axionite_pick_root_selected",
                      rx=int(root_xy[0]), ry=int(root_xy[1]),
                      bx=int(bridge_xy[0]), by=int(bridge_xy[1]),
                      need_clear=need_clear)
        if need_clear:
            if not COMPETITION_MODE:
                log_event(rnd, uid, "economy", f"({cur_xy[0]},{cur_xy[1]})",
                          "economy_axionite_pick_root_need_clear",
                          bx=int(bridge_xy[0]), by=int(bridge_xy[1]))
            state.phase = "axionite_pick_root_bridge"
            return
        state.plan_steps = ()
        state.plan_index = 0
        state.defer_step_once = False
        state.phase = "axionite_plan_root_bridge"
        return

    root_xy = ctx.get("root_xy")
    if state.phase == "axionite_pick_root_bridge":
        if not COMPETITION_MODE:
            log_event(rnd, uid, "economy", f"({cur_xy[0]},{cur_xy[1]})",
                      "economy_axionite_pick_root_bridge_entry")
        # Re-run the combined picker (arrived here after clearing an enemy tile).
        ti_harvesters = _collect_friendly_titanium_harvesters(local_map, state)
        debug_report = {}
        bridge_xy, new_root_xy, need_clear = _pick_root_bridge_and_root(
            local_map, foundry_xy, state.core_xy, cur_xy,
            ti_harvester_positions=ti_harvesters,
            debug_report=debug_report,
        )
        if bridge_xy is None or new_root_xy is None:
            _log_root_pick_debug_report(
                rnd, uid, cur_xy, foundry_xy, debug_report,
                "economy_axionite_pick_root_bridge_no_candidates",
            )
            retries = int(ctx.get("root_pick_retry_count", 0)) + 1
            ctx["root_pick_retry_count"] = retries
            if retries < 32:
                return

            log_event(
                rnd, uid, "economy", f"({cur_xy[0]},{cur_xy[1]})",
                "economy_axionite_ore_blocked_no_root_bridge",
                ore=(f"({ore_xy[0]},{ore_xy[1]})" if isinstance(ore_xy, tuple) else "none"),
                foundry=(f"({foundry_xy[0]},{foundry_xy[1]})" if isinstance(foundry_xy, tuple) else "none"),
                root="none",
                retries=retries,
            )
            _mark_axionite_ore_inaccessible(state, ctx, ore_xy)
            return
        ctx["root_pick_retry_count"] = 0
        ctx["root_xy"] = new_root_xy
        ctx["root_bridge_xy"] = bridge_xy
        if not COMPETITION_MODE:
            log_event(rnd, uid, "economy", f"({cur_xy[0]},{cur_xy[1]})",
                      "economy_axionite_pick_root_bridge_selected",
                      rx=int(new_root_xy[0]), ry=int(new_root_xy[1]),
                      bx=int(bridge_xy[0]), by=int(bridge_xy[1]),
                      need_clear=need_clear)
        if need_clear:
            if not COMPETITION_MODE:
                log_event(rnd, uid, "economy", f"({cur_xy[0]},{cur_xy[1]})",
                          "economy_axionite_pick_root_bridge_need_clear",
                          bx=int(bridge_xy[0]), by=int(bridge_xy[1]))
            _start_network_attack(
                state, bridge_xy, "axionite_pick_root_bridge",
                "axionite_clear_root_bridge_tile",
            )
            return
        state.plan_steps = ()
        state.plan_index = 0
        state.defer_step_once = False
        state.phase = "axionite_plan_root_bridge"
        return

    root_bridge_xy = ctx.get("root_bridge_xy")
    if state.phase == "axionite_plan_root_bridge":
        if not (isinstance(root_bridge_xy, tuple) and len(root_bridge_xy) == 2):
            if not COMPETITION_MODE:
                log_event(rnd, uid, "economy", f"({cur_xy[0]},{cur_xy[1]})",
                          "economy_axionite_plan_root_bridge_no_bridge_xy")
            state.phase = "axionite_pick_root_bridge"
            return
        root_bridge_xy = (int(root_bridge_xy[0]), int(root_bridge_xy[1]))
        if not COMPETITION_MODE:
            log_event(rnd, uid, "economy", f"({cur_xy[0]},{cur_xy[1]})",
                      "economy_axionite_plan_root_bridge_entry",
                      bx=root_bridge_xy[0], by=root_bridge_xy[1])
        if cur_xy == root_bridge_xy:
            if not COMPETITION_MODE:
                log_event(rnd, uid, "economy", f"({cur_xy[0]},{cur_xy[1]})",
                          "economy_axionite_plan_root_bridge_already_at_bridge")
            state.phase = "axionite_build_root_bridge"
            return
        steps = _astar_cardinal_plan(
            local_map,
            cur_xy,
            root_bridge_xy,
            max_expansions=640,
            tile_passable_fn=lambda x, y: _is_general_movement_passable(
                local_map,
                x,
                y,
                root_bridge_xy,
            ),
        )
        if not steps:
            if not COMPETITION_MODE:
                log_event(rnd, uid, "economy", f"({cur_xy[0]},{cur_xy[1]})",
                          "economy_axionite_plan_root_bridge_no_path",
                          bx=root_bridge_xy[0], by=root_bridge_xy[1])
            state.phase = "axionite_pick_root_bridge"
            return
        if not COMPETITION_MODE:
            log_event(rnd, uid, "economy", f"({cur_xy[0]},{cur_xy[1]})",
                      "economy_axionite_plan_root_bridge_path_found",
                      bx=root_bridge_xy[0], by=root_bridge_xy[1],
                      steps=len(steps))
        state.plan_steps = steps
        state.plan_index = 0
        state.defer_step_once = True
        state.phase = "axionite_follow_root_bridge"
        return

    if state.phase == "axionite_follow_root_bridge":
        if not (isinstance(root_bridge_xy, tuple) and len(root_bridge_xy) == 2):
            if not COMPETITION_MODE:
                log_event(rnd, uid, "economy", f"({cur_xy[0]},{cur_xy[1]})",
                          "economy_axionite_follow_root_bridge_no_bridge_xy")
            state.phase = "axionite_pick_root_bridge"
            return
        root_bridge_xy = (int(root_bridge_xy[0]), int(root_bridge_xy[1]))
        if not COMPETITION_MODE:
            log_event(rnd, uid, "economy", f"({cur_xy[0]},{cur_xy[1]})",
                      "economy_axionite_follow_root_bridge_entry",
                      bx=root_bridge_xy[0], by=root_bridge_xy[1],
                      plan_index=state.plan_index)
        if cur_xy == root_bridge_xy:
            state.plan_steps = ()
            state.plan_index = 0
            state.defer_step_once = False
            state.phase = "axionite_build_root_bridge"
            return

        if state.defer_step_once:
            state.defer_step_once = False
            return

        if state.plan_index >= len(state.plan_steps):
            if not COMPETITION_MODE:
                log_event(rnd, uid, "economy", f"({cur_xy[0]},{cur_xy[1]})",
                          "economy_axionite_follow_root_bridge_replan",
                          bx=root_bridge_xy[0], by=root_bridge_xy[1])
            state.phase = "axionite_plan_root_bridge"
            return

        nxt = state.plan_steps[state.plan_index]
        if not _is_adjacent_step(cur_xy, nxt):
            if not COMPETITION_MODE:
                log_event(rnd, uid, "economy", f"({cur_xy[0]},{cur_xy[1]})",
                          "economy_axionite_follow_root_bridge_step_invalid",
                          bx=root_bridge_xy[0], by=root_bridge_xy[1])
            state.phase = "axionite_plan_root_bridge"
            return

        move_result = _execute_step_toward(c, local_map, cur_xy, nxt, rnd, uid)
        if move_result == "moved":
            state.issued_move_last_tick = True
            state.expected_xy_after_move = nxt
            state.plan_index += 1
            return
        if move_result in ("built", "wait_cd"):
            return
        if not COMPETITION_MODE:
            log_event(rnd, uid, "economy", f"({cur_xy[0]},{cur_xy[1]})",
                      "economy_axionite_follow_root_bridge_move_failed",
                      move_result=str(move_result))
        state.phase = "axionite_plan_root_bridge"
        return

    if state.phase == "axionite_build_root_bridge":
        if not COMPETITION_MODE:
            log_event(rnd, uid, "economy", f"({cur_xy[0]},{cur_xy[1]})",
                      "economy_axionite_build_root_bridge_entry",
                      root_xy_ok=isinstance(root_xy, tuple),
                      bridge_xy_ok=isinstance(root_bridge_xy, tuple))
        if not (isinstance(root_xy, tuple) and len(root_xy) == 2):
            state.phase = "axionite_pick_root"
            return
        if not (isinstance(root_bridge_xy, tuple) and len(root_bridge_xy) == 2):
            state.phase = "axionite_pick_root_bridge"
            return
        root_xy = (int(root_xy[0]), int(root_xy[1]))
        root_bridge_xy = (int(root_bridge_xy[0]), int(root_bridge_xy[1]))
        if cur_xy != root_bridge_xy:
            if not COMPETITION_MODE:
                log_event(rnd, uid, "economy", f"({cur_xy[0]},{cur_xy[1]})",
                          "economy_axionite_build_root_bridge_not_at_bridge",
                          bx=root_bridge_xy[0], by=root_bridge_xy[1])
            state.phase = "axionite_plan_root_bridge"
            return

        bridge_result = _build_bridge_on_tile(
            c,
            root_bridge_xy,
            root_xy,
            rnd,
            uid,
            state=state,
            require_target_match_for_existing=True,
        )
        if bridge_result == "wait_cd":
            return
        if bridge_result in ("built", "already_built"):
            ctx["root_bridge_reused"] = (bridge_result == "already_built")
            if ctx["root_bridge_reused"]:
                log_event(
                    rnd,
                    uid,
                    "economy",
                    f"({cur_xy[0]},{cur_xy[1]})",
                    "economy_axionite_root_bridge_reused",
                    bx=root_bridge_xy[0],
                    by=root_bridge_xy[1],
                    rx=root_xy[0],
                    ry=root_xy[1],
                )
            # New ordering: root bridge is built after the ti→foundry link
            # completes, so from here we proceed directly to the core route.
            state.phase = "axionite_pick_core_route"
            return

        if _is_enemy_replaceable_blocker_at(local_map, root_bridge_xy[0], root_bridge_xy[1]):
            _start_network_attack(
                state,
                root_bridge_xy,
                "axionite_build_root_bridge",
                "axionite_root_bridge_tile_blocked",
            )
            return
        state.phase = "axionite_pick_root_bridge"
        return

    if state.phase == "axionite_pick_titanium_pair":
        if not COMPETITION_MODE:
            log_event(rnd, uid, "economy", f"({cur_xy[0]},{cur_xy[1]})",
                      "economy_axionite_pick_titanium_pair_entry")
        foundry_xy = ctx.get("foundry_xy")
        if not (isinstance(foundry_xy, tuple) and len(foundry_xy) == 2):
            if not COMPETITION_MODE:
                log_event(rnd, uid, "economy", f"({cur_xy[0]},{cur_xy[1]})",
                          "economy_axionite_pick_titanium_pair_no_foundry")
            state.phase = "axionite_pick_foundry"
            return
        foundry_xy = (int(foundry_xy[0]), int(foundry_xy[1]))

        paired_links = ctx.get("paired_ti_foundry_links")
        if not isinstance(paired_links, set):
            paired_links = set()
            ctx["paired_ti_foundry_links"] = paired_links

        blocked_links = ctx.get("blocked_ti_foundry_links")
        if not isinstance(blocked_links, set):
            blocked_links = set()
            ctx["blocked_ti_foundry_links"] = blocked_links

        ti_candidates = _collect_friendly_titanium_harvesters(local_map, state)
        ti_candidates.sort(key=lambda p: (
            _manhattan(p, foundry_xy), p[0], p[1]))

        plan_candidates = []
        for ti_xy in ti_candidates:
            pair_key = _axionite_ti_pair_key(foundry_xy, ti_xy)
            if pair_key in paired_links or pair_key in blocked_links:
                continue
            source_xy = _pick_pair_link_endpoint_near_harvester(
                local_map, ti_xy, foundry_xy,
            )
            if not (isinstance(source_xy, tuple) and len(source_xy) == 2):
                continue
            plan_candidates.append({
                "ti_xy": ti_xy,
                "source_xy": source_xy,
                "pair_key": pair_key,
                "pair_mode": "existing",
            })

        if not plan_candidates:
            if not COMPETITION_MODE:
                log_event(rnd, uid, "economy", f"({cur_xy[0]},{cur_xy[1]})",
                          "economy_axionite_pick_titanium_pair_no_candidates",
                          fx=foundry_xy[0], fy=foundry_xy[1],
                          ti_harvesters=len(ti_candidates))
            state.phase = "axionite_fallback_ti_pick_ore"
            return

        if not COMPETITION_MODE:
            log_event(rnd, uid, "economy", f"({cur_xy[0]},{cur_xy[1]})",
                      "economy_axionite_pick_titanium_pair_candidates_found",
                      fx=foundry_xy[0], fy=foundry_xy[1],
                      count=len(plan_candidates))
        ctx["ti_plan_candidates"] = plan_candidates
        ctx["ti_plan_index"] = 0
        state.phase = "axionite_pick_ti_route"
        return

    if state.phase == "axionite_pick_ti_route":
        # Ordering guarantee: the foundry's output bridge ("root") is built
        # AFTER this ti→foundry pipeline completes (see axionite_pick_root /
        # axionite_build_core_route). During ti→foundry planning the root
        # does not yet exist, so it does not need to be blocked. If that
        # ordering ever changes, add root_xy to the blocked set here.
        if not COMPETITION_MODE:
            log_event(rnd, uid, "economy", f"({cur_xy[0]},{cur_xy[1]})",
                      "economy_axionite_pick_ti_route_entry",
                      idx=int(ctx.get("ti_plan_index") or 0),
                      candidates=len(ctx.get("ti_plan_candidates") or []))
        foundry_xy = ctx.get("foundry_xy")
        if not (isinstance(foundry_xy, tuple) and len(foundry_xy) == 2):
            state.phase = "axionite_pick_foundry"
            return
        foundry_xy = (int(foundry_xy[0]), int(foundry_xy[1]))

        blocked_links = ctx.get("blocked_ti_foundry_links")
        if not isinstance(blocked_links, set):
            blocked_links = set()
            ctx["blocked_ti_foundry_links"] = blocked_links

        candidates = ctx.get("ti_plan_candidates")
        if not isinstance(candidates, list):
            candidates = []
        idx = int(ctx.get("ti_plan_index") or 0)

        # Skip already-blocked candidates (cheap, no A*)
        while idx < len(candidates):
            ck = candidates[idx].get("pair_key")
            if isinstance(ck, tuple) and ck in blocked_links:
                idx += 1
            else:
                break

        if idx >= len(candidates):
            if not COMPETITION_MODE:
                log_event(rnd, uid, "economy", f"({cur_xy[0]},{cur_xy[1]})",
                          "economy_axionite_pick_ti_route_exhausted",
                          total=len(candidates))
            ctx["ti_plan_candidates"] = []
            _axionite_ti_failure_transition(
                state, ctx, ore_xy, rnd, uid, cur_xy,
                reason="pick_ti_route_exhausted",
            )
            return

        ctx["ti_plan_index"] = idx
        entry = candidates[idx]
        ti_xy = entry.get("ti_xy")
        source_xy = entry.get("source_xy")
        pair_key = entry.get("pair_key")
        pair_mode = str(entry.get("pair_mode") or "existing")

        ti_xy_ok = isinstance(ti_xy, tuple) and len(ti_xy) == 2
        src_ok = isinstance(source_xy, tuple) and len(source_xy) == 2
        if not ti_xy_ok or not src_ok:
            if not COMPETITION_MODE:
                log_event(rnd, uid, "economy", f"({cur_xy[0]},{cur_xy[1]})",
                          "economy_axionite_pick_ti_route_bad_entry",
                          idx=idx, mode=pair_mode)
            blocked_links.add(pair_key)
            ctx["ti_plan_index"] = idx + 1
            return

        if not COMPETITION_MODE:
            log_event(rnd, uid, "economy", f"({cur_xy[0]},{cur_xy[1]})",
                      "economy_axionite_pick_ti_route_planning",
                      ti_x=int(ti_xy[0]), ti_y=int(ti_xy[1]),
                      mode=pair_mode, idx=idx)

        # Single A* call for this round — avoids routing through existing infra
        nodes = _plan_axionite_ti_link_nodes(
            local_map,
            source_xy,
            foundry_xy,
            blocked_tiles=_axionite_root_halo_tiles(ctx.get("root_xy")),
            block_friendly_infra=True,
        )

        if not nodes:
            blocked_links.add(pair_key)
            log_event(
                rnd, uid, "economy",
                f"({cur_xy[0]},{cur_xy[1]})",
                "economy_axionite_ti_route_plan_failed",
                ti=(f"({ti_xy[0]},{ti_xy[1]})" if ti_xy_ok else "none"),
                mode=pair_mode,
            )
            # Single-attempt policy: go straight to splitter fallback on the
            # first planning failure rather than iterating remaining harvesters
            # — per user spec, other candidates typically hit the same geometry.
            _axionite_ti_failure_transition(
                state, ctx, ore_xy, rnd, uid, cur_xy,
                reason="pick_ti_route_plan_failed",
            )
            return

        link_seq = _compose_axionite_link_sequence(
            nodes,
            foundry_xy,
            build_order="forward",
            terminal_mode="bridge",
        )
        if not link_seq:
            if not COMPETITION_MODE:
                log_event(rnd, uid, "economy", f"({cur_xy[0]},{cur_xy[1]})",
                          "economy_axionite_pick_ti_route_seq_empty",
                          ti_x=int(ti_xy[0]), ti_y=int(ti_xy[1]),
                          mode=pair_mode)
            blocked_links.add(pair_key)
            _axionite_ti_failure_transition(
                state, ctx, ore_xy, rnd, uid, cur_xy,
                reason="pick_ti_route_seq_empty",
            )
            return

        if not COMPETITION_MODE:
            log_event(rnd, uid, "economy", f"({cur_xy[0]},{cur_xy[1]})",
                      "economy_axionite_pick_ti_route_ready",
                      ti_x=int(ti_xy[0]), ti_y=int(ti_xy[1]),
                      mode=pair_mode, nodes=len(nodes), seq=len(link_seq))
        ctx["ti_pair_xy"] = ti_xy
        ctx["ti_pair_key"] = pair_key
        ctx["pair_mode"] = pair_mode
        ctx["link_nodes"] = nodes
        ctx["link_terminal_xy"] = foundry_xy
        ctx["link_sequence"] = link_seq
        ctx["link_sequence_index"] = 0
        state.phase = "axionite_build_ti_route"
        return

    if state.phase == "axionite_fallback_ti_pick_ore":
        if not COMPETITION_MODE:
            log_event(rnd, uid, "economy", f"({cur_xy[0]},{cur_xy[1]})",
                      "economy_axionite_fallback_ti_pick_ore_entry")
        root_halo = _axionite_root_halo_tiles(ctx.get("root_xy"))
        blocked_ti = _axionite_blocked_fallback_titanium_ores(state)
        foundry_xy = ctx.get("foundry_xy")
        if not (isinstance(foundry_xy, tuple) and len(foundry_xy) == 2):
            state.phase = "axionite_pick_foundry"
            return
        foundry_xy = (int(foundry_xy[0]), int(foundry_xy[1]))

        paired_links = ctx.get("paired_ti_foundry_links")
        if not isinstance(paired_links, set):
            paired_links = set()
            ctx["paired_ti_foundry_links"] = paired_links

        blocked_links = ctx.get("blocked_ti_foundry_links")
        if not isinstance(blocked_links, set):
            blocked_links = set()
            ctx["blocked_ti_foundry_links"] = blocked_links

        known_ti_all = _known_unharvested_titanium(
            local_map,
            core_xy=state.core_xy,
            min_core_cheb=_ORE_CORE_EXCLUSION_MIN_CHEB,
        )
        known_ti = [
            p for p in known_ti_all
            if (
                p not in blocked_ti
                and p not in root_halo
                and _axionite_ti_pair_key(foundry_xy, p) not in paired_links
                and _axionite_ti_pair_key(foundry_xy, p) not in blocked_links
            )
        ]
        if not known_ti:
            if not COMPETITION_MODE:
                log_event(rnd, uid, "economy", f"({cur_xy[0]},{cur_xy[1]})",
                          "economy_axionite_fallback_ti_pick_ore_none_available",
                          known_all=len(known_ti_all), blocked=len(blocked_ti))
            return

        ore_xy = min(
            known_ti,
            key=lambda p: (
                _manhattan(p, foundry_xy),
                p[0],
                p[1],
            ),
        )
        if not COMPETITION_MODE:
            log_event(rnd, uid, "economy", f"({cur_xy[0]},{cur_xy[1]})",
                      "economy_axionite_fallback_ti_pick_ore_selected",
                      ore_x=int(ore_xy[0]), ore_y=int(ore_xy[1]),
                      candidates=len(known_ti))
        ctx["fallback_ti_ore_xy"] = ore_xy
        ctx["fallback_ti_goal_xy"] = None
        state.phase = "axionite_fallback_ti_pick_goal"
        return

    if state.phase == "axionite_fallback_ti_pick_goal":
        if not COMPETITION_MODE:
            _fto = ctx.get("fallback_ti_ore_xy")
            _fto_ok = isinstance(_fto, tuple) and len(_fto) == 2
            log_event(rnd, uid, "economy", f"({cur_xy[0]},{cur_xy[1]})",
                      "economy_axionite_fallback_ti_pick_goal_entry",
                      ore_x=int(_fto[0]) if _fto_ok else -1,
                      ore_y=int(_fto[1]) if _fto_ok else -1)
        fallback_ore = ctx.get("fallback_ti_ore_xy")
        if not (isinstance(fallback_ore, tuple) and len(fallback_ore) == 2):
            if not COMPETITION_MODE:
                log_event(rnd, uid, "economy", f"({cur_xy[0]},{cur_xy[1]})",
                          "economy_axionite_fallback_ti_pick_goal_no_ore")
            state.phase = "axionite_fallback_ti_pick_ore"
            return
        goal_xy = _pick_titanium_adjacent_goal(local_map, fallback_ore, cur_xy)
        if not (isinstance(goal_xy, tuple) and len(goal_xy) == 2):
            if not COMPETITION_MODE:
                log_event(rnd, uid, "economy", f"({cur_xy[0]},{cur_xy[1]})",
                          "economy_axionite_fallback_ti_pick_goal_no_adjacent",
                          ore_x=int(fallback_ore[0]), ore_y=int(fallback_ore[1]))
            _axionite_blocked_fallback_titanium_ores(state).add(
                (int(fallback_ore[0]), int(fallback_ore[1]))
            )
            ctx["fallback_ti_ore_xy"] = None
            state.phase = "axionite_fallback_ti_pick_ore"
            return
        if not COMPETITION_MODE:
            log_event(rnd, uid, "economy", f"({cur_xy[0]},{cur_xy[1]})",
                      "economy_axionite_fallback_ti_pick_goal_selected",
                      ore_x=int(fallback_ore[0]), ore_y=int(fallback_ore[1]),
                      goal_x=int(goal_xy[0]), goal_y=int(goal_xy[1]))
        ctx["fallback_ti_goal_xy"] = goal_xy
        state.phase = "axionite_fallback_ti_plan_goal"
        state.plan_steps = ()
        state.plan_index = 0
        state.defer_step_once = False
        return

    if state.phase == "axionite_fallback_ti_plan_goal":
        fallback_ore = ctx.get("fallback_ti_ore_xy")
        fallback_goal = ctx.get("fallback_ti_goal_xy")
        if not (
            isinstance(fallback_ore, tuple)
            and len(fallback_ore) == 2
            and isinstance(fallback_goal, tuple)
            and len(fallback_goal) == 2
        ):
            if not COMPETITION_MODE:
                log_event(rnd, uid, "economy", f"({cur_xy[0]},{cur_xy[1]})",
                          "economy_axionite_fallback_ti_plan_goal_missing_ctx")
            state.phase = "axionite_fallback_ti_pick_goal"
            return

        fallback_goal = (int(fallback_goal[0]), int(fallback_goal[1]))
        if not COMPETITION_MODE:
            log_event(rnd, uid, "economy", f"({cur_xy[0]},{cur_xy[1]})",
                      "economy_axionite_fallback_ti_plan_goal_entry",
                      ore_x=int(fallback_ore[0]), ore_y=int(fallback_ore[1]),
                      goal_x=fallback_goal[0], goal_y=fallback_goal[1])
        root_halo = _axionite_root_halo_tiles(ctx.get("root_xy"))
        if cur_xy == fallback_goal:
            if not COMPETITION_MODE:
                log_event(rnd, uid, "economy", f"({cur_xy[0]},{cur_xy[1]})",
                          "economy_axionite_fallback_ti_plan_goal_already_at_goal")
            state.phase = "axionite_fallback_ti_build_harvester"
            return

        steps = _astar_cardinal_plan(
            local_map,
            cur_xy,
            fallback_goal,
            max_expansions=768,
            tile_passable_fn=lambda x, y: (
                (x, y) == fallback_goal
                or ((x, y) not in root_halo)
            ) and _is_general_movement_passable(local_map, x, y, fallback_goal),
        )
        if not steps:
            if not COMPETITION_MODE:
                log_event(rnd, uid, "economy", f"({cur_xy[0]},{cur_xy[1]})",
                          "economy_axionite_fallback_ti_plan_goal_no_path",
                          ore_x=int(fallback_ore[0]), ore_y=int(fallback_ore[1]))
            _axionite_blocked_fallback_titanium_ores(state).add(
                (int(fallback_ore[0]), int(fallback_ore[1]))
            )
            ctx["fallback_ti_ore_xy"] = None
            ctx["fallback_ti_goal_xy"] = None
            state.phase = "axionite_fallback_ti_pick_ore"
            return

        if not COMPETITION_MODE:
            log_event(rnd, uid, "economy", f"({cur_xy[0]},{cur_xy[1]})",
                      "economy_axionite_fallback_ti_plan_goal_path_found",
                      goal_x=fallback_goal[0], goal_y=fallback_goal[1],
                      steps=len(steps))
        state.plan_steps = steps
        state.plan_index = 0
        state.defer_step_once = True
        state.phase = "axionite_fallback_ti_follow_goal"
        return

    if state.phase == "axionite_fallback_ti_follow_goal":
        fallback_goal = ctx.get("fallback_ti_goal_xy")
        if not (isinstance(fallback_goal, tuple) and len(fallback_goal) == 2):
            if not COMPETITION_MODE:
                log_event(rnd, uid, "economy", f"({cur_xy[0]},{cur_xy[1]})",
                          "economy_axionite_fallback_ti_follow_goal_no_goal")
            state.phase = "axionite_fallback_ti_pick_goal"
            return

        fallback_goal = (int(fallback_goal[0]), int(fallback_goal[1]))
        if not COMPETITION_MODE:
            log_event(rnd, uid, "economy", f"({cur_xy[0]},{cur_xy[1]})",
                      "economy_axionite_fallback_ti_follow_goal_entry",
                      goal_x=fallback_goal[0], goal_y=fallback_goal[1],
                      plan_index=state.plan_index)
        if cur_xy == fallback_goal:
            state.plan_steps = ()
            state.plan_index = 0
            state.defer_step_once = False
            state.phase = "axionite_fallback_ti_build_harvester"
            return

        if state.defer_step_once:
            state.defer_step_once = False
            return
        if state.plan_index >= len(state.plan_steps):
            if not COMPETITION_MODE:
                log_event(rnd, uid, "economy", f"({cur_xy[0]},{cur_xy[1]})",
                          "economy_axionite_fallback_ti_follow_goal_replan",
                          goal_x=fallback_goal[0], goal_y=fallback_goal[1])
            state.phase = "axionite_fallback_ti_plan_goal"
            return

        nxt = state.plan_steps[state.plan_index]
        if not _is_adjacent_step(cur_xy, nxt):
            if not COMPETITION_MODE:
                log_event(rnd, uid, "economy", f"({cur_xy[0]},{cur_xy[1]})",
                          "economy_axionite_fallback_ti_follow_goal_step_invalid")
            state.phase = "axionite_fallback_ti_plan_goal"
            return

        move_result = _execute_step_toward(c, local_map, cur_xy, nxt, rnd, uid)
        if move_result == "moved":
            state.issued_move_last_tick = True
            state.expected_xy_after_move = nxt
            state.plan_index += 1
            return
        if move_result in ("built", "wait_cd"):
            return
        if not COMPETITION_MODE:
            log_event(rnd, uid, "economy", f"({cur_xy[0]},{cur_xy[1]})",
                      "economy_axionite_fallback_ti_follow_goal_move_failed",
                      move_result=str(move_result))
        state.phase = "axionite_fallback_ti_plan_goal"
        return

    if state.phase == "axionite_fallback_ti_build_harvester":
        fallback_ore = ctx.get("fallback_ti_ore_xy")
        fallback_goal = ctx.get("fallback_ti_goal_xy")
        if not (
            isinstance(fallback_ore, tuple)
            and len(fallback_ore) == 2
            and isinstance(fallback_goal, tuple)
            and len(fallback_goal) == 2
        ):
            if not COMPETITION_MODE:
                log_event(rnd, uid, "economy", f"({cur_xy[0]},{cur_xy[1]})",
                          "economy_axionite_fallback_ti_build_harvester_missing_ctx")
            state.phase = "axionite_fallback_ti_pick_ore"
            return
        fallback_goal = (int(fallback_goal[0]), int(fallback_goal[1]))
        fallback_ore = (int(fallback_ore[0]), int(fallback_ore[1]))
        if not COMPETITION_MODE:
            log_event(rnd, uid, "economy", f"({cur_xy[0]},{cur_xy[1]})",
                      "economy_axionite_fallback_ti_build_harvester_entry",
                      ore_x=fallback_ore[0], ore_y=fallback_ore[1],
                      goal_x=fallback_goal[0], goal_y=fallback_goal[1])
        if cur_xy != fallback_goal:
            if not COMPETITION_MODE:
                log_event(rnd, uid, "economy", f"({cur_xy[0]},{cur_xy[1]})",
                          "economy_axionite_fallback_ti_build_harvester_not_at_goal",
                          ore_x=fallback_ore[0], ore_y=fallback_ore[1])
            state.phase = "axionite_fallback_ti_plan_goal"
            return
        if c.get_action_cooldown() > 0:
            return

        ore_pos = Position(fallback_ore[0], fallback_ore[1])
        try:
            if c.can_build_harvester(ore_pos):
                new_id = c.build_harvester(ore_pos)
                if isinstance(new_id, int):
                    state.built_entity_ids.add(new_id)
                state.built_harvester_positions.add(fallback_ore)
                ctx["ti_pair_xy"] = fallback_ore
                foundry_xy = ctx.get("foundry_xy")
                if not (isinstance(foundry_xy, tuple) and len(foundry_xy) == 2):
                    state.phase = "axionite_pick_foundry"
                    return
                foundry_xy = (int(foundry_xy[0]), int(foundry_xy[1]))

                source_xy = _pick_pair_link_endpoint_near_harvester(
                    local_map,
                    fallback_ore,
                    foundry_xy,
                )
                if not (isinstance(source_xy, tuple) and len(source_xy) == 2):
                    if not COMPETITION_MODE:
                        log_event(rnd, uid, "economy",
                                  f"({cur_xy[0]},{cur_xy[1]})",
                                  "economy_axionite_fallback_ti_build_harvester_no_endpoint",
                                  ore_x=fallback_ore[0], ore_y=fallback_ore[1])
                    _axionite_blocked_fallback_titanium_ores(
                        state).add(fallback_ore)
                    blocked_links = ctx.get("blocked_ti_foundry_links")
                    if isinstance(blocked_links, set):
                        blocked_links.add(_axionite_ti_pair_key(
                            foundry_xy, fallback_ore))
                    ctx["fallback_ti_ore_xy"] = None
                    ctx["fallback_ti_goal_xy"] = None
                    state.phase = "axionite_fallback_ti_pick_ore"
                    return

                if not COMPETITION_MODE:
                    log_event(rnd, uid, "economy",
                              f"({cur_xy[0]},{cur_xy[1]})",
                              "economy_axionite_fallback_ti_harvester_built",
                              ore_x=fallback_ore[0], ore_y=fallback_ore[1],
                              fx=foundry_xy[0], fy=foundry_xy[1])
                pair_key = _axionite_ti_pair_key(foundry_xy, fallback_ore)
                ctx["ti_pair_key"] = pair_key
                ctx["pair_mode"] = "fallback"
                ctx["ti_plan_candidates"] = [{
                    "ti_xy": fallback_ore,
                    "source_xy": source_xy,
                    "pair_key": pair_key,
                    "pair_mode": "fallback",
                }]
                ctx["ti_plan_index"] = 0
                state.phase = "axionite_pick_ti_route"
                return
        except GameError:
            return
        return

    if state.phase == "axionite_build_ti_route":
        if not COMPETITION_MODE:
            log_event(rnd, uid, "economy", f"({cur_xy[0]},{cur_xy[1]})",
                      "economy_axionite_build_ti_route_entry",
                      seq_idx=int(ctx.get("link_sequence_index", 0)),
                      seq_len=len(tuple(ctx.get("link_sequence", ()))))
        seq_result = _run_axionite_link_build_sequence(
            c,
            state,
            local_map,
            cur_xy,
            rnd,
            uid,
            ctx,
            "axionite_build_ti_route",
        )
        if seq_result == "done":
            pair_key = ctx.get("ti_pair_key")
            paired_links = ctx.get("paired_ti_foundry_links")
            if (
                isinstance(pair_key, tuple)
                and len(pair_key) == 4
                and isinstance(paired_links, set)
            ):
                paired_links.add(tuple(int(v) for v in pair_key))

            nodes = tuple(ctx.get("link_nodes", ()))
            feeder_tiles = []
            for p in nodes:
                if isinstance(p, tuple) and len(p) == 2:
                    feeder_tiles.append((int(p[0]), int(p[1])))
            ctx["ti_feeder_tiles"] = tuple(feeder_tiles)
            # Permanently mark the built route as an obstacle for other
            # network building states (movement planning ignores this set).
            for ft in feeder_tiles:
                state.axionite_ti_route_blocked_tiles.add(ft)
            _sync_dynamic_blocked_tiles(state, local_map)
            log_event(
                rnd,
                uid,
                "economy",
                f"({cur_xy[0]},{cur_xy[1]})",
                "economy_axionite_ti_route_complete",
                pair=(
                    f"({pair_key[2]},{pair_key[3]})"
                    if isinstance(pair_key, tuple) and len(pair_key) == 4
                    else "none"
                ),
                nodes=len(nodes),
                feeder_tiles=len(feeder_tiles),
            )
            ctx["ti_pair_key"] = None

            # Fix 2: if the foundry already has an output-ready cardinal
            # transport (a friendly bridge whose target is not the foundry,
            # or a friendly conveyor/splitter/armoured-conveyor whose
            # direction points away from the foundry), the refined axionite
            # can flow through that existing network — no root bridge or
            # core route needed. Short-circuit to done.
            skip_foundry_xy = ctx.get("foundry_xy")
            if (
                isinstance(skip_foundry_xy, tuple)
                and len(skip_foundry_xy) == 2
                and _foundry_has_output_ready_adjacent_transport(
                    local_map,
                    (int(skip_foundry_xy[0]), int(skip_foundry_xy[1])),
                )
            ):
                if not COMPETITION_MODE:
                    log_event(
                        rnd, uid, "economy", f"({cur_xy[0]},{cur_xy[1]})",
                        "economy_axionite_skip_root_existing_output",
                        fx=int(skip_foundry_xy[0]),
                        fy=int(skip_foundry_xy[1]),
                    )
                _reset_axionite_pipeline_for_next_ore(state, ctx)
                state.phase = "axionite_done"
                return

            # New ordering (per user fix): root bridge placement runs AFTER
            # the ti→foundry link completes. Any root_bridge_reused logic
            # used to live here when root bridge came first; it now lives in
            # axionite_build_root_bridge, which is reached next.
            state.phase = "axionite_pick_root"
            state.plan_steps = ()
            state.plan_index = 0
            state.defer_step_once = False
            return
        if seq_result == "failed":
            pair_key = ctx.get("ti_pair_key")
            blocked_links = ctx.get("blocked_ti_foundry_links")
            if (
                isinstance(pair_key, tuple)
                and len(pair_key) == 4
                and isinstance(blocked_links, set)
            ):
                blocked_links.add(tuple(int(v) for v in pair_key))

            log_event(
                rnd,
                uid,
                "economy",
                f"({cur_xy[0]},{cur_xy[1]})",
                "economy_axionite_ti_route_failed",
                pair=(
                    f"({pair_key[2]},{pair_key[3]})"
                    if isinstance(pair_key, tuple) and len(pair_key) == 4
                    else "none"
                ),
                mode=str(ctx.get("pair_mode") or "none"),
                seq_idx=int(ctx.get("link_sequence_index", 0)),
                seq_len=len(tuple(ctx.get("link_sequence", ()))),
            )

            ctx["ti_pair_key"] = None
            ctx["ti_pair_xy"] = None
            ctx["link_sequence"] = ()
            ctx["link_sequence_index"] = 0
            # Single-attempt policy: on any ti-route build failure — whether
            # the harvester-sourced "existing" route or the splitter "splice"
            # route — escalate via the unified failure transition (splitter
            # fallback first, then abandon). Previously the existing-mode
            # branch looped through remaining candidates and the splice-mode
            # branch fell into axionite_fallback_ti_pick_ore; both caused the
            # multi-fallback loops the user observed.
            _axionite_ti_failure_transition(
                state, ctx, ore_xy, rnd, uid, cur_xy,
                reason=f"build_ti_route_failed_{ctx.get('pair_mode') or 'existing'}",
            )
        return

    # ---------------------------------------------------------------
    # Splitter-splice fallback phases (Item C).
    # Triggered when every direct ti→foundry candidate has failed. The
    # bot walks to a consistent conveyor near the foundry, clears a
    # perpendicular side tile, replaces the conveyor with a splitter,
    # and re-plans a ti→foundry link from the new splitter side-tap.
    # ---------------------------------------------------------------
    if state.phase == "axionite_fallback_pick_consistent":
        state.axionite_fallback_attempted = True
        foundry_xy_c = ctx.get("foundry_xy")
        if not (isinstance(foundry_xy_c, tuple) and len(foundry_xy_c) == 2):
            state.phase = "axionite_pick_foundry"
            return
        foundry_xy_c = (int(foundry_xy_c[0]), int(foundry_xy_c[1]))

        splice = _find_consistent_conveyor_near_foundry(
            local_map, foundry_xy_c)
        if splice is None:
            log_event(
                rnd, uid, "economy",
                f"({cur_xy[0]},{cur_xy[1]})",
                "economy_axionite_fallback_no_consistent_conveyor",
                fx=foundry_xy_c[0], fy=foundry_xy_c[1],
            )
            # No splice candidate → abandon this ore permanently.
            blocked_ax.add((int(ore_xy[0]), int(ore_xy[1])))
            _reset_axionite_pipeline_for_next_ore(state, ctx)
            return

        state.axionite_fallback_splitter_xy = splice["splitter_xy"]
        state.axionite_fallback_splitter_dir_name = splice["splitter_dir_name"]
        state.axionite_fallback_side_tap_xy = splice["side_tap_xy"]
        state.axionite_fallback_foundry_xy = foundry_xy_c
        state.axionite_fallback_ore_xy = (int(ore_xy[0]), int(ore_xy[1])) \
            if isinstance(ore_xy, tuple) and len(ore_xy) == 2 else None
        state.axionite_fallback_return_xy = cur_xy
        state.plan_steps = ()
        state.plan_index = 0
        state.defer_step_once = False
        log_event(
            rnd, uid, "economy",
            f"({cur_xy[0]},{cur_xy[1]})",
            "economy_axionite_fallback_splice_selected",
            sx=splice["splitter_xy"][0], sy=splice["splitter_xy"][1],
            dir=splice["splitter_dir_name"],
            tx=splice["side_tap_xy"][0], ty=splice["side_tap_xy"][1],
        )
        state.phase = "axionite_fallback_plan_goto"
        return

    if state.phase == "axionite_fallback_plan_goto":
        splitter_xy = state.axionite_fallback_splitter_xy
        side_tap_xy = state.axionite_fallback_side_tap_xy
        if not (
            isinstance(splitter_xy, tuple) and len(splitter_xy) == 2
            and isinstance(side_tap_xy, tuple) and len(side_tap_xy) == 2
        ):
            state.phase = "axionite_fallback_pick_consistent"
            return

        # Target the side-tap tile as the stand position — from there the
        # splitter (cardinally adjacent) is within action radius for both
        # destroy and build, and we already know the tile is walkable
        # (MAP_FREE or friendly road).
        if cur_xy == side_tap_xy:
            state.phase = "axionite_fallback_clear_side"
            return

        steps = _astar_cardinal_plan(
            local_map,
            cur_xy,
            side_tap_xy,
            max_expansions=640,
            tile_passable_fn=lambda x, y: _is_general_movement_passable(
                local_map, x, y, side_tap_xy,
            ),
        )
        if not steps:
            steps = _astar_cardinal_plan(
                local_map,
                cur_xy,
                side_tap_xy,
                max_expansions=640,
                tile_passable_fn=lambda x, y: _is_general_movement_passable(
                    local_map, x, y, side_tap_xy, respect_halo=False,
                ),
            )
        if not steps:
            log_event(
                rnd, uid, "economy",
                f"({cur_xy[0]},{cur_xy[1]})",
                "economy_axionite_fallback_goto_plan_failed",
                tx=side_tap_xy[0], ty=side_tap_xy[1],
            )
            if isinstance(state.axionite_fallback_ore_xy, tuple):
                blocked_ax.add(state.axionite_fallback_ore_xy)
            _reset_axionite_pipeline_for_next_ore(state, ctx)
            return

        state.plan_steps = steps
        state.plan_index = 0
        state.phase = "axionite_fallback_goto"
        return

    if state.phase == "axionite_fallback_goto":
        side_tap_xy = state.axionite_fallback_side_tap_xy
        if not (isinstance(side_tap_xy, tuple) and len(side_tap_xy) == 2):
            state.phase = "axionite_fallback_pick_consistent"
            return
        if cur_xy == side_tap_xy:
            state.phase = "axionite_fallback_clear_side"
            state.plan_steps = ()
            state.plan_index = 0
            return

        if state.plan_index >= len(state.plan_steps):
            state.phase = "axionite_fallback_plan_goto"
            return
        nxt = state.plan_steps[state.plan_index]
        if not _is_adjacent_step(cur_xy, nxt):
            state.phase = "axionite_fallback_plan_goto"
            return
        move_result = _execute_step_toward(c, local_map, cur_xy, nxt, rnd, uid)
        if move_result == "moved":
            state.issued_move_last_tick = True
            state.expected_xy_after_move = nxt
            state.plan_index += 1
            return
        if move_result in ("built", "wait_cd"):
            return
        state.phase = "axionite_fallback_plan_goto"
        return

    if state.phase == "axionite_fallback_clear_side":
        side_tap_xy = state.axionite_fallback_side_tap_xy
        splitter_xy = state.axionite_fallback_splitter_xy
        if not (
            isinstance(side_tap_xy, tuple) and len(side_tap_xy) == 2
            and isinstance(splitter_xy, tuple) and len(splitter_xy) == 2
        ):
            state.phase = "axionite_fallback_pick_consistent"
            return

        if cur_xy != side_tap_xy:
            # Movement slipped — replan.
            state.phase = "axionite_fallback_plan_goto"
            return

        tap_rec = _known_building_at(local_map, side_tap_xy[0], side_tap_xy[1])
        my_team = getattr(local_map, "my_team", None)

        if isinstance(tap_rec, dict):
            etype = tap_rec.get("entity_type")
            team = tap_rec.get("team")

            if team == my_team and etype == EntityType.ROAD:
                # Friendly road: destroy is free and doesn't cost action cd.
                try:
                    tap_pos = Position(side_tap_xy[0], side_tap_xy[1])
                    if c.can_destroy(tap_pos):
                        c.destroy(tap_pos)
                        log_event(
                            rnd, uid, "economy",
                            f"({cur_xy[0]},{cur_xy[1]})",
                            "economy_axionite_fallback_destroyed_friendly_road",
                            tx=side_tap_xy[0], ty=side_tap_xy[1],
                        )
                except GameError:
                    pass
                state.phase = "axionite_fallback_replace_splitter"
                return

            # Enemy roads are destroyable via attack; enemy transports (non
            # armoured) too. Builder must stand on the tile to attack. Since
            # we are already standing on side_tap_xy, we can fire directly.
            enemy_attackable = (
                team != my_team
                and etype in (
                    EntityType.ROAD,
                    EntityType.CONVEYOR,
                    EntityType.SPLITTER,
                    EntityType.BRIDGE,
                )
            )
            if enemy_attackable:
                if c.get_action_cooldown() == 0:
                    try:
                        my_pos = c.get_position()
                        if c.can_fire(my_pos):
                            c.fire(my_pos)
                            log_event(
                                rnd, uid, "economy",
                                f"({cur_xy[0]},{cur_xy[1]})",
                                "economy_axionite_fallback_attack_enemy",
                                tx=side_tap_xy[0], ty=side_tap_xy[1],
                                etype=str(etype.name).lower(),
                            )
                    except GameError:
                        pass
                # Stay in this phase until the enemy tile is cleared.
                return

            # Armoured conveyor or unexpected friendly building → cannot
            # clear. Abandon this ore.
            log_event(
                rnd, uid, "economy",
                f"({cur_xy[0]},{cur_xy[1]})",
                "economy_axionite_fallback_side_unclearable",
                tx=side_tap_xy[0], ty=side_tap_xy[1],
                etype=str(etype.name).lower() if etype is not None else "none",
                friendly=1 if team == my_team else 0,
            )
            if isinstance(state.axionite_fallback_ore_xy, tuple):
                blocked_ax.add(state.axionite_fallback_ore_xy)
            _reset_axionite_pipeline_for_next_ore(state, ctx)
            return

        # No building — MAP_FREE (or MAP_UNKNOWN): nothing to clear.
        state.phase = "axionite_fallback_replace_splitter"
        return

    if state.phase == "axionite_fallback_replace_splitter":
        splitter_xy = state.axionite_fallback_splitter_xy
        side_tap_xy = state.axionite_fallback_side_tap_xy
        dir_name = state.axionite_fallback_splitter_dir_name
        if not (
            isinstance(splitter_xy, tuple) and len(splitter_xy) == 2
            and isinstance(side_tap_xy, tuple) and len(side_tap_xy) == 2
            and isinstance(dir_name, str)
        ):
            state.phase = "axionite_fallback_pick_consistent"
            return

        # Bot must be adjacent to splitter_xy (not standing on it — splitters
        # are not walkable-while-building targets for non-walkable builds).
        if cur_xy != side_tap_xy:
            state.phase = "axionite_fallback_plan_goto"
            return

        splitter_pos = Position(splitter_xy[0], splitter_xy[1])
        # Verify the original conveyor still has the expected direction before
        # destroying — the world may have changed since the scan.
        rec = _known_building_at(local_map, splitter_xy[0], splitter_xy[1])
        my_team = getattr(local_map, "my_team", None)
        if not (
            isinstance(rec, dict)
            and rec.get("team") == my_team
            and rec.get("entity_type") == EntityType.CONVEYOR
            and rec.get("direction") is not None
            and rec.get("direction").name == dir_name
        ):
            log_event(
                rnd, uid, "economy",
                f"({cur_xy[0]},{cur_xy[1]})",
                "economy_axionite_fallback_splice_invalid",
                sx=splitter_xy[0], sy=splitter_xy[1],
                dir=dir_name,
            )
            if isinstance(state.axionite_fallback_ore_xy, tuple):
                blocked_ax.add(state.axionite_fallback_ore_xy)
            _reset_axionite_pipeline_for_next_ore(state, ctx)
            return

        # Destroy the existing conveyor (free; no cooldown).
        try:
            if c.can_destroy(splitter_pos):
                c.destroy(splitter_pos)
                log_event(
                    rnd, uid, "economy",
                    f"({cur_xy[0]},{cur_xy[1]})",
                    "economy_axionite_fallback_destroyed_conveyor",
                    sx=splitter_xy[0], sy=splitter_xy[1],
                )
        except GameError:
            pass

        if c.get_action_cooldown() > 0:
            return  # Wait for cooldown to build splitter next round.

        out_dir = _CARDINAL_DIRECTION_BY_NAME.get(dir_name)
        if out_dir is None:
            if isinstance(state.axionite_fallback_ore_xy, tuple):
                blocked_ax.add(state.axionite_fallback_ore_xy)
            _reset_axionite_pipeline_for_next_ore(state, ctx)
            return

        try:
            if c.can_build_splitter(splitter_pos, out_dir):
                new_id = c.build_splitter(splitter_pos, out_dir)
                if isinstance(new_id, int):
                    state.built_entity_ids.add(new_id)
                state.built_transport_positions.add(splitter_xy)
                log_event(
                    rnd, uid, "economy",
                    f"({cur_xy[0]},{cur_xy[1]})",
                    "economy_axionite_fallback_splitter_built",
                    sx=splitter_xy[0], sy=splitter_xy[1],
                    dir=dir_name,
                )
                state.phase = "axionite_fallback_replan"
                return
        except GameError:
            pass
        # Build failed (likely insufficient resources or tile state) — stay
        # and retry next round.
        return

    if state.phase == "axionite_fallback_replan":
        side_tap_xy = state.axionite_fallback_side_tap_xy
        foundry_xy_r = state.axionite_fallback_foundry_xy
        if not (
            isinstance(side_tap_xy, tuple) and len(side_tap_xy) == 2
            and isinstance(foundry_xy_r, tuple) and len(foundry_xy_r) == 2
        ):
            state.phase = "axionite_fallback_pick_consistent"
            return

        nodes = _plan_axionite_ti_link_nodes(
            local_map,
            side_tap_xy,
            foundry_xy_r,
            blocked_tiles=_axionite_root_halo_tiles(ctx.get("root_xy")),
            block_friendly_infra=True,
        )
        if not nodes:
            log_event(
                rnd, uid, "economy",
                f"({cur_xy[0]},{cur_xy[1]})",
                "economy_axionite_fallback_replan_failed",
                tx=side_tap_xy[0], ty=side_tap_xy[1],
                fx=foundry_xy_r[0], fy=foundry_xy_r[1],
            )
            if isinstance(state.axionite_fallback_ore_xy, tuple):
                blocked_ax.add(state.axionite_fallback_ore_xy)
            _reset_axionite_pipeline_for_next_ore(state, ctx)
            return

        link_seq = _compose_axionite_link_sequence(
            nodes,
            foundry_xy_r,
            build_order="forward",
            terminal_mode="bridge",
        )
        if not link_seq:
            if isinstance(state.axionite_fallback_ore_xy, tuple):
                blocked_ax.add(state.axionite_fallback_ore_xy)
            _reset_axionite_pipeline_for_next_ore(state, ctx)
            return

        # Plug into the normal ti_route build machinery — it iterates
        # link_sequence and updates ti_feeder_tiles on completion. The
        # pair_key in the ti_route path must be a 4-tuple of ints, since the
        # completion/failure branches do `int(v) for v in pair_key`. We
        # don't have a real (foundry, ti-harvester) pair here (the splice
        # source is a splitter, not a harvester), so leave ti_pair_key as
        # None — the add-to-paired/blocked branches gracefully skip None.
        ctx["ti_pair_xy"] = side_tap_xy
        ctx["ti_pair_key"] = None
        ctx["pair_mode"] = "splice"
        ctx["link_nodes"] = nodes
        ctx["link_terminal_xy"] = foundry_xy_r
        ctx["link_sequence"] = link_seq
        ctx["link_sequence_index"] = 0
        log_event(
            rnd, uid, "economy",
            f"({cur_xy[0]},{cur_xy[1]})",
            "economy_axionite_fallback_replan_ready",
            tx=side_tap_xy[0], ty=side_tap_xy[1],
            fx=foundry_xy_r[0], fy=foundry_xy_r[1],
            nodes=len(nodes), seq=len(link_seq),
        )
        state.phase = "axionite_build_ti_route"
        return

    if state.phase == "axionite_pick_core_route":
        replan_from_cur = bool(ctx.pop("core_route_replan_from_cur", False))
        if not COMPETITION_MODE:
            log_event(rnd, uid, "economy", f"({cur_xy[0]},{cur_xy[1]})",
                      "economy_axionite_pick_core_route_entry",
                      root_xy_ok=isinstance(ctx.get("root_xy"), tuple),
                      blocked_core=len(ctx.get("blocked_core_link_candidates") or set()),
                      from_cur=1 if replan_from_cur else 0)
        root_xy = ctx.get("root_xy")
        if not (isinstance(root_xy, tuple) and len(root_xy) == 2):
            _axionite_debug_log_throttled(
                ctx,
                rnd,
                uid,
                cur_xy,
                "economy_axionite_core_root_missing",
                interval=20,
                phase="axionite_pick_core_route",
            )
            state.phase = "axionite_pick_root"
            return

        root_xy = (int(root_xy[0]), int(root_xy[1]))

        # plan_start_xy = the source of the NEW link segment.
        # * On a fresh entry from axionite_build_root_bridge, plan from root_xy
        #   (we're building the whole chain).
        # * After a collision in axionite_build_core_route, plan from cur_xy so
        #   the partial chain from root_xy to cur_xy stays intact and the new
        #   segment extends forward from where the bot is standing — mirrors
        #   the main titanium pipeline's behavior on collisions.
        plan_start_xy = cur_xy if replan_from_cur else root_xy

        blocked_core = ctx.get("blocked_core_link_candidates")
        if not isinstance(blocked_core, set):
            blocked_core = set()
            ctx["blocked_core_link_candidates"] = blocked_core

        excluded_tiles = _axionite_core_excluded_tiles(ctx, root_xy)
        excluded_tuple = tuple(sorted(excluded_tiles))

        highway_candidate = None
        highway_target = _select_bridge_escape_target_v2(
            state,
            local_map,
            plan_start_xy,
            state.core_xy,
            exclude_positions=excluded_tuple,
        )
        if isinstance(highway_target, tuple) and len(highway_target) == 2:
            highway_candidate = {
                "mode": "highway_direct",
                "bridge_pos": plan_start_xy,
                "bridge_target": (int(highway_target[0]), int(highway_target[1])),
                "source_network_key": None,
                "source_conveyor": None,
            }
            highway_key = _axionite_core_candidate_key(highway_candidate)
            if highway_key in blocked_core:
                highway_candidate = None

        log_event(
            rnd,
            uid,
            "economy",
            f"({cur_xy[0]},{cur_xy[1]})",
            "economy_axionite_network_highway_target_debug_v2",
            available=1 if highway_candidate is not None else 0,
            root=f"({root_xy[0]},{root_xy[1]})",
            start=f"({plan_start_xy[0]},{plan_start_xy[1]})",
            tx=(
                int(highway_target[0])
                if isinstance(highway_target, tuple) and len(highway_target) == 2
                else -1
            ),
            ty=(
                int(highway_target[1])
                if isinstance(highway_target, tuple) and len(highway_target) == 2
                else -1
            ),
            excluded=len(excluded_tiles),
        )

        direct_candidates = _axionite_core_direct_candidates(
            state,
            plan_start_xy,
            excluded_tiles,
        )
        preview_items = [
            f"({cnd['bridge_pos'][0]},{cnd['bridge_pos'][1]}):d{_manhattan(plan_start_xy, cnd['bridge_pos'])}"
            for cnd in direct_candidates[:12]
        ]
        if len(direct_candidates) > 12:
            preview_items.append(f"...(+{len(direct_candidates) - 12})")
        log_event(
            rnd,
            uid,
            "economy",
            f"({cur_xy[0]},{cur_xy[1]})",
            "economy_axionite_network_direct_targets_debug_v2",
            count=len(direct_candidates),
            blocked=len(blocked_core),
            targets=",".join(preview_items),
        )

        chosen_candidate = None
        chosen_nodes = ()

        if highway_candidate is not None:
            chosen_candidate = highway_candidate
            chosen_nodes = (plan_start_xy,)

        if chosen_candidate is None:
            for cnd in direct_candidates:
                cnd_key = _axionite_core_candidate_key(cnd)
                if cnd_key in blocked_core:
                    continue

                # Concern 1 fix: block friendly infra so the planner cannot
                # route through (and cause the build sequence to overwrite)
                # existing friendly conveyors/bridges. When no direct path
                # through truly-empty tiles exists, the highway splice path
                # above is correctly preferred.
                cnd_nodes = _plan_axionite_link_nodes(
                    local_map,
                    plan_start_xy,
                    cnd["bridge_pos"],
                    blocked_tiles=excluded_tiles,
                    block_friendly_infra=True,
                )
                if cnd_nodes:
                    chosen_candidate = cnd
                    chosen_nodes = cnd_nodes
                    break

                if cnd_key is not None:
                    blocked_core.add(cnd_key)

        if chosen_candidate is None:
            lidar_candidate = _select_lidar_bridge_candidate(
                state,
                local_map,
                plan_start_xy,
            )
            if isinstance(lidar_candidate, dict):
                cnd_key = _axionite_core_candidate_key(lidar_candidate)
                bridge_pos = lidar_candidate.get("bridge_pos")
                bridge_target = lidar_candidate.get("bridge_target")
                if (
                    cnd_key not in blocked_core
                    and isinstance(bridge_pos, tuple)
                    and len(bridge_pos) == 2
                    and isinstance(bridge_target, tuple)
                    and len(bridge_target) == 2
                    and (int(bridge_pos[0]), int(bridge_pos[1])) not in excluded_tiles
                    and (int(bridge_target[0]), int(bridge_target[1])) not in excluded_tiles
                ):
                    cnd_nodes = _plan_axionite_link_nodes(
                        local_map,
                        plan_start_xy,
                        (int(bridge_pos[0]), int(bridge_pos[1])),
                        blocked_tiles=excluded_tiles,
                        block_friendly_infra=True,
                    )
                    if cnd_nodes:
                        chosen_candidate = {
                            "mode": str(lidar_candidate.get("mode") or "lidar_indirect"),
                            "bridge_pos": (int(bridge_pos[0]), int(bridge_pos[1])),
                            "bridge_target": (
                                int(bridge_target[0]),
                                int(bridge_target[1]),
                            ),
                            "source_network_key": lidar_candidate.get("source_network_key"),
                            "source_conveyor": lidar_candidate.get("source_conveyor"),
                        }
                        chosen_nodes = cnd_nodes
                    elif cnd_key is not None:
                        blocked_core.add(cnd_key)

        if chosen_candidate is None:
            no_cnd_streak = int(ctx.get("core_no_candidate_streak", 0)) + 1
            ctx["core_no_candidate_streak"] = no_cnd_streak
            _axionite_debug_log_throttled(
                ctx,
                rnd,
                uid,
                cur_xy,
                "economy_axionite_network_no_candidate_v2",
                interval=15,
                root=f"({root_xy[0]},{root_xy[1]})",
                direct=len(direct_candidates),
                blocked=len(blocked_core),
                excluded=len(excluded_tiles),
                streak=no_cnd_streak,
            )

            if no_cnd_streak >= int(_AXIONITE_CORE_NO_CANDIDATE_LIMIT):
                log_event(
                    rnd,
                    uid,
                    "economy",
                    f"({cur_xy[0]},{cur_xy[1]})",
                    "economy_axionite_core_route_skipped_no_candidate",
                    ore=(
                        f"({ore_xy[0]},{ore_xy[1]})"
                        if isinstance(ore_xy, tuple)
                        else "none"
                    ),
                    foundry=(
                        f"({foundry_xy[0]},{foundry_xy[1]})"
                        if isinstance(foundry_xy, tuple)
                        else "none"
                    ),
                    rounds=no_cnd_streak,
                )
                _reset_axionite_pipeline_for_next_ore(state, ctx)
            return

        ctx["core_no_candidate_streak"] = 0

        bridge_pos = chosen_candidate["bridge_pos"]
        bridge_target = chosen_candidate["bridge_target"]
        mode = str(chosen_candidate.get("mode") or "unknown")

        link_seq = _compose_axionite_link_sequence(
            chosen_nodes,
            bridge_target,
            build_order="forward",
            terminal_mode="bridge",
        )
        if not link_seq:
            cnd_key = _axionite_core_candidate_key(chosen_candidate)
            if cnd_key is not None:
                blocked_core.add(cnd_key)
            return
        ctx["core_route_candidate"] = chosen_candidate
        ctx["link_nodes"] = chosen_nodes
        ctx["link_terminal_xy"] = bridge_target
        ctx["link_sequence"] = link_seq
        ctx["link_sequence_index"] = 0
        log_event(
            rnd,
            uid,
            "economy",
            f"({cur_xy[0]},{cur_xy[1]})",
            "economy_axionite_network_target_selected_v2",
            mode=mode,
            bx=bridge_pos[0],
            by=bridge_pos[1],
            tx=bridge_target[0],
            ty=bridge_target[1],
        )
        log_event(
            rnd,
            uid,
            "economy",
            f"({cur_xy[0]},{cur_xy[1]})",
            "economy_axionite_network_plan_ready_v2",
            bx=bridge_pos[0],
            by=bridge_pos[1],
            tx=bridge_target[0],
            ty=bridge_target[1],
            steps=max(0, len(chosen_nodes) - 1),
            mode=mode,
        )
        log_event(
            rnd,
            uid,
            "economy",
            f"({cur_xy[0]},{cur_xy[1]})",
            "economy_axionite_network_plan_dump_v2",
            start=f"({plan_start_xy[0]},{plan_start_xy[1]})",
            goal=f"({bridge_pos[0]},{bridge_pos[1]})",
            mode=mode,
            path=_format_plan_dump(plan_start_xy, chosen_nodes[1:]),
        )
        state.phase = "axionite_build_core_route"
        return

    if state.phase == "axionite_build_core_route":
        if not COMPETITION_MODE:
            candidate = ctx.get("core_route_candidate")
            log_event(rnd, uid, "economy", f"({cur_xy[0]},{cur_xy[1]})",
                      "economy_axionite_build_core_route_entry",
                      seq_idx=int(ctx.get("link_sequence_index", 0)),
                      seq_len=len(tuple(ctx.get("link_sequence", ()))),
                      mode=str(candidate.get("mode") or "none") if isinstance(candidate, dict) else "none")
        seq_result = _run_axionite_link_build_sequence(
            c,
            state,
            local_map,
            cur_xy,
            rnd,
            uid,
            ctx,
            "axionite_build_core_route",
        )
        if seq_result == "done":
            _done_candidate = ctx.get("core_route_candidate")
            if isinstance(_done_candidate, dict):
                _bpos = _done_candidate.get("bridge_pos")
                _btgt = _done_candidate.get("bridge_target")
                _bpos_ok = isinstance(_bpos, tuple) and len(_bpos) == 2
                _btgt_ok = isinstance(_btgt, tuple) and len(_btgt) == 2
                if _bpos_ok and _btgt_ok and isinstance(ore_xy, tuple) and len(ore_xy) == 2:
                    state.active_network_id = _ensure_active_network_id_for_harvester(
                        state, ore_xy, rnd)
                    _record_active_network_terminal_bridge(state, _bpos, _btgt, rnd)
            ctx["core_route_candidate"] = None
            log_event(
                rnd,
                uid,
                "economy",
                f"({cur_xy[0]},{cur_xy[1]})",
                "economy_axionite_pipeline_complete",
                ore=(f"({ore_xy[0]},{ore_xy[1]})" if isinstance(
                    ore_xy, tuple) else "none"),
                foundry=(
                    f"({foundry_xy[0]},{foundry_xy[1]})"
                    if isinstance(foundry_xy, tuple)
                    else "none"
                ),
            )

            # Reset per-ore context and continue with the next axionite ore.
            _reset_axionite_pipeline_for_next_ore(state, ctx)
            return
        if seq_result == "failed":
            candidate = ctx.get("core_route_candidate")
            blocked_core = ctx.get("blocked_core_link_candidates")
            candidate_key = _axionite_core_candidate_key(candidate)
            if isinstance(blocked_core, set) and candidate_key is not None:
                blocked_core.add(candidate_key)

            _axionite_debug_log_throttled(
                ctx,
                rnd,
                uid,
                cur_xy,
                "economy_axionite_core_route_failed",
                interval=10,
                mode=(
                    str(candidate.get("mode") or "none")
                    if isinstance(candidate, dict)
                    else "none"
                ),
                seq_idx=int(ctx.get("link_sequence_index", 0)),
                seq_len=len(tuple(ctx.get("link_sequence", ()))),
            )
            state.phase = "axionite_pick_core_route"
            ctx["core_route_candidate"] = None
            ctx["link_sequence"] = ()
            ctx["link_sequence_index"] = 0
            # Flag the next pick_core_route entry to replan from cur_xy so
            # the partial chain from root_xy through cur_xy stays intact
            # and the new segment extends forward (mirrors main titanium
            # pipeline's collision-handling behaviour).
            ctx["core_route_replan_from_cur"] = True
        return

    if state.phase == "axionite_done":
        if not COMPETITION_MODE:
            log_event(rnd, uid, "economy", f"({cur_xy[0]},{cur_xy[1]})",
                      "economy_axionite_done_resume")
        _resume_exploration_after_harvest(state)
        return


# ===== AXIONITE SABOTAGE =====
# Phases: axionite_sabotage_pick → axionite_sabotage_pick_goal →
#         axionite_sabotage_plan_goal → axionite_sabotage_follow_goal →
#         axionite_sabotage_attack_start → axionite_sabotage_finalize

def _run_axionite_sabotage(
    c: Controller,
    state: EconomyState,
    local_map,
    cur_xy,
    rnd: int,
    uid: int,
):
    _ = c
    ctx = _ensure_axionite_ctx(state)

    if state.phase not in _SABOTAGE_AXIONITE_PHASES:
        state.phase = "axionite_sabotage_pick"
        state.plan_steps = ()
        state.plan_index = 0
        state.defer_step_once = False
        ctx["sabotage_ore_xy"] = None
        ctx["sabotage_goal_xy"] = None
        return

    if state.phase == "axionite_sabotage_pick":
        if not COMPETITION_MODE:
            log_event(rnd, uid, "economy", f"({cur_xy[0]},{cur_xy[1]})",
                      "economy_axionite_sabotage_pick_entry")
        targets = _known_enemy_axionite_harvesters(
            local_map,
            state=state,
            core_xy=state.core_xy,
            min_core_cheb=_ORE_CORE_EXCLUSION_MIN_CHEB,
        )
        if not targets:
            if not COMPETITION_MODE:
                log_event(rnd, uid, "economy", f"({cur_xy[0]},{cur_xy[1]})",
                          "economy_axionite_sabotage_pick_no_targets")
            _resume_exploration_after_harvest(state)
            return
        _, ore_xy = min(
            targets,
            key=lambda item: (
                _manhattan(cur_xy, item[1]),
                item[1][0],
                item[1][1],
            ),
        )
        ctx["sabotage_ore_xy"] = (int(ore_xy[0]), int(ore_xy[1]))
        ctx["sabotage_goal_xy"] = None
        if not COMPETITION_MODE:
            log_event(rnd, uid, "economy", f"({cur_xy[0]},{cur_xy[1]})",
                      "economy_axionite_sabotage_pick_selected",
                      ox=int(ore_xy[0]), oy=int(ore_xy[1]),
                      targets=len(targets))
        state.phase = "axionite_sabotage_pick_goal"
        return

    ore_xy = ctx.get("sabotage_ore_xy")
    if not (isinstance(ore_xy, tuple) and len(ore_xy) == 2):
        state.phase = "axionite_sabotage_pick"
        return
    ore_xy = (int(ore_xy[0]), int(ore_xy[1]))

    if state.phase == "axionite_sabotage_pick_goal":
        if not COMPETITION_MODE:
            log_event(rnd, uid, "economy", f"({cur_xy[0]},{cur_xy[1]})",
                      "economy_axionite_sabotage_pick_goal_entry",
                      ox=ore_xy[0], oy=ore_xy[1])
        goal_xy = _pick_enemy_harvester_takeover_goal(
            local_map, ore_xy, cur_xy)
        if not (isinstance(goal_xy, tuple) and len(goal_xy) == 2):
            if not COMPETITION_MODE:
                log_event(rnd, uid, "economy", f"({cur_xy[0]},{cur_xy[1]})",
                          "economy_axionite_sabotage_pick_goal_no_stand",
                          ox=ore_xy[0], oy=ore_xy[1])
            _resume_exploration_after_harvest(state)
            return
        ctx["sabotage_goal_xy"] = goal_xy
        state.plan_steps = ()
        state.plan_index = 0
        state.defer_step_once = False
        if not COMPETITION_MODE:
            log_event(rnd, uid, "economy", f"({cur_xy[0]},{cur_xy[1]})",
                      "economy_axionite_sabotage_pick_goal_selected",
                      ox=ore_xy[0], oy=ore_xy[1],
                      gx=int(goal_xy[0]), gy=int(goal_xy[1]))
        state.phase = "axionite_sabotage_plan_goal"
        return

    goal_xy = ctx.get("sabotage_goal_xy")
    if not (isinstance(goal_xy, tuple) and len(goal_xy) == 2):
        state.phase = "axionite_sabotage_pick_goal"
        return
    goal_xy = (int(goal_xy[0]), int(goal_xy[1]))

    if state.phase == "axionite_sabotage_plan_goal":
        if not COMPETITION_MODE:
            log_event(rnd, uid, "economy", f"({cur_xy[0]},{cur_xy[1]})",
                      "economy_axionite_sabotage_plan_goal_entry",
                      ox=ore_xy[0], oy=ore_xy[1],
                      gx=goal_xy[0], gy=goal_xy[1])
        if cur_xy == goal_xy:
            state.phase = "axionite_sabotage_attack_start"
            return
        steps = _astar_cardinal_plan(
            local_map,
            cur_xy,
            goal_xy,
            max_expansions=640,
            tile_passable_fn=lambda x, y: _is_general_movement_passable(
                local_map,
                x,
                y,
                goal_xy,
            ),
        )
        if not steps:
            if not COMPETITION_MODE:
                log_event(rnd, uid, "economy", f"({cur_xy[0]},{cur_xy[1]})",
                          "economy_axionite_sabotage_plan_goal_no_path",
                          ox=ore_xy[0], oy=ore_xy[1],
                          gx=goal_xy[0], gy=goal_xy[1])
            _resume_exploration_after_harvest(state)
            return
        state.plan_steps = steps
        state.plan_index = 0
        state.defer_step_once = True
        if not COMPETITION_MODE:
            log_event(rnd, uid, "economy", f"({cur_xy[0]},{cur_xy[1]})",
                      "economy_axionite_sabotage_plan_goal_path_found",
                      steps=len(steps), gx=goal_xy[0], gy=goal_xy[1])
        state.phase = "axionite_sabotage_follow_goal"
        return

    if state.phase == "axionite_sabotage_follow_goal":
        if not COMPETITION_MODE:
            log_event(rnd, uid, "economy", f"({cur_xy[0]},{cur_xy[1]})",
                      "economy_axionite_sabotage_follow_goal_entry",
                      gx=goal_xy[0], gy=goal_xy[1],
                      step=state.plan_index, total=len(state.plan_steps))
        if cur_xy == goal_xy:
            state.phase = "axionite_sabotage_attack_start"
            state.plan_steps = ()
            state.plan_index = 0
            state.defer_step_once = False
            return

        if state.defer_step_once:
            state.defer_step_once = False
            return
        if state.plan_index >= len(state.plan_steps):
            state.phase = "axionite_sabotage_plan_goal"
            return

        nxt = state.plan_steps[state.plan_index]
        if not _is_adjacent_step(cur_xy, nxt):
            state.phase = "axionite_sabotage_plan_goal"
            return

        move_result = _execute_step_toward(c, local_map, cur_xy, nxt, rnd, uid)
        if move_result == "moved":
            state.issued_move_last_tick = True
            state.expected_xy_after_move = nxt
            state.plan_index += 1
            return
        if move_result in ("built", "wait_cd"):
            return
        state.phase = "axionite_sabotage_plan_goal"
        return

    if state.phase == "axionite_sabotage_attack_start":
        if not COMPETITION_MODE:
            log_event(rnd, uid, "economy", f"({cur_xy[0]},{cur_xy[1]})",
                      "economy_axionite_sabotage_attack_start_entry",
                      ox=ore_xy[0], oy=ore_xy[1])
        if _manhattan(cur_xy, ore_xy) != 1:
            if not COMPETITION_MODE:
                log_event(rnd, uid, "economy", f"({cur_xy[0]},{cur_xy[1]})",
                          "economy_axionite_sabotage_attack_not_adjacent",
                          ox=ore_xy[0], oy=ore_xy[1])
            state.phase = "axionite_sabotage_pick_goal"
            return
        targets = _adjacent_enemy_conveyor_targets(local_map, ore_xy)
        if targets:
            if not COMPETITION_MODE:
                log_event(rnd, uid, "economy", f"({cur_xy[0]},{cur_xy[1]})",
                          "economy_axionite_sabotage_attack_targets_found",
                          count=len(targets), ox=ore_xy[0], oy=ore_xy[1])
            _start_network_attack(
                state,
                targets[0],
                "axionite_sabotage_finalize",
                "enemy_axionite_sabotage",
                target_queue=targets,
                return_xy=cur_xy,
            )
            return
        if not COMPETITION_MODE:
            log_event(rnd, uid, "economy", f"({cur_xy[0]},{cur_xy[1]})",
                      "economy_axionite_sabotage_attack_no_conveyors",
                      ox=ore_xy[0], oy=ore_xy[1])
        state.phase = "axionite_sabotage_finalize"
        return

    if state.phase == "axionite_sabotage_finalize":
        if not COMPETITION_MODE:
            log_event(rnd, uid, "economy", f"({cur_xy[0]},{cur_xy[1]})",
                      "economy_axionite_sabotage_finalize",
                      ox=ore_xy[0], oy=ore_xy[1])
        ctx["sabotage_ore_xy"] = None
        ctx["sabotage_goal_xy"] = None
        _resume_exploration_after_harvest(state)
        return


# ===== CONVEYOR REPAIR =====
# Phases: repair_pick_target → repair_plan_harvester → repair_follow_harvester →
#         repair_attack_adjacent → repair_attack_path →
#         repair_rebuild_plan → repair_rebuild_follow → repair_rebuild_build

def _normalise_conveyor_memory_entries(entries):
    out = []
    if not isinstance(entries, (tuple, list)):
        return ()

    for entry in entries:
        if not isinstance(entry, (tuple, list)):
            continue
        if len(entry) < 3:
            continue

        try:
            x = int(entry[0])
            y = int(entry[1])
        except (TypeError, ValueError):
            continue

        raw_dir = entry[2]
        if isinstance(raw_dir, Direction):
            direction_name = raw_dir.name
        elif isinstance(raw_dir, str):
            direction_name = raw_dir
        else:
            continue

        if direction_name not in _CARDINAL_DIRECTION_BY_NAME:
            continue

        expected_id = -1
        if len(entry) >= 4 and isinstance(entry[3], int):
            expected_id = int(entry[3])

        out.append((x, y, direction_name, expected_id))

    return tuple(out)


def _visible_friendly_conveyor_match(
    local_map,
    tile_xy,
    direction_name: str | None = None,
    require_direction_match: bool = False,
):
    tx = int(tile_xy[0])
    ty = int(tile_xy[1])

    if not local_map.in_bounds(tx, ty):
        return False, None, None, False
    if not _tile_is_known(local_map, tx, ty):
        return False, None, None, False

    try:
        if not local_map.is_visible(tx, ty):
            return False, None, None, False
    except (AttributeError, TypeError, GameError):
        return False, None, None, False

    rec = _known_building_at(local_map, tx, ty)
    if not isinstance(rec, dict):
        return False, None, None, True
    if rec.get("team") != getattr(local_map, "my_team", None):
        return False, None, None, True
    if rec.get("entity_type") != EntityType.CONVEYOR:
        return False, None, None, True

    direction = rec.get("direction")
    seen_dir_name = direction.name if direction is not None else None
    seen_id = _entity_id_from_rec(rec)
    if require_direction_match and isinstance(direction_name, str):
        if seen_dir_name != direction_name:
            return False, seen_id, seen_dir_name, True

    return True, seen_id, seen_dir_name, True


def _enemy_repair_blocker_kind_from_map(local_map, tile_xy) -> str | None:
    tx = int(tile_xy[0])
    ty = int(tile_xy[1])
    rec = _known_building_at(local_map, tx, ty)
    if not isinstance(rec, dict):
        return None
    if rec.get("team") == getattr(local_map, "my_team", None):
        return None

    etype = rec.get("entity_type")
    if etype == EntityType.ARMOURED_CONVEYOR:
        return "armoured"
    if etype == EntityType.ROAD:
        return "road"
    if etype == EntityType.CONVEYOR:
        return "conveyor"
    if etype == EntityType.SPLITTER:
        return "splitter"
    if etype == EntityType.BRIDGE:
        return "bridge"
    return None


def _enemy_repair_blocker_kind_from_controller(c: Controller, tile_xy) -> str | None:
    tp = Position(int(tile_xy[0]), int(tile_xy[1]))
    try:
        tile_id = c.get_tile_building_id(tp)
    except GameError:
        return None

    if not isinstance(tile_id, int):
        return None

    try:
        if c.get_team(tile_id) == c.get_team():
            return None
        etype = c.get_entity_type(tile_id)
    except GameError:
        return None

    if etype == EntityType.ARMOURED_CONVEYOR:
        return "armoured"
    if etype == EntityType.ROAD:
        return "road"
    if etype == EntityType.CONVEYOR:
        return "conveyor"
    if etype == EntityType.SPLITTER:
        return "splitter"
    if etype == EntityType.BRIDGE:
        return "bridge"
    return None


def _format_repair_pending_dump(pending_map, max_items: int = 20) -> str:
    if not isinstance(pending_map, dict) or not pending_map:
        return ""

    items = []
    for tile_xy, entry in pending_map.items():
        if not (isinstance(tile_xy, tuple) and len(tile_xy) == 2):
            continue
        if not isinstance(entry, dict):
            continue

        tx = int(tile_xy[0])
        ty = int(tile_xy[1])
        direction_name = entry.get("direction")
        if not isinstance(direction_name, str):
            direction_name = "?"
        network_id = entry.get("network_id")
        expected_id = entry.get("expected_id")
        fails = entry.get("fails", 0)

        items.append(
            (
                tx,
                ty,
                direction_name,
                int(network_id) if isinstance(network_id, int) else -1,
                int(expected_id) if isinstance(expected_id, int) else -1,
                max(0, int(fails)),
            )
        )

    if not items:
        return ""

    items.sort(key=lambda item: (item[0], item[1], item[2]))
    shown = items[:max_items]
    out = ",".join(
        (
            f"({tx},{ty},{direction_name},nid={network_id},id={expected_id},f={fails})"
        )
        for tx, ty, direction_name, network_id, expected_id, fails in shown
    )
    if len(items) > max_items:
        out += f",...(+{len(items) - max_items})"
    return out


def _repair_harvester_build_goal(local_map, harvester_xy, cur_xy):
    hx = int(harvester_xy[0])
    hy = int(harvester_xy[1])

    def _collect_candidates(respect_halo: bool):
        out = []
        for dx, dy in CARDINAL_DELTAS:
            gx = hx + dx
            gy = hy + dy
            if not local_map.in_bounds(gx, gy):
                continue
            if not _is_general_movement_passable(
                local_map,
                gx,
                gy,
                (gx, gy),
                respect_halo=respect_halo,
            ):
                continue
            out.append((_manhattan(cur_xy, (gx, gy)), gx, gy))
        return out

    candidates = _collect_candidates(True)
    if not candidates:
        candidates = _collect_candidates(False)

    if not candidates:
        return None

    _, gx, gy = min(candidates)
    return (gx, gy)
def _repair_expected_adjacent_conveyor_tiles(rec, harvester_xy):
    hx = int(harvester_xy[0])
    hy = int(harvester_xy[1])

    out = []
    seen = set()
    for x, y, _direction_name, _expected_id in _normalise_conveyor_memory_entries(
        rec.get("conveyor_memory", ())
    ):
        if abs(int(x) - hx) + abs(int(y) - hy) != 1:
            continue
        txy = (int(x), int(y))
        if txy in seen:
            continue
        seen.add(txy)
        out.append(txy)

    out.sort(key=lambda p: (p[0], p[1]))
    return tuple(out)


def _adjacent_enemy_repair_blocker_targets(local_map, center_xy):
    cx, cy = int(center_xy[0]), int(center_xy[1])
    my_team = getattr(local_map, "my_team", None)
    out = []
    for dx, dy in _ADJACENT_DELTAS_8:
        tx = cx + dx
        ty = cy + dy
        if not local_map.in_bounds(tx, ty):
            continue
        rec = _known_building_at(local_map, tx, ty)
        if not isinstance(rec, dict):
            continue
        if rec.get("team") == my_team:
            continue
        if rec.get("entity_type") not in _REPLACEABLE_ENEMY_BLOCKERS:
            continue
        out.append((tx, ty))

    out.sort(key=lambda p: (_manhattan((cx, cy), p), p[0], p[1]))
    return tuple(out)


def _repair_build_conveyor_memory_index(rec):
    memory = _normalise_conveyor_memory_entries(rec.get("conveyor_memory", ()))
    if not memory:
        return {}
    out = {}
    for x, y, direction_name, expected_id in memory:
        out[(int(x), int(y))] = (
            direction_name,
            int(expected_id) if isinstance(expected_id, int) else -1,
        )
    return out


def _repair_collect_bridge_to_harvester_chain(rec):
    chain = []
    for p in rec.get("transport_chain_positions", ()):
        if isinstance(p, tuple) and len(p) == 2:
            chain.append((int(p[0]), int(p[1])))

    if not chain:
        return ()

    terminal = rec.get("terminal_bridge_pos")
    if isinstance(terminal, tuple) and len(terminal) == 2:
        terminal_xy = (int(terminal[0]), int(terminal[1]))
        if terminal_xy in chain:
            idx = chain.index(terminal_xy)
            return tuple(chain[:idx + 1])

    return tuple(chain)


def _repair_collect_enemy_blockers_along_chain(local_map, chain_xy):
    my_team = getattr(local_map, "my_team", None)
    targets = []
    for tx, ty in chain_xy:
        if not local_map.in_bounds(tx, ty):
            continue
        if not _tile_is_known(local_map, tx, ty):
            continue
        rec = _known_building_at(local_map, tx, ty)
        if not isinstance(rec, dict):
            continue
        if rec.get("team") == my_team:
            continue
        if rec.get("entity_type") in _REPLACEABLE_ENEMY_BLOCKERS:
            targets.append((tx, ty))

    deduped = []
    seen = set()
    for txy in targets:
        if txy in seen:
            continue
        seen.add(txy)
        deduped.append(txy)
    return tuple(deduped)
def _rebuild_broken_transport_tile_cache(state: EconomyState):
    tiles_by_id = getattr(state, "broken_network_tiles_by_id", None)
    if not isinstance(tiles_by_id, dict) or not tiles_by_id:
        state.broken_transport_tiles = set()
        return

    out = set()
    for tiles in tiles_by_id.values():
        if not isinstance(tiles, (tuple, list, set)):
            continue
        for p in tiles:
            if isinstance(p, tuple) and len(p) == 2:
                out.add((int(p[0]), int(p[1])))
    state.broken_transport_tiles = out


def _clear_broken_network_tracking_for_network(
    state: EconomyState,
    network_id: int,
):
    if not isinstance(network_id, int):
        return

    state.broken_network_ids.discard(network_id)
    tiles_by_id = getattr(state, "broken_network_tiles_by_id", None)
    if isinstance(tiles_by_id, dict):
        tiles_by_id.pop(network_id, None)
    _rebuild_broken_transport_tile_cache(state)


def _remove_tile_from_broken_tracking(state: EconomyState, tile_xy) -> bool:
    tx = int(tile_xy[0])
    ty = int(tile_xy[1])
    target = (tx, ty)

    changed = False
    tiles_by_id = getattr(state, "broken_network_tiles_by_id", None)
    if isinstance(tiles_by_id, dict):
        for network_id, tiles in tuple(tiles_by_id.items()):
            if not isinstance(network_id, int):
                continue
            if not isinstance(tiles, (tuple, list, set)):
                continue

            tile_set = {
                (int(p[0]), int(p[1]))
                for p in tiles
                if isinstance(p, tuple) and len(p) == 2
            }
            if target not in tile_set:
                continue

            tile_set.discard(target)
            changed = True
            if tile_set:
                tiles_by_id[network_id] = tuple(sorted(tile_set))
            else:
                tiles_by_id.pop(network_id, None)

    if changed:
        _rebuild_broken_transport_tile_cache(state)
    return changed


def _is_friendly_broken_transport_tile(state: EconomyState, local_map, tile_xy) -> bool:
    tx = int(tile_xy[0])
    ty = int(tile_xy[1])

    rec = _known_building_at(local_map, tx, ty)
    if not isinstance(rec, dict):
        return False
    if rec.get("team") != getattr(local_map, "my_team", None):
        return False
    if rec.get("entity_type") not in _TRANSPORT_ENTITY_TYPES:
        return False

    broken_tiles = getattr(state, "broken_transport_tiles", None)
    if not isinstance(broken_tiles, set):
        return False
    return (tx, ty) in broken_tiles


def _destroy_friendly_broken_transport_blocker(
    c: Controller,
    state: EconomyState,
    local_map,
    tile_xy,
    rnd: int,
    uid: int,
    source_phase: str,
) -> bool:
    tx = int(tile_xy[0])
    ty = int(tile_xy[1])
    target_xy = (tx, ty)

    if not _is_friendly_broken_transport_tile(state, local_map, target_xy):
        return False

    # Friendly bridges are valid logistics sinks. Do not destroy them as
    # "broken blockers"; let bridge-escape logic terminate into them instead.
    rec = _known_building_at(local_map, tx, ty)
    if isinstance(rec, dict) and rec.get("entity_type") == EntityType.BRIDGE:
        return False

    tp = Position(tx, ty)
    try:
        if not c.can_destroy(tp):
            return False
    except GameError:
        return False

    try:
        c.destroy(tp)
    except GameError:
        return False

    _remove_tile_from_broken_tracking(state, target_xy)
    log_event(
        rnd,
        uid,
        "economy",
        f"({tx},{ty})",
        "economy_network_broken_friendly_blocker_destroyed_v2",
        phase=source_phase,
    )
    return True


def _destroy_standing_transport_for_highway_switch(
    c: Controller,
    local_map,
    cur_xy,
    rnd: int,
    uid: int,
    source_phase: str,
) -> bool:
    tx = int(cur_xy[0])
    ty = int(cur_xy[1])

    rec = _known_building_at(local_map, tx, ty)
    if not isinstance(rec, dict):
        return True
    if rec.get("team") != getattr(local_map, "my_team", None):
        return True

    etype = rec.get("entity_type")
    if etype not in (
        EntityType.CONVEYOR,
        EntityType.SPLITTER,
        EntityType.ARMOURED_CONVEYOR,
        EntityType.BRIDGE,
    ):
        return True

    tp = Position(tx, ty)
    try:
        if not c.can_destroy(tp):
            return False
    except GameError:
        return False

    try:
        c.destroy(tp)
    except GameError:
        return False

    etype_name = str(getattr(etype, "name", etype))
    log_event(
        rnd,
        uid,
        "economy",
        f"({tx},{ty})",
        "economy_network_switch_underfoot_destroyed_v2",
        phase=source_phase,
        kind=etype_name,
    )
    return True


def _mark_network_broken_for_reacquire(
    state: EconomyState,
    network_id: int,
    rnd: int,
    preserve_memory: bool = False,
):
    rec = state.network_records.get(network_id)
    if not isinstance(rec, dict):
        return ()

    broken_tiles = {
        (int(p[0]), int(p[1]))
        for p in rec.get("transport_chain_positions", ())
        if isinstance(p, tuple) and len(p) == 2
    }
    terminal = rec.get("terminal_bridge_pos")
    if isinstance(terminal, tuple) and len(terminal) == 2:
        broken_tiles.add((int(terminal[0]), int(terminal[1])))

    state.broken_network_ids.add(network_id)

    existing_tiles = {
        (int(p[0]), int(p[1]))
        for p in state.broken_network_tiles_by_id.get(network_id, ())
        if isinstance(p, tuple) and len(p) == 2
    }
    existing_tiles.update(broken_tiles)
    if existing_tiles:
        state.broken_network_tiles_by_id[network_id] = tuple(
            sorted(existing_tiles)
        )
    else:
        state.broken_network_tiles_by_id.pop(network_id, None)
    _rebuild_broken_transport_tile_cache(state)

    old_terminal = rec.get("terminal_bridge_pos")
    harvester_pos = rec.get("harvester_pos")

    if not preserve_memory:
        rec["transport_chain_positions"] = ()
        rec["conveyor_memory"] = ()
        rec["terminal_bridge_pos"] = None
        rec["bridge_target_pos"] = None
    rec["complete"] = False
    rec["built_by_me"] = False
    rec["last_seen_round"] = rnd

    state.network_records[network_id] = rec

    if isinstance(old_terminal, tuple) and len(old_terminal) == 2:
        old_terminal = (int(old_terminal[0]), int(old_terminal[1]))
        if state.network_id_by_terminal.get(old_terminal) == network_id:
            state.network_id_by_terminal.pop(old_terminal, None)

    if isinstance(harvester_pos, tuple) and len(harvester_pos) == 2:
        state.network_id_by_harvester[(
            int(harvester_pos[0]),
            int(harvester_pos[1]),
        )] = network_id

    state.provisional_network_ids.add(network_id)
    return tuple(sorted(existing_tiles))


def _repair_handoff_to_takeover_rebuild(
    state: EconomyState,
    target_xy,
    network_id: int,
    rnd: int,
    uid: int,
    cur_xy,
):
    hx = int(target_xy[0])
    hy = int(target_xy[1])
    broken_tiles = _mark_network_broken_for_reacquire(
        state,
        network_id,
        rnd,
        preserve_memory=True,
    )
    cooldown_until = int(rnd) + _REPAIR_HANDOFF_REENQUEUE_COOLDOWN_ROUNDS
    state.repair_handoff_cooldown_until[(hx, hy)] = cooldown_until

    rec = state.network_records.get(network_id)
    can_memory_rebuild = False
    if isinstance(rec, dict):
        chain = _repair_collect_bridge_to_harvester_chain(rec)
        memory_index = _repair_build_conveyor_memory_index(rec)
        if chain and memory_index:
            can_memory_rebuild = any(
                abs(int(tx) - hx) + abs(int(ty) - hy) == 1
                for tx, ty in memory_index.keys()
            )

    if can_memory_rebuild:
        state.repair_target_xy = (hx, hy)
        state.repair_target_direction_name = "HARVESTER"
        state.repair_target_network_id = int(network_id)
        state.repair_target_expected_id = -1
        state.repair_harvester_goal_xy = None
        state.repair_rebuild_sequence = ()
        state.repair_rebuild_index = 0
        state.repair_rebuild_avoid_tiles = set()
        state.plan_steps = ()
        state.plan_index = 0
        state.defer_step_once = False
        state.phase = "repair_rebuild_plan"

        log_event(
            rnd,
            uid,
            "economy",
            f"({cur_xy[0]},{cur_xy[1]})",
            "economy_repair_handoff_takeover_rebuild",
            hx=hx,
            hy=hy,
            nid=network_id,
            broken_tiles=len(broken_tiles),
            pending=len(state.repair_pending_conveyors),
            cooldown_until=cooldown_until,
            mode="memory_rebuild",
        )
        return

    broken_tiles = _mark_network_broken_for_reacquire(
        state,
        network_id,
        rnd,
        preserve_memory=False,
    )

    _drop_repair_target(state, (hx, hy))

    state.repair_resume_phase = None
    state.repair_target_xy = None
    state.repair_target_direction_name = None
    state.repair_target_network_id = None
    state.repair_target_expected_id = None
    state.repair_harvester_goal_xy = None
    state.repair_rebuild_sequence = ()
    state.repair_rebuild_index = 0
    state.repair_rebuild_avoid_tiles = set()

    state.plan_steps = ()
    state.plan_index = 0
    state.defer_step_once = False

    state.phase = "network_select_candidate"
    state.network_wait_logged = False
    state.network_target = None
    _reset_network_path_state(state)
    state.network_escape_bridge_target = None
    state.network_highway_pending_harvester = (hx, hy)
    state.network_highway_active_harvester = None
    state.highway_excluded_transport = ()
    state.active_network_id = _ensure_active_network_id_for_harvester(
        state,
        (hx, hy),
        rnd,
    )

    log_event(
        rnd,
        uid,
        "economy",
        f"({cur_xy[0]},{cur_xy[1]})",
        "economy_repair_handoff_takeover_rebuild",
        hx=hx,
        hy=hy,
        nid=network_id,
        broken_tiles=len(broken_tiles),
        pending=len(state.repair_pending_conveyors),
        cooldown_until=cooldown_until,
        mode="network_takeover",
    )


def _repair_network_needs_attention(local_map, rec, harvester_xy) -> bool:
    if _adjacent_enemy_repair_blocker_targets(local_map, harvester_xy):
        return True

    expected_adjacent = _repair_expected_adjacent_conveyor_tiles(
        rec,
        harvester_xy,
    )
    if expected_adjacent:
        checked_any = False
        matched_any = False
        for tile_xy in expected_adjacent:
            matched, _, _, checked = _visible_friendly_conveyor_match(
                local_map,
                tile_xy,
            )
            if checked:
                checked_any = True
            if checked and matched:
                matched_any = True
                break

        if checked_any and not matched_any:
            return True

    chain_xy = _repair_collect_bridge_to_harvester_chain(rec)
    if not chain_xy:
        return False

    if _repair_collect_enemy_blockers_along_chain(local_map, chain_xy):
        return True

    my_team = getattr(local_map, "my_team", None)
    for tx, ty in chain_xy:
        if not local_map.in_bounds(tx, ty):
            continue
        if not _tile_is_known(local_map, tx, ty):
            continue

        try:
            if not local_map.is_visible(tx, ty):
                continue
        except (AttributeError, TypeError, GameError):
            continue

        tile_rec = _known_building_at(local_map, tx, ty)
        if not isinstance(tile_rec, dict):
            return True
        if tile_rec.get("team") != my_team:
            return True
        if tile_rec.get("entity_type") not in _TRANSPORT_ENTITY_TYPES:
            return True

    return False


def _repair_prepare_reroute_sequence(local_map, rec, avoid_tiles=()):
    chain = _repair_collect_bridge_to_harvester_chain(rec)
    if len(chain) < 2:
        return ()

    start_xy = (int(chain[0][0]), int(chain[0][1]))
    fallback_end_xy = (int(chain[-1][0]), int(chain[-1][1]))

    avoid_set = {
        (int(p[0]), int(p[1]))
        for p in avoid_tiles
        if isinstance(p, tuple) and len(p) == 2
    }
    my_team = getattr(local_map, "my_team", None)

    def is_friendly_transport_tile(x: int, y: int) -> bool:
        rec_xy = _known_building_at(local_map, x, y)
        if not isinstance(rec_xy, dict):
            return False
        if rec_xy.get("team") != my_team:
            return False
        return rec_xy.get("entity_type") in _TRANSPORT_ENTITY_TYPES

    endpoint_candidates = []
    seen_endpoints = set()
    harvester_xy = rec.get("harvester_pos")
    if isinstance(harvester_xy, tuple) and len(harvester_xy) == 2:
        hx = int(harvester_xy[0])
        hy = int(harvester_xy[1])
        for dx, dy in CARDINAL_DELTAS:
            ex = hx + dx
            ey = hy + dy
            end_xy = (ex, ey)
            if end_xy in seen_endpoints:
                continue
            if not local_map.in_bounds(ex, ey):
                continue
            if end_xy in avoid_set:
                continue
            seen_endpoints.add(end_xy)
            endpoint_candidates.append(end_xy)

    if fallback_end_xy not in seen_endpoints and fallback_end_xy not in avoid_set:
        endpoint_candidates.append(fallback_end_xy)

    if not endpoint_candidates:
        return ()

    # Prefer non-friendly-transport endpoints so reroute avoids tiles occupied by
    # existing friendly conveyor/bridge layouts that conflict with the repair plan.
    endpoint_candidates.sort(
        key=lambda p: (
            1 if is_friendly_transport_tile(p[0], p[1]) else 0,
            _manhattan(start_xy, p),
            p[0],
            p[1],
        )
    )

    best_nodes = None
    for end_xy in endpoint_candidates:
        if start_xy == end_xy:
            continue
        if is_friendly_transport_tile(end_xy[0], end_xy[1]):
            continue

        def passable_fn(x: int, y: int) -> bool:
            if (x, y) != start_xy and (x, y) != end_xy and (x, y) in avoid_set:
                return False

            if not _is_conveyor_planner_passable(local_map, x, y, end_xy):
                return False

            if (x, y) == start_xy or (x, y) == end_xy:
                return True

            return not is_friendly_transport_tile(x, y)

        steps = _astar_cardinal_plan(
            local_map,
            start_xy,
            end_xy,
            max_expansions=1536,
            tile_passable_fn=passable_fn,
        )
        if not steps:
            continue

        nodes = (start_xy, *steps)
        if nodes[-1] != end_xy:
            continue

        if best_nodes is None or len(nodes) < len(best_nodes):
            best_nodes = nodes

    if best_nodes is None:
        return ()

    seq = []
    for idx in range(len(best_nodes) - 1, 0, -1):
        tx, ty = best_nodes[idx]
        nx, ny = best_nodes[idx - 1]
        out_dir = _CARDINAL_DIRECTION_BY_DELTA.get((nx - tx, ny - ty))
        if out_dir is None:
            return ()
        seq.append((int(tx), int(ty), out_dir.name, -1))

    return tuple(seq)


def _repair_prepare_rebuild_sequence(local_map, rec, avoid_tiles=()):
    avoid_set = {
        (int(p[0]), int(p[1]))
        for p in avoid_tiles
        if isinstance(p, tuple) and len(p) == 2
    }
    if avoid_set:
        return _repair_prepare_reroute_sequence(local_map, rec, avoid_set)

    chain = _repair_collect_bridge_to_harvester_chain(rec)
    memory_index = _repair_build_conveyor_memory_index(rec)
    if not chain or not memory_index:
        return ()

    seq = []
    for tx, ty in reversed(chain):
        mem = memory_index.get((tx, ty))
        if mem is None:
            continue
        direction_name, expected_id = mem
        seq.append((tx, ty, direction_name, expected_id))

    return tuple(seq)
def _enqueue_repair_target(
    state: EconomyState,
    network_id: int,
    tile_xy,
    direction_name: str,
    expected_id: int,
    rnd: int,
):
    pending = state.repair_pending_conveyors
    if not isinstance(pending, dict):
        state.repair_pending_conveyors = {}
        pending = state.repair_pending_conveyors

    key = (int(tile_xy[0]), int(tile_xy[1]))
    if _is_repair_handoff_cooldown_active(state, key, rnd):
        return False

    entry = pending.get(key)
    if not isinstance(entry, dict):
        pending[key] = {
            "network_id": int(network_id),
            "direction": direction_name,
            "expected_id": int(expected_id) if isinstance(expected_id, int) else -1,
            "fails": 0,
        }
        if len(pending) == 1:
            state.repair_enqueue_round = rnd
        return True

    entry["network_id"] = int(network_id)
    entry["direction"] = direction_name
    entry["expected_id"] = int(expected_id) if isinstance(
        expected_id, int) else -1
    entry["fails"] = int(entry.get("fails", 0))
    return False


def _is_repair_handoff_cooldown_active(state: EconomyState, tile_xy, rnd: int) -> bool:
    cooldown = state.repair_handoff_cooldown_until
    if not isinstance(cooldown, dict):
        state.repair_handoff_cooldown_until = {}
        return False

    key = (int(tile_xy[0]), int(tile_xy[1]))
    hold_until = cooldown.get(key)
    if isinstance(hold_until, int) and rnd < hold_until:
        return True

    if hold_until is not None:
        cooldown.pop(key, None)

    return False


def _drop_repair_target(state: EconomyState, tile_xy):
    pending = state.repair_pending_conveyors
    if not isinstance(pending, dict):
        return

    key = (int(tile_xy[0]), int(tile_xy[1]))
    pending.pop(key, None)
    if not pending:
        state.repair_enqueue_round = -1


def _increment_repair_target_failures(state: EconomyState, tile_xy):
    pending = state.repair_pending_conveyors
    if not isinstance(pending, dict):
        return 0

    key = (int(tile_xy[0]), int(tile_xy[1]))
    entry = pending.get(key)
    if not isinstance(entry, dict):
        return 0

    entry["fails"] = max(0, int(entry.get("fails", 0))) + 1
    return int(entry["fails"])


def _update_conveyor_memory_id_in_all_networks(
    state: EconomyState,
    tile_xy,
    direction_name: str | None,
    entity_id: int | None,
):
    if not isinstance(direction_name, str):
        direction_name = None
    if isinstance(direction_name, str) and direction_name not in _CARDINAL_DIRECTION_BY_NAME:
        direction_name = None

    if not isinstance(entity_id, int) and direction_name is None:
        return 0

    tx = int(tile_xy[0])
    ty = int(tile_xy[1])
    changed = 0

    for network_id, rec in state.network_records.items():
        if not isinstance(network_id, int):
            continue
        if not isinstance(rec, dict):
            continue

        entries = _normalise_conveyor_memory_entries(
            rec.get("conveyor_memory", ()))
        if not entries:
            continue

        updated = []
        rec_changed = False
        for x, y, dir_name, old_id in entries:
            if (x, y) == (tx, ty):
                next_dir = direction_name if isinstance(
                    direction_name, str) else dir_name
                next_id = int(entity_id) if isinstance(
                    entity_id, int) else old_id
                updated.append((x, y, next_dir, next_id))
                if next_dir != dir_name or next_id != old_id:
                    rec_changed = True
                    changed += 1
            else:
                updated.append((x, y, dir_name, old_id))

        if rec_changed:
            rec["conveyor_memory"] = tuple(updated)
            state.network_records[network_id] = rec

    return changed


def _scan_conveyor_integrity_once(state: EconomyState, local_map, rnd: int, uid: int, cur_xy):
    if state.phase not in _REPAIR_SCAN_PHASES:
        return

    records = getattr(state, "network_records", None)
    if not isinstance(records, dict):
        state.network_records = {}
        records = state.network_records

    candidate_ids = []
    harvester_by_network = {}
    for network_id, rec in records.items():
        if not isinstance(network_id, int):
            continue
        if not isinstance(rec, dict):
            continue
        if not rec.get("complete", False):
            continue
        if _normalise_conveyor_memory_entries(rec.get("conveyor_memory", ())):
            harvester_xy = rec.get("harvester_pos")
            if not (isinstance(harvester_xy, tuple) and len(harvester_xy) == 2):
                continue
            harvester_by_network[network_id] = (
                int(harvester_xy[0]),
                int(harvester_xy[1]),
            )
            candidate_ids.append(network_id)

    candidate_ids.sort()
    scan_network_ids = []
    scan_network_set = set()

    if candidate_ids:
        cursor = max(0, int(state.repair_scan_cursor))
        scan_count = min(len(candidate_ids), _REPAIR_SCAN_NETWORKS_PER_ROUND)
        state.repair_scan_cursor = cursor + scan_count

        for scan_offset in range(scan_count):
            network_id = candidate_ids[(
                cursor + scan_offset) % len(candidate_ids)]
            scan_network_ids.append(network_id)
            scan_network_set.add(network_id)

    # Prioritize any currently visible harvesters whose network needs repair so
    # patrol-time detection is not delayed by round-robin scan cursor position.
    for network_id in candidate_ids:
        rec = records.get(network_id)
        if not isinstance(rec, dict):
            continue

        harvester_xy = harvester_by_network.get(network_id)
        if not (isinstance(harvester_xy, tuple) and len(harvester_xy) == 2):
            continue

        hx, hy = harvester_xy
        if not _tile_is_known(local_map, hx, hy):
            continue
        try:
            if not local_map.is_visible(hx, hy):
                continue
        except (AttributeError, TypeError, GameError):
            continue

        if not _repair_network_needs_attention(local_map, rec, harvester_xy):
            continue

        if network_id not in scan_network_set:
            scan_network_ids.append(network_id)
            scan_network_set.add(network_id)

    pending_before = len(state.repair_pending_conveyors)
    scanned_ids = []
    candidates_hit = 0
    visible_hits = 0
    visible_threats = set()

    for network_id in scan_network_ids:
        rec = records.get(network_id)
        if not isinstance(rec, dict):
            continue

        scanned_ids.append(network_id)

        harvester_xy = harvester_by_network.get(network_id)
        if not (isinstance(harvester_xy, tuple) and len(harvester_xy) == 2):
            continue

        if not _repair_network_needs_attention(local_map, rec, harvester_xy):
            continue

        if _enqueue_repair_target(
            state,
            network_id,
            harvester_xy,
            "HARVESTER",
            -1,
            rnd,
        ):
            candidates_hit += 1

    # Fallback: scan visible friendly/stolen harvesters directly for adjacent
    # enemy blockers, even if this bot lacks complete network memory.
    my_team = getattr(local_map, "my_team", None)
    for hxy in sorted(getattr(local_map, "titanium_harvesters", set())):
        if not (isinstance(hxy, tuple) and len(hxy) == 2):
            continue

        hx = int(hxy[0])
        hy = int(hxy[1])
        if not local_map.in_bounds(hx, hy):
            continue
        if not _tile_is_known(local_map, hx, hy):
            continue

        try:
            if not local_map.is_visible(hx, hy):
                continue
        except (AttributeError, TypeError, GameError):
            continue

        hrec = _known_building_at(local_map, hx, hy)
        if not isinstance(hrec, dict):
            continue
        if hrec.get("entity_type") != EntityType.HARVESTER:
            continue

        hid = _entity_id_from_rec(hrec)
        trusted_stolen = (
            (hx, hy) in state.harvester_stolen_positions
            or (isinstance(hid, int) and hid in state.harvester_ids_stolen)
        )
        if hrec.get("team") != my_team and not trusted_stolen:
            continue

        attack_targets = _adjacent_enemy_repair_blocker_targets(
            local_map, (hx, hy))
        if not attack_targets:
            continue

        network_id = state.network_id_by_harvester.get((hx, hy))
        if not isinstance(network_id, int) or not isinstance(records.get(network_id), dict):
            network_id = _ensure_active_network_id_for_harvester(
                state,
                (hx, hy),
                rnd,
            )
            records = state.network_records

        if not isinstance(network_id, int):
            continue

        if _enqueue_repair_target(
            state,
            network_id,
            (hx, hy),
            "HARVESTER",
            -1,
            rnd,
        ):
            candidates_hit += 1
            visible_hits += 1
        visible_threats.add((hx, hy))

    pending_after = len(state.repair_pending_conveyors)
    if candidates_hit > 0 or pending_before != pending_after:
        log_event(
            rnd,
            uid,
            "economy",
            f"({cur_xy[0]},{cur_xy[1]})",
            "economy_repair_scan_delta",
            nid=(scanned_ids[0] if len(scanned_ids) == 1 else -1),
            scanned_networks=len(scanned_ids),
            scanned_nids=_compact_sorted_ints(scanned_ids),
            candidates=len(candidate_ids),
            harvester_hits=candidates_hit,
            visible_threats=len(visible_threats),
            visible_hits=visible_hits,
            pending=pending_after,
            queue=_format_repair_pending_dump(state.repair_pending_conveyors),
        )


def _should_start_conveyor_repair(state: EconomyState, rnd: int) -> bool:
    pending = state.repair_pending_conveyors
    if not isinstance(pending, dict) or not pending:
        return False
    if state.repair_enqueue_round >= 0 and rnd <= state.repair_enqueue_round:
        return False
    return True


def _start_conveyor_repair(state: EconomyState, rnd: int, uid: int, cur_xy):
    state.repair_resume_phase = state.phase
    state.repair_target_xy = None
    state.repair_target_direction_name = None
    state.repair_target_network_id = None
    state.repair_target_expected_id = None
    state.repair_harvester_goal_xy = None
    state.repair_rebuild_sequence = ()
    state.repair_rebuild_index = 0
    state.plan_steps = ()
    state.plan_index = 0
    state.defer_step_once = False
    state.phase = "repair_pick_target"

    log_event(
        rnd,
        uid,
        "economy",
        f"({cur_xy[0]},{cur_xy[1]})",
        "economy_repair_start",
        pending=len(state.repair_pending_conveyors),
        resume=state.repair_resume_phase,
        queue=_format_repair_pending_dump(state.repair_pending_conveyors),
    )


def _normalise_repair_target_network_id(
    state: EconomyState,
    target_xy,
    entry_network_id,
    rnd: int,
) -> int | None:
    tx = int(target_xy[0])
    ty = int(target_xy[1])
    hxy = (tx, ty)

    if isinstance(entry_network_id, int):
        rec = state.network_records.get(entry_network_id)
        if isinstance(rec, dict):
            hp = rec.get("harvester_pos")
            if isinstance(hp, tuple) and len(hp) == 2:
                if (int(hp[0]), int(hp[1])) == hxy:
                    return int(entry_network_id)

    mapped_id = state.network_id_by_harvester.get(hxy)
    if isinstance(mapped_id, int):
        rec = state.network_records.get(mapped_id)
        if isinstance(rec, dict):
            return int(mapped_id)

    best_id = None
    best_key = None
    for network_id, rec in state.network_records.items():
        if not isinstance(network_id, int):
            continue
        if not isinstance(rec, dict):
            continue
        hp = rec.get("harvester_pos")
        if not (isinstance(hp, tuple) and len(hp) == 2):
            continue
        if (int(hp[0]), int(hp[1])) != hxy:
            continue

        key = (
            0 if rec.get("complete", False) else 1,
            int(network_id),
        )
        if best_key is None or key < best_key:
            best_key = key
            best_id = int(network_id)

    if isinstance(best_id, int):
        return best_id

    return _ensure_active_network_id_for_harvester(
        state,
        hxy,
        rnd,
    )


def _pick_best_repair_target(state: EconomyState, cur_xy, rnd: int):
    pending = state.repair_pending_conveyors
    if not isinstance(pending, dict) or not pending:
        return None

    candidates = []
    invalid_keys = []
    for tile_xy, entry in pending.items():
        if not (isinstance(tile_xy, tuple) and len(tile_xy) == 2):
            invalid_keys.append(tile_xy)
            continue
        if not isinstance(entry, dict):
            invalid_keys.append(tile_xy)
            continue

        tx = int(tile_xy[0])
        ty = int(tile_xy[1])
        network_id = _normalise_repair_target_network_id(
            state,
            (tx, ty),
            entry.get("network_id"),
            rnd,
        )
        if not isinstance(network_id, int):
            invalid_keys.append(tile_xy)
            continue
        entry["network_id"] = int(network_id)

        net_rec = state.network_records.get(network_id)
        if not isinstance(net_rec, dict):
            invalid_keys.append(tile_xy)
            continue

        fails = max(0, int(entry.get("fails", 0)))

        candidates.append(
            (
                fails,
                _manhattan(cur_xy, (tx, ty)),
                tx,
                ty,
                (tx, ty),
                (int(network_id) if isinstance(network_id, int) else None),
            )
        )

    for key in invalid_keys:
        pending.pop(key, None)

    if not pending:
        state.repair_enqueue_round = -1
        return None

    if not candidates:
        return None

    _, _, _, _, tile_xy, network_id = min(
        candidates)
    return tile_xy, network_id


def _repair_resume_phase(state: EconomyState):
    resume_phase = state.repair_resume_phase

    if resume_phase in _HARVEST_TITANIUM_PHASES:
        if state.harvest_ore_xy is None:
            state.phase = "harvest_pick_ore"
        elif state.harvest_goal_xy is None:
            state.phase = "harvest_pick_goal"
        else:
            state.phase = "harvest_plan_goal"
        return

    if resume_phase in _HARVEST_AXIONITE_PHASES:
        state.phase = (
            "axionite_pick_ore"
            if resume_phase == "axionite_done"
            else resume_phase
        )
        return

    if resume_phase in (
        "patrol_enter",
        "patrol_generate_waypoints",
        "patrol_replace_waypoint",
        "patrol_plan_waypoint",
        "patrol_follow_plan",
    ):
        state.phase = (
            "patrol_plan_waypoint"
            if state.explore_waypoints
            else "patrol_generate_waypoints"
        )
        return

    if resume_phase in (
        "launched",
        "explore_generate_waypoints",
        "explore_replace_waypoint",
        "explore_plan_waypoint",
        "explore_follow_plan",
        "explore_done",
    ):
        state.phase = (
            "explore_plan_waypoint"
            if state.explore_waypoints
            else "explore_generate_waypoints"
        )
        return

    state.phase = (
        "patrol_generate_waypoints"
        if state.patrol_unlocked
        else "explore_generate_waypoints"
    )


def _finish_conveyor_repair(state: EconomyState, rnd: int, uid: int, cur_xy):
    resume_phase = state.repair_resume_phase

    state.repair_target_xy = None
    state.repair_target_direction_name = None
    state.repair_target_network_id = None
    state.repair_target_expected_id = None
    state.repair_harvester_goal_xy = None
    state.repair_rebuild_sequence = ()
    state.repair_rebuild_index = 0
    state.repair_rebuild_avoid_tiles = set()
    state.plan_steps = ()
    state.plan_index = 0
    state.defer_step_once = False
    _repair_resume_phase(state)
    state.repair_resume_phase = None

    log_event(
        rnd,
        uid,
        "economy",
        f"({cur_xy[0]},{cur_xy[1]})",
        "economy_repair_complete_resume",
        resume=resume_phase,
        phase=state.phase,
    )


def _run_conveyor_repair(
    c: Controller,
    state: EconomyState,
    local_map,
    cur_xy,
    rnd: int,
    uid: int,
):
    pending = state.repair_pending_conveyors
    if not isinstance(pending, dict) or not pending:
        _finish_conveyor_repair(state, rnd, uid, cur_xy)
        return

    if state.phase == "repair_pick_target":
        choice = _pick_best_repair_target(state, cur_xy, rnd)
        if choice is None:
            _finish_conveyor_repair(state, rnd, uid, cur_xy)
            return

        target_xy, network_id = choice

        rec = state.network_records.get(network_id)
        if not isinstance(rec, dict):
            _drop_repair_target(state, target_xy)
            return

        if not _repair_network_needs_attention(local_map, rec, target_xy):
            _drop_repair_target(state, target_xy)
            log_event(
                rnd,
                uid,
                "economy",
                f"({cur_xy[0]},{cur_xy[1]})",
                "economy_repair_skip_network_healthy",
                hx=target_xy[0],
                hy=target_xy[1],
                nid=network_id,
            )
            return

        state.repair_target_xy = target_xy
        state.repair_target_direction_name = "HARVESTER"
        state.repair_target_network_id = network_id
        state.repair_target_expected_id = -1
        state.repair_harvester_goal_xy = None
        state.repair_rebuild_sequence = ()
        state.repair_rebuild_index = 0
        state.repair_rebuild_avoid_tiles = set()
        state.plan_steps = ()
        state.plan_index = 0
        state.defer_step_once = False
        state.phase = "repair_plan_harvester"
        log_event(
            rnd,
            uid,
            "economy",
            f"({cur_xy[0]},{cur_xy[1]})",
            "economy_repair_harvester_target_selected",
            hx=target_xy[0],
            hy=target_xy[1],
            nid=(network_id if isinstance(network_id, int) else -1),
        )
        return

    if state.phase == "repair_plan_harvester":
        target_xy = state.repair_target_xy
        network_id = state.repair_target_network_id
        if not (isinstance(target_xy, tuple) and len(target_xy) == 2):
            state.phase = "repair_pick_target"
            return
        if not isinstance(network_id, int):
            state.phase = "repair_pick_target"
            return

        if target_xy not in state.repair_pending_conveyors:
            state.phase = "repair_pick_target"
            return

        rec = state.network_records.get(network_id)
        if not isinstance(rec, dict):
            _drop_repair_target(state, target_xy)
            state.phase = "repair_pick_target"
            return

        if not _repair_network_needs_attention(local_map, rec, target_xy):
            _drop_repair_target(state, target_xy)
            state.phase = "repair_pick_target"
            log_event(
                rnd,
                uid,
                "economy",
                f"({cur_xy[0]},{cur_xy[1]})",
                "economy_repair_skip_network_healthy",
                hx=target_xy[0],
                hy=target_xy[1],
                nid=network_id,
            )
            return

        goal_xy = _repair_harvester_build_goal(local_map, target_xy, cur_xy)
        if not (isinstance(goal_xy, tuple) and len(goal_xy) == 2):
            fails = _increment_repair_target_failures(state, target_xy)
            if fails >= _REPAIR_MAX_PLAN_RETRIES:
                _drop_repair_target(state, target_xy)
                state.phase = "repair_pick_target"
            return

        state.repair_harvester_goal_xy = goal_xy

        if cur_xy == goal_xy:
            state.plan_steps = ()
            state.plan_index = 0
            state.phase = "repair_attack_adjacent"
            return

        steps = _astar_cardinal_plan(
            local_map,
            cur_xy,
            goal_xy,
            max_expansions=512,
            tile_passable_fn=lambda x, y: _is_general_movement_passable(
                local_map,
                x,
                y,
                goal_xy,
            ),
        )

        used_relaxed_halo = 0
        if not steps:
            steps = _astar_cardinal_plan(
                local_map,
                cur_xy,
                goal_xy,
                max_expansions=512,
                tile_passable_fn=lambda x, y: _is_general_movement_passable(
                    local_map,
                    x,
                    y,
                    goal_xy,
                    respect_halo=False,
                ),
            )
            if steps:
                used_relaxed_halo = 1

        if not steps:
            fails = _increment_repair_target_failures(state, target_xy)
            state.plan_steps = ()
            state.plan_index = 0
            state.phase = "repair_pick_target"
            if fails >= _REPAIR_MAX_PLAN_RETRIES:
                _drop_repair_target(state, target_xy)
                log_event(
                    rnd,
                    uid,
                    "economy",
                    f"({cur_xy[0]},{cur_xy[1]})",
                    "economy_repair_harvester_deferred",
                    hx=target_xy[0],
                    hy=target_xy[1],
                    reason="plan_unreachable",
                    fails=fails,
                )
                return

            log_event(
                rnd,
                uid,
                "economy",
                f"({cur_xy[0]},{cur_xy[1]})",
                "economy_repair_harvester_plan_pending",
                hx=target_xy[0],
                hy=target_xy[1],
            )
            return

        state.plan_steps = steps
        state.plan_index = 0
        state.phase = "repair_follow_harvester"
        log_event(
            rnd,
            uid,
            "economy",
            f"({cur_xy[0]},{cur_xy[1]})",
            "economy_repair_harvester_plan_ready",
            hx=target_xy[0],
            hy=target_xy[1],
            gx=goal_xy[0],
            gy=goal_xy[1],
            steps=len(steps),
            relaxed_halo=used_relaxed_halo,
        )
        return

    if state.phase == "repair_follow_harvester":
        target_xy = state.repair_target_xy
        if not (isinstance(target_xy, tuple) and len(target_xy) == 2):
            state.phase = "repair_pick_target"
            return

        goal_xy = state.repair_harvester_goal_xy
        if not (isinstance(goal_xy, tuple) and len(goal_xy) == 2):
            state.phase = "repair_plan_harvester"
            return

        if target_xy not in state.repair_pending_conveyors:
            state.phase = "repair_pick_target"
            return

        if cur_xy == goal_xy:
            state.plan_steps = ()
            state.plan_index = 0
            state.phase = "repair_attack_adjacent"
            return

        if state.plan_index >= len(state.plan_steps):
            state.phase = "repair_plan_harvester"
            return

        nxt = state.plan_steps[state.plan_index]
        if not _is_adjacent_step(cur_xy, nxt):
            state.phase = "repair_plan_harvester"
            return

        result = _execute_step_toward(c, local_map, cur_xy, nxt, rnd, uid)
        if result == "moved":
            state.issued_move_last_tick = True
            state.expected_xy_after_move = nxt
            state.plan_index += 1
            return
        if result in ("wait_cd", "built"):
            return

        state.plan_steps = ()
        state.plan_index = 0
        state.phase = "repair_plan_harvester"
        fails = _increment_repair_target_failures(state, target_xy)
        state.plan_steps = ()
        state.plan_index = 0
        if fails >= _REPAIR_MAX_PLAN_RETRIES:
            _drop_repair_target(state, target_xy)
            state.phase = "repair_pick_target"
            log_event(
                rnd,
                uid,
                "economy",
                f"({cur_xy[0]},{cur_xy[1]})",
                "economy_repair_harvester_deferred",
                hx=target_xy[0],
                hy=target_xy[1],
                reason="move_blocked",
                fails=fails,
            )
        return

    if state.phase == "repair_attack_adjacent":
        target_xy = state.repair_target_xy
        network_id = state.repair_target_network_id
        if not (isinstance(target_xy, tuple) and len(target_xy) == 2):
            state.phase = "repair_pick_target"
            return

        if not isinstance(network_id, int):
            state.phase = "repair_pick_target"
            return

        if target_xy not in state.repair_pending_conveyors:
            state.phase = "repair_pick_target"
            return

        goal_xy = _repair_harvester_build_goal(local_map, target_xy, cur_xy)
        if isinstance(goal_xy, tuple) and len(goal_xy) == 2 and cur_xy != goal_xy:
            state.phase = "repair_plan_harvester"
            return

        attack_targets = _adjacent_enemy_repair_blocker_targets(
            local_map,
            target_xy,
        )
        if not attack_targets:
            state.phase = "repair_attack_path"
            log_event(
                rnd,
                uid,
                "economy",
                f"({cur_xy[0]},{cur_xy[1]})",
                "economy_repair_adjacent_no_blockers",
                hx=target_xy[0],
                hy=target_xy[1],
                nid=network_id,
            )
            return

        _start_network_attack(
            state,
            attack_targets[0],
            "repair_attack_path",
            "repair_adjacent_enemy_blockers",
            target_queue=attack_targets,
            return_xy=cur_xy,
        )
        log_event(
            rnd,
            uid,
            "economy",
            f"({cur_xy[0]},{cur_xy[1]})",
            "economy_repair_adjacent_attack_start",
            hx=target_xy[0],
            hy=target_xy[1],
            targets=len(attack_targets),
            nid=network_id,
        )
        return

    if state.phase == "repair_attack_path":
        target_xy = state.repair_target_xy
        network_id = state.repair_target_network_id
        if not (isinstance(target_xy, tuple) and len(target_xy) == 2):
            state.phase = "repair_pick_target"
            return
        if not isinstance(network_id, int):
            state.phase = "repair_pick_target"
            return
        if target_xy not in state.repair_pending_conveyors:
            state.phase = "repair_pick_target"
            return

        rec = state.network_records.get(network_id)
        if not isinstance(rec, dict):
            _drop_repair_target(state, target_xy)
            state.phase = "repair_pick_target"
            return

        _repair_handoff_to_takeover_rebuild(
            state,
            target_xy,
            network_id,
            rnd,
            uid,
            cur_xy,
        )
        return

    if state.phase == "repair_rebuild_plan":
        target_xy = state.repair_target_xy
        network_id = state.repair_target_network_id
        if not (isinstance(target_xy, tuple) and len(target_xy) == 2):
            state.phase = "repair_pick_target"
            return
        if not isinstance(network_id, int):
            state.phase = "repair_pick_target"
            return
        if target_xy not in state.repair_pending_conveyors:
            state.phase = "repair_pick_target"
            return

        rec = state.network_records.get(network_id)
        if not isinstance(rec, dict):
            _drop_repair_target(state, target_xy)
            state.phase = "repair_pick_target"
            return

        seq = _repair_prepare_rebuild_sequence(
            local_map,
            rec,
            avoid_tiles=state.repair_rebuild_avoid_tiles,
        )
        if not seq:
            _drop_repair_target(state, target_xy)
            state.repair_rebuild_avoid_tiles = set()
            state.phase = "repair_pick_target"
            return

        state.repair_rebuild_sequence = seq
        state.repair_rebuild_index = 0
        state.plan_steps = ()
        state.plan_index = 0
        state.phase = "repair_rebuild_follow"
        log_event(
            rnd,
            uid,
            "economy",
            f"({cur_xy[0]},{cur_xy[1]})",
            "economy_repair_rebuild_sequence_ready",
            hx=target_xy[0],
            hy=target_xy[1],
            nid=network_id,
            conveyors=len(seq),
        )
        return

    if state.phase == "repair_rebuild_follow":
        target_xy = state.repair_target_xy
        if not (isinstance(target_xy, tuple) and len(target_xy) == 2):
            state.phase = "repair_pick_target"
            return
        if target_xy not in state.repair_pending_conveyors:
            state.phase = "repair_pick_target"
            return

        seq = state.repair_rebuild_sequence
        idx = int(state.repair_rebuild_index)
        if idx >= len(seq):
            _drop_repair_target(state, target_xy)
            state.repair_target_xy = None
            state.repair_target_network_id = None
            state.repair_harvester_goal_xy = None
            state.repair_rebuild_sequence = ()
            state.repair_rebuild_index = 0
            state.repair_rebuild_avoid_tiles = set()
            state.plan_steps = ()
            state.plan_index = 0
            if not state.repair_pending_conveyors:
                _finish_conveyor_repair(state, rnd, uid, cur_xy)
                return
            state.phase = "repair_pick_target"
            return

        tx, ty, _, _ = seq[idx]
        tile_xy = (int(tx), int(ty))
        if cur_xy == tile_xy:
            state.phase = "repair_rebuild_build"
            state.plan_steps = ()
            state.plan_index = 0
            return

        if state.plan_index >= len(state.plan_steps):
            steps = _astar_cardinal_plan(
                local_map,
                cur_xy,
                tile_xy,
                max_expansions=384,
                tile_passable_fn=lambda x, y: _is_general_movement_passable(
                    local_map,
                    x,
                    y,
                    tile_xy,
                ),
            )
            if not steps:
                steps = _astar_cardinal_plan(
                    local_map,
                    cur_xy,
                    tile_xy,
                    max_expansions=384,
                    tile_passable_fn=lambda x, y: _is_general_movement_passable(
                        local_map,
                        x,
                        y,
                        tile_xy,
                        respect_halo=False,
                    ),
                )
            if not steps:
                fails = _increment_repair_target_failures(state, target_xy)
                if fails >= _REPAIR_MAX_PLAN_RETRIES:
                    _drop_repair_target(state, target_xy)
                    state.phase = "repair_pick_target"
                    return
                return
            state.plan_steps = steps
            state.plan_index = 0

        nxt = state.plan_steps[state.plan_index]
        if not _is_adjacent_step(cur_xy, nxt):
            state.plan_steps = ()
            state.plan_index = 0
            return

        result = _execute_step_toward(c, local_map, cur_xy, nxt, rnd, uid)
        if result == "moved":
            state.issued_move_last_tick = True
            state.expected_xy_after_move = nxt
            state.plan_index += 1
            return
        if result in ("wait_cd", "built"):
            return

        state.plan_steps = ()
        state.plan_index = 0
        return

    if state.phase == "repair_rebuild_build":
        target_xy = state.repair_target_xy
        if not (isinstance(target_xy, tuple) and len(target_xy) == 2):
            state.phase = "repair_pick_target"
            return
        if target_xy not in state.repair_pending_conveyors:
            state.phase = "repair_pick_target"
            return

        seq = state.repair_rebuild_sequence
        idx = int(state.repair_rebuild_index)
        if idx >= len(seq):
            state.phase = "repair_rebuild_follow"
            return

        tx, ty, direction_name, _expected_id = seq[idx]
        tile_xy = (int(tx), int(ty))
        direction_name = str(direction_name)

        if cur_xy != tile_xy:
            state.phase = "repair_rebuild_follow"
            return

        matched, seen_id, seen_dir_name, checked = _visible_friendly_conveyor_match(
            local_map,
            tile_xy,
            direction_name=direction_name,
            require_direction_match=True,
        )
        if checked and matched:
            _update_conveyor_memory_id_in_all_networks(
                state,
                tile_xy,
                seen_dir_name,
                seen_id,
            )
            state.repair_rebuild_index = idx + 1
            state.phase = "repair_rebuild_follow"
            return

        tile_rec = _known_building_at(local_map, tile_xy[0], tile_xy[1])
        friendly_transport_blocker = (
            isinstance(tile_rec, dict)
            and tile_rec.get("team") == getattr(local_map, "my_team", None)
            and tile_rec.get("entity_type") in _TRANSPORT_ENTITY_TYPES
        )
        if checked and friendly_transport_blocker and not matched:
            already_avoided = tile_xy in state.repair_rebuild_avoid_tiles
            state.repair_rebuild_avoid_tiles.add(tile_xy)
            if already_avoided:
                fails = _increment_repair_target_failures(state, target_xy)
                if fails >= _REPAIR_MAX_PLAN_RETRIES:
                    _drop_repair_target(state, target_xy)
                    state.repair_rebuild_avoid_tiles = set()
                    state.phase = "repair_pick_target"
                    return

            state.repair_rebuild_sequence = ()
            state.repair_rebuild_index = 0
            state.plan_steps = ()
            state.plan_index = 0
            state.phase = "repair_rebuild_plan"
            log_event(
                rnd,
                uid,
                "economy",
                f"({cur_xy[0]},{cur_xy[1]})",
                "economy_repair_rebuild_reroute_friendly_blocker",
                tx=tile_xy[0],
                ty=tile_xy[1],
                expect=direction_name,
                seen=(seen_dir_name if isinstance(
                    seen_dir_name, str) else "none"),
                already=(1 if already_avoided else 0),
            )
            return

        blocker_kind = _enemy_repair_blocker_kind_from_map(local_map, tile_xy)
        if blocker_kind is None:
            blocker_kind = _enemy_repair_blocker_kind_from_controller(
                c, tile_xy)

        if blocker_kind == "armoured":
            log_event(
                rnd,
                uid,
                "economy",
                f"({cur_xy[0]},{cur_xy[1]})",
                "economy_repair_rebuild_armoured_abort",
                tx=tile_xy[0],
                ty=tile_xy[1],
                hx=target_xy[0],
                hy=target_xy[1],
            )
            _drop_repair_target(state, target_xy)
            state.repair_target_xy = None
            state.repair_target_network_id = None
            state.repair_harvester_goal_xy = None
            state.repair_rebuild_sequence = ()
            state.repair_rebuild_index = 0
            state.repair_rebuild_avoid_tiles = set()
            state.plan_steps = ()
            state.plan_index = 0
            if not state.repair_pending_conveyors:
                _finish_conveyor_repair(state, rnd, uid, cur_xy)
                return
            state.phase = "repair_pick_target"
            return

        if blocker_kind is not None:
            _start_network_attack(
                state,
                tile_xy,
                "repair_rebuild_build",
                f"repair_rebuild_enemy_{blocker_kind}",
            )
            log_event(
                rnd,
                uid,
                "economy",
                f"({cur_xy[0]},{cur_xy[1]})",
                "economy_repair_rebuild_attack_start",
                tx=tile_xy[0],
                ty=tile_xy[1],
                kind=blocker_kind,
            )
            return

        out_dir = _CARDINAL_DIRECTION_BY_NAME.get(direction_name)
        if out_dir is None:
            state.repair_rebuild_index = idx + 1
            state.phase = "repair_rebuild_follow"
            return

        build_result = _build_conveyor_on_tile(
            c,
            tile_xy,
            out_dir,
            rnd,
            uid,
            "economy_repair_rebuild_conveyor_built",
            state=state,
        )
        if build_result == "wait_cd":
            return

        if build_result == "built":
            new_id = None
            tp = Position(tile_xy[0], tile_xy[1])
            try:
                tile_id = c.get_tile_building_id(tp)
            except GameError:
                tile_id = None

            if isinstance(tile_id, int):
                try:
                    if (
                        c.get_team(tile_id) == c.get_team()
                        and c.get_entity_type(tile_id) == EntityType.CONVEYOR
                    ):
                        d = c.get_direction(tile_id)
                        if d is not None and d.name == direction_name:
                            new_id = tile_id
                except GameError:
                    pass

            if isinstance(new_id, int):
                _update_conveyor_memory_id_in_all_networks(
                    state,
                    tile_xy,
                    direction_name,
                    new_id,
                )

            state.repair_rebuild_index = idx + 1
            state.phase = "repair_rebuild_follow"
            return

        fails = _increment_repair_target_failures(state, target_xy)
        if fails >= _REPAIR_MAX_PLAN_RETRIES:
            _drop_repair_target(state, target_xy)
            state.phase = "repair_pick_target"
            return
        return

    state.phase = "repair_pick_target"


# ===== TITANIUM HARVEST =====
# Phases: harvest_enter → harvest_pick_ore → harvest_pick_goal →
#         harvest_plan_goal → harvest_follow_plan → harvest_build →
#         harvest_takeover_* (enemy harvester takeover)

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
        state.harvest_takeover_harvester_id = None
        state.harvest_takeover_ore_xy = None
        state.plan_steps = ()
        state.plan_index = 0
        state.defer_step_once = False
        state.network_target = None
        _reset_network_path_state(state)
        state.network_wait_logged = False
        known_ti = _known_unharvested_titanium_unblocked(
            local_map,
            state.harvest_blocked_ores,
            core_xy=state.core_xy,
            min_core_cheb=_ORE_CORE_EXCLUSION_MIN_CHEB,
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
        _run_titanium_network_v2(
            c,
            state,
            local_map,
            cur_xy,
            rnd,
            uid,
        )
        return

    if state.phase == "harvest_takeover_finalize":
        _finalize_harvester_takeover(state, local_map, cur_xy, rnd, uid)
        return

    if state.phase in _HARVEST_TAKEOVER_PHASES:
        _run_enemy_harvester_takeover(c, state, local_map, cur_xy, rnd, uid)
        return

    known_ti_all = _known_unharvested_titanium(
        local_map,
        core_xy=state.core_xy,
        min_core_cheb=_ORE_CORE_EXCLUSION_MIN_CHEB,
    )
    known_ti_all_set = set(known_ti_all)
    state.harvest_blocked_ores.intersection_update(known_ti_all_set)
    known_ti = _known_unharvested_titanium_unblocked(
        local_map,
        state.harvest_blocked_ores,
        core_xy=state.core_xy,
        min_core_cheb=_ORE_CORE_EXCLUSION_MIN_CHEB,
    )

    if state.phase == "harvest_enter":
        state.phase = "harvest_pick_ore"
        return

    if state.phase == "harvest_pick_ore":
        if not known_ti:
            enemy_harvesters = _known_enemy_titanium_harvesters(
                local_map,
                state=state,
                core_xy=state.core_xy,
                min_core_cheb=_ORE_CORE_EXCLUSION_MIN_CHEB,
            )
            if enemy_harvesters:
                takeover_id, takeover_ore = min(
                    enemy_harvesters,
                    key=lambda item: (
                        _manhattan(cur_xy, item[1]),
                        item[1][0],
                        item[1][1],
                    ),
                )
                _start_enemy_harvester_takeover(
                    state,
                    cur_xy,
                    takeover_ore,
                    takeover_id,
                    rnd,
                    uid,
                )
                return

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

        # Prefer ores that were not deprioritised by an earlier failed build.
        # Deprioritised ores are only considered once no normal candidates
        # remain — this keeps contention from starving us of progress while
        # still avoiding an infinite retry loop on the contended ore.
        preferred_ores = [
            o for o in known_ti if o not in state.harvest_deprioritised_ores
        ]
        if preferred_ores:
            ore_xy = _pick_nearest_titanium_ore(cur_xy, preferred_ores)
        else:
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
    if ore_xy is None:
        state.phase = "harvest_pick_ore"
        return

    if ore_xy not in known_ti_all_set:
        takeover_id = _enemy_titanium_harvester_id_at(local_map, ore_xy)
        if isinstance(takeover_id, int) or _is_enemy_titanium_harvester_at(local_map, ore_xy):
            if _enemy_harvester_should_be_assumed_stolen(local_map, ore_xy):
                _mark_harvester_and_adjacent_transports_stolen(
                    state,
                    local_map,
                    ore_xy,
                    takeover_id,
                )
                state.harvest_ore_xy = None
                state.harvest_goal_xy = None
                state.plan_steps = ()
                state.plan_index = 0
                state.defer_step_once = False
                state.phase = "harvest_pick_ore"
                log_event(
                    rnd,
                    uid,
                    "economy",
                    f"({cur_xy[0]},{cur_xy[1]})",
                    "economy_harvester_takeover_skipped_assumed_stolen",
                    ox=ore_xy[0],
                    oy=ore_xy[1],
                    hid=(takeover_id if isinstance(takeover_id, int) else -1),
                )
                return
            _start_enemy_harvester_takeover(
                state,
                cur_xy,
                ore_xy,
                takeover_id,
                rnd,
                uid,
            )
            return
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

        def passable_fn(x: int, y: int) -> bool:
            return _is_general_movement_passable(local_map, x, y, goal_xy)

        steps = _astar_cardinal_plan(
            local_map,
            cur_xy,
            goal_xy,
            max_expansions=plan_budget,
            tile_passable_fn=passable_fn,
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
        if not _is_adjacent_step(cur_xy, nxt):
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
            can_build = c.can_build_harvester(ore_pos)
        except GameError:
            can_build = False

        if can_build:
            try:
                new_harvester_id = c.build_harvester(ore_pos)
                if isinstance(new_harvester_id, int):
                    state.built_entity_ids.add(new_harvester_id)
                state.built_harvester_positions.add((ore_xy[0], ore_xy[1]))
                state.harvest_deprioritised_ores.discard(ore_xy)
                state.phase = "network_select_candidate"
                state.network_wait_logged = False
                state.network_target = None
                _reset_network_path_state(state)
                state.network_escape_bridge_target = None
                state.network_highway_pending_harvester = ore_xy
                state.network_highway_active_harvester = None
                state.active_network_id = _ensure_active_network_id_for_harvester(
                    state,
                    ore_xy,
                    rnd,
                )
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
                    "economy_network_highway_fallback_pending_v2",
                    hx=ore_xy[0],
                    hy=ore_xy[1],
                )
                return
            except GameError:
                # build_harvester raised despite can_build being True — most
                # likely another bot built on the same ore tile this tick.
                # Enter the confirm state to resolve contention.
                state.harvest_build_pending_ore_xy = ore_xy
                state.harvest_build_confirm_deadline = rnd + 1
                state.phase = "harvest_confirm_other_bot_building"
                log_event(
                    rnd,
                    uid,
                    "economy",
                    f"({cur_xy[0]},{cur_xy[1]})",
                    "economy_harvest_build_contention_exception",
                    ox=ore_xy[0],
                    oy=ore_xy[1],
                )
                return

        # can_build_harvester returned False. This could be:
        #  - insufficient resources (we'd retry next round)
        #  - tile already holds a building (likely another friendly bot built)
        # Disambiguate via the ore-contention confirm flow. On the next tick
        # (post other-bot actions) vision will show what — if anything — was
        # built there.
        if _team_titanium(c) < int(HARVESTER_BASE_COST[0]):
            # Resource wait; stay in place and retry next round without
            # triggering the contention flow (don't permanently deprioritise
            # an ore just because we were briefly broke).
            return

        state.harvest_build_pending_ore_xy = ore_xy
        state.harvest_build_confirm_deadline = rnd + 1
        state.phase = "harvest_confirm_other_bot_building"
        log_event(
            rnd,
            uid,
            "economy",
            f"({cur_xy[0]},{cur_xy[1]})",
            "economy_harvest_build_contention_cannot_build",
            ox=ore_xy[0],
            oy=ore_xy[1],
        )
        return

    if state.phase == "harvest_confirm_other_bot_building":
        pending_ore = state.harvest_build_pending_ore_xy
        if not (isinstance(pending_ore, tuple) and len(pending_ore) == 2):
            state.harvest_build_pending_ore_xy = None
            state.phase = "harvest_pick_ore"
            return

        if rnd < state.harvest_build_confirm_deadline:
            # Give one tick for other bots to have acted and for our vision to
            # reflect the result.
            return

        pending_ore = (int(pending_ore[0]), int(pending_ore[1]))
        rec = _known_building_at(local_map, pending_ore[0], pending_ore[1])
        my_team = getattr(local_map, "my_team", None)

        if isinstance(rec, dict) and rec.get("entity_type") == EntityType.HARVESTER:
            if rec.get("team") == my_team:
                # Friendly harvester — someone else built it. Treat ore as
                # harvested; drop any lingering deprioritisation.
                state.harvest_blocked_ores.add(pending_ore)
                state.harvest_deprioritised_ores.discard(pending_ore)
                log_event(
                    rnd,
                    uid,
                    "economy",
                    f"({cur_xy[0]},{cur_xy[1]})",
                    "economy_harvest_confirm_friendly_built",
                    ox=pending_ore[0],
                    oy=pending_ore[1],
                )
            else:
                # Enemy harvester — takeover is the right response.
                takeover_id = _entity_id_from_rec(rec)
                _start_enemy_harvester_takeover(
                    state,
                    cur_xy,
                    pending_ore,
                    takeover_id if isinstance(takeover_id, int) else None,
                    rnd,
                    uid,
                )
                state.harvest_build_pending_ore_xy = None
                state.harvest_build_confirm_deadline = -1
                return
        else:
            # No harvester visible. Maybe the ore was briefly blocked by a
            # moving bot, or another bot is mid-travel. Deprioritise so we
            # try the next ore first — the entry persists until we see a
            # harvester there or build one ourselves (per user spec).
            state.harvest_deprioritised_ores.add(pending_ore)
            log_event(
                rnd,
                uid,
                "economy",
                f"({cur_xy[0]},{cur_xy[1]})",
                "economy_harvest_confirm_no_harvester",
                ox=pending_ore[0],
                oy=pending_ore[1],
            )

        state.harvest_build_pending_ore_xy = None
        state.harvest_build_confirm_deadline = -1
        state.harvest_ore_xy = None
        state.harvest_goal_xy = None
        state.plan_steps = ()
        state.plan_index = 0
        state.defer_step_once = False
        state.phase = "harvest_pick_ore"
        return


# ===== POST-LAUNCH EXPLORATION =====
# Phases: explore_generate_waypoints → explore_replace_waypoint →
#         explore_plan_waypoint → explore_follow_plan → explore_done →
#         patrol_generate_waypoints → patrol_replace_waypoint →
#         patrol_plan_waypoint → patrol_follow_plan

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
        state.phase = "explore_generate_waypoints"
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

        def passable_fn(x: int, y: int) -> bool:
            return _is_general_movement_passable(local_map, x, y, target_xy)

        steps = _astar_cardinal_plan(
            local_map,
            cur_xy,
            target_xy,
            max_expansions=plan_budget,
            tile_passable_fn=passable_fn,
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
        if not _is_adjacent_step(cur_xy, nxt):
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


# ===== NETWORK BUILDING v2 =====
# Main conveyor/bridge network builder. Connects titanium harvesters back to
# the core using conveyors, bridges, and splitters.
# Phases: network_wait → network_select_candidate →
#         network_plan_path → network_plan_path_resume →
#         network_bridge_escape_* → network_attack_blocker →
#         conveyor_initialisation → conveyor_execution → conveyor_termination

def _run_titanium_network_v2(
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
        core_xy=state.core_xy,
        min_core_cheb=_ORE_CORE_EXCLUSION_MIN_CHEB,
    )

    _refresh_friendly_network_registry(state, local_map, rnd)
    _recompute_direct_anchor_availability(state, local_map)

    if state.phase == "network_wait_resources":
        _run_resource_wait_phase(c, state, cur_xy, rnd, uid, "network")
        return

    if state.phase == "network_wait":
        if not state.network_wait_logged:
            log_event(
                rnd,
                uid,
                "economy",
                f"({cur_xy[0]},{cur_xy[1]})",
                "economy_network_behavior_pending_v2",
                direct_available=len(state.direct_anchor_available),
                known_networks=len(state.network_records),
            )
            state.network_wait_logged = True
        state.phase = "network_select_candidate"
        return

    if state.phase == "network_select_candidate":
        state.highway_excluded_transport = ()
        if (
            state.active_network_id is None
            and isinstance(state.network_highway_pending_harvester, tuple)
            and len(state.network_highway_pending_harvester) == 2
        ):
            state.active_network_id = _ensure_active_network_id_for_harvester(
                state,
                state.network_highway_pending_harvester,
                rnd,
            )
        has_pending_highway = state.network_highway_pending_harvester is not None

        pending_highway_source = None
        pending_highway_for_repair = False
        if (
            isinstance(state.network_highway_pending_harvester, tuple)
            and len(state.network_highway_pending_harvester) == 2
        ):
            pending_highway_source = (
                int(state.network_highway_pending_harvester[0]),
                int(state.network_highway_pending_harvester[1]),
            )
            pending_network_id = state.network_id_by_harvester.get(
                pending_highway_source)
            pending_highway_for_repair = (
                isinstance(pending_network_id, int)
                and pending_network_id in state.broken_network_ids
            )

        if pending_highway_for_repair:
            state.network_target = None
            state.network_escape_bridge_target = None
            _reset_network_path_state(state)
            state.phase = "network_bridge_escape_check"
            if pending_highway_source is not None:
                log_event(
                    rnd,
                    uid,
                    "economy",
                    f"({cur_xy[0]},{cur_xy[1]})",
                    "economy_network_pending_highway_priority_v2",
                    hx=int(pending_highway_source[0]),
                    hy=int(pending_highway_source[1]),
                )
            return

        direct_candidates = _collect_direct_anchor_candidates(
            state, cur_xy, rnd)
        direct_preview = ",".join(
            f"({bx},{by}):d{dist}"
            for dist, bx, by, _, _ in direct_candidates
        )
        deferred_direct = len(
            getattr(state, "network_unreachable_direct_until", {}))
        log_event(
            rnd,
            uid,
            "economy",
            f"({cur_xy[0]},{cur_xy[1]})",
            "economy_network_direct_targets_debug_v2",
            count=len(direct_candidates),
            blocked=len(state.direct_anchor_blocked),
            deferred=deferred_direct,
            targets=direct_preview,
        )

        # Selection priority: direct -> highway -> indirect.
        candidate = _select_direct_bridge_candidate(
            state, local_map, cur_xy, rnd)
        if candidate is None and has_pending_highway:
            state.phase = "network_bridge_escape_check"
            return
        if candidate is None:
            candidate = _select_lidar_bridge_candidate(
                state,
                local_map,
                cur_xy,
                rnd,
                uid,
            )
        if candidate is None:
            _clear_network_target_state(state, clear_highway=False)
            _network_fallback_to_next_objective(state, cur_xy, known_ti)
            log_event(
                rnd,
                uid,
                "economy",
                f"({cur_xy[0]},{cur_xy[1]})",
                "economy_network_no_candidate_v2",
                direct_available=len(state.direct_anchor_available),
                known_networks=len(state.network_records),
            )
            return

        state.network_target = candidate
        _reset_network_path_state(state)
        state.phase = "network_plan_path"
        log_event(
            rnd,
            uid,
            "economy",
            f"({cur_xy[0]},{cur_xy[1]})",
            "economy_network_target_selected_v2",
            mode=candidate["mode"],
            bx=candidate["bridge_pos"][0],
            by=candidate["bridge_pos"][1],
            tx=candidate["bridge_target"][0],
            ty=candidate["bridge_target"][1],
        )
        return

    if state.phase == "network_bridge_escape_check":
        highway_source = state.network_highway_pending_harvester
        if isinstance(highway_source, tuple) and len(highway_source) == 2:
            highway_source = (
                int(highway_source[0]),
                int(highway_source[1]),
            )
        else:
            highway_source = None

        selection_stats = {}

        if highway_source is not None:
            state.highway_excluded_transport = ()
            target_xy = _select_harvester_highway_bridge_target(
                state,
                local_map,
                highway_source,
                cur_xy,
                state.core_xy,
                exclude_positions=state.highway_excluded_transport,
                selection_stats=selection_stats,
            )
        else:
            target_xy = _select_bridge_escape_target_v2(
                state,
                local_map,
                cur_xy,
                state.core_xy,
                exclude_positions=state.highway_excluded_transport,
                selection_stats=selection_stats,
            )

        if target_xy is None:
            retry_same_target = bool(
                state.network_target is not None
                and highway_source is None
                and state.highway_excluded_transport
            )

            if retry_same_target:
                state.network_escape_bridge_target = None
                _reset_network_path_state(state)
                state.phase = "network_plan_path"
                log_event(
                    rnd,
                    uid,
                    "economy",
                    f"({cur_xy[0]},{cur_xy[1]})",
                    "economy_network_bridge_escape_replan_same_target_v2",
                    excluded=len(state.highway_excluded_transport),
                    highway=1 if highway_source is not None else 0,
                )
                return

            _clear_network_target_state(state, clear_highway=True)
            state.phase = "network_select_candidate"
            log_event(
                rnd,
                uid,
                "economy",
                f"({cur_xy[0]},{cur_xy[1]})",
                "economy_network_bridge_escape_none_v2",
                highway=1 if highway_source is not None else 0,
            )
            return

        state.network_escape_bridge_target = target_xy
        state.network_highway_active_harvester = highway_source
        state.network_highway_pending_harvester = None
        state.phase = "network_bridge_escape_execute"

        target_owner_ids = ()
        target_owner_index = _build_network_tile_owner_index(state)
        target_owners = target_owner_index.get(
            (int(target_xy[0]), int(target_xy[1])),
            set(),
        )
        if isinstance(target_owners, set):
            target_owner_ids = tuple(
                sorted(
                    int(owner)
                    for owner in target_owners
                    if isinstance(owner, int)
                )
            )

        source_network_id = _resolve_source_network_id(state, highway_source)
        active_network_id = (
            int(state.active_network_id)
            if isinstance(state.active_network_id, int)
            else -1
        )

        log_event(
            rnd,
            uid,
            "economy",
            f"({cur_xy[0]},{cur_xy[1]})",
            "economy_network_bridge_escape_ready_v2",
            tx=target_xy[0],
            ty=target_xy[1],
            highway=1 if highway_source is not None else 0,
            fallback=selection_stats.get("highway_local_fallback", 0),
            source_network=(source_network_id if isinstance(
                source_network_id, int) else -1),
            active_network=active_network_id,
            target_owner_ids=_compact_sorted_ints(target_owner_ids),
        )
        return

    if state.phase == "network_bridge_escape_execute":
        target_xy = state.network_escape_bridge_target
        if not (isinstance(target_xy, tuple) and len(target_xy) == 2):
            state.phase = "network_bridge_escape_check"
            return

        if _is_enemy_replaceable_blocker_at(local_map, cur_xy[0], cur_xy[1]):
            _start_network_attack(
                state,
                cur_xy,
                "network_bridge_escape_execute",
                "bridge_escape_self_blocked",
            )
            return

        bridge_result = _build_bridge_on_tile(
            c,
            cur_xy,
            target_xy,
            rnd,
            uid,
            state=state,
            replace_mismatched_existing_bridge=True,
        )
        if bridge_result == "wait_cd":
            return

        if bridge_result in ("built", "already_built"):
            if _deprioritize_successful_direct_anchor(state, state.network_target):
                log_event(
                    rnd,
                    uid,
                    "economy",
                    f"({cur_xy[0]},{cur_xy[1]})",
                    "economy_network_direct_anchor_deprioritized_v2",
                    bx=state.network_target["bridge_pos"][0],
                    by=state.network_target["bridge_pos"][1],
                )

            highway_source = state.network_highway_active_harvester
            if isinstance(highway_source, tuple) and len(highway_source) == 2:
                _record_bridge_only_titanium_link(
                    state,
                    highway_source,
                    cur_xy,
                    target_xy,
                )

            _record_active_network_terminal_bridge(
                state,
                cur_xy,
                target_xy,
                rnd,
            )

            _refresh_friendly_network_registry(state, local_map, rnd)
            _sync_dynamic_blocked_tiles(state, local_map)
            _clear_network_target_state(state, clear_highway=True)
            _network_fallback_to_next_objective(state, cur_xy, known_ti)
            log_event(
                rnd,
                uid,
                "economy",
                f"({cur_xy[0]},{cur_xy[1]})",
                "economy_network_bridge_escape_complete_v2",
                tx=target_xy[0],
                ty=target_xy[1],
                bridge_only_count=len(state.network_bridge_only_ti_links),
            )
            return

        if _is_enemy_replaceable_blocker_at(local_map, cur_xy[0], cur_xy[1]):
            _start_network_attack(
                state,
                cur_xy,
                "network_bridge_escape_execute",
                "bridge_escape_retry_after_attack",
            )
            return

        _clear_network_target_state(state, clear_highway=True)
        state.phase = "network_select_candidate"
        log_event(
            rnd,
            uid,
            "economy",
            f"({cur_xy[0]},{cur_xy[1]})",
            "economy_network_bridge_escape_invalid_v2",
        )
        return

    candidate = state.network_target
    if candidate is None:
        _reset_network_path_state(state)
        state.phase = "network_select_candidate"
        return

    if state.phase in ("network_plan_path", "network_plan_path_resume"):
        goal_xy = candidate["bridge_pos"]
        steps = ()
        plan_budget = 0
        if state.phase == "network_plan_path":
            if cur_xy == goal_xy:
                steps = ()
                plan_budget = 0
            else:
                excluded_positions = {
                    (int(p[0]), int(p[1]))
                    for p in state.highway_excluded_transport
                    if isinstance(p, tuple) and len(p) == 2
                }
                excluded_network_ids = set()
                tile_owner_index = None

                goal_mh = _manhattan(cur_xy, goal_xy)
                plan_budget = max(896, min(2048, goal_mh * 18))
                plan_session = _astar_cardinal_start_session(
                    cur_xy,
                    goal_xy,
                    max_expansions=plan_budget,
                )
                plan_session["excluded_positions"] = tuple(
                    sorted(excluded_positions))
                plan_session["excluded_network_ids"] = tuple(
                    sorted(excluded_network_ids))
                plan_session["tile_owner_index"] = tile_owner_index
                state.network_plan_session = plan_session
                state.phase = "network_plan_path_resume"

        if cur_xy != goal_xy:
            plan_session = state.network_plan_session
            if not isinstance(plan_session, dict):
                state.phase = "network_plan_path"
                return

            if plan_session.get("start_xy") != cur_xy or plan_session.get("goal_xy") != goal_xy:
                state.network_plan_session = None
                state.phase = "network_plan_path"
                return

            excluded_positions = {
                (int(p[0]), int(p[1]))
                for p in plan_session.get("excluded_positions", ())
                if isinstance(p, tuple) and len(p) == 2
            }
            excluded_network_ids = {
                int(i)
                for i in plan_session.get("excluded_network_ids", ())
                if isinstance(i, int)
            }
            tile_owner_index = plan_session.get("tile_owner_index")

            def passable_fn(x: int, y: int) -> bool:
                return _is_conveyor_planner_passable(
                    local_map,
                    x,
                    y,
                    goal_xy,
                    exclude_positions=excluded_positions,
                    exclude_network_ids=excluded_network_ids,
                    tile_owner_index=tile_owner_index,
                )

            def cost_fn(x: int, y: int) -> int:
                return _planner_tile_soft_cost(state, x, y)

            status, steps, _ = _astar_cardinal_continue_session(
                local_map,
                plan_session,
                step_expansions=_NETWORK_ASTAR_RESUME_STEP_EXPANSIONS,
                tile_passable_fn=passable_fn,
                tile_extra_cost_fn=cost_fn,
            )
            plan_budget = int(plan_session.get("max_expansions", 0))
            plan_session["rounds"] = int(plan_session.get("rounds", 0)) + 1

            if status == "in_progress":
                return

            if status != "found":
                if candidate.get("mode") == "direct":
                    removed = _defer_unreachable_direct_candidate(
                        state,
                        candidate.get("bridge_pos"),
                        rnd,
                    )
                    if removed:
                        log_event(
                            rnd,
                            uid,
                            "economy",
                            f"({cur_xy[0]},{cur_xy[1]})",
                            "economy_network_direct_removed_unreachable_v2",
                            bx=goal_xy[0],
                            by=goal_xy[1],
                            invalid_total=len(
                                state.network_invalid_bridge_positions),
                        )

                _reset_network_path_state(state)
                state.phase = "network_select_candidate"
                log_event(
                    rnd,
                    uid,
                    "economy",
                    f"({cur_xy[0]},{cur_xy[1]})",
                    "economy_network_plan_unreachable_v2",
                    bx=goal_xy[0],
                    by=goal_xy[1],
                    exhausted=1 if status == "budget_exhausted" else 0,
                )
                return

        state.network_plan_session = None
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
            "economy_network_plan_ready_v2",
            bx=goal_xy[0],
            by=goal_xy[1],
            tx=candidate["bridge_target"][0],
            ty=candidate["bridge_target"][1],
            steps=max(0, len(state.network_path_nodes) - 1),
            budget=plan_budget,
        )
        log_event(
            rnd,
            uid,
            "economy",
            f"({cur_xy[0]},{cur_xy[1]})",
            "economy_network_plan_dump_v2",
            start=f"({cur_xy[0]},{cur_xy[1]})",
            goal=f"({goal_xy[0]},{goal_xy[1]})",
            mode=candidate.get("mode"),
            path=_format_plan_dump(cur_xy, steps),
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
            _clear_network_target_state(state, clear_highway=False)
            state.phase = "network_select_candidate"
            log_event(
                rnd,
                uid,
                "economy",
                f"({cur_xy[0]},{cur_xy[1]})",
                "economy_network_armoured_blocked_v2",
                bx=nxt[0],
                by=nxt[1],
            )
            return

        if _is_enemy_replaceable_blocker_at(local_map, nxt[0], nxt[1]):
            _start_network_blocker_clearance(
                state,
                local_map,
                cur_xy,
                nxt,
                "conveyor_initialisation",
                "enemy_transport_ahead",
            )
            return

        if _is_unit_halo_predicted_blocked(local_map, nxt[0], nxt[1], from_xy=cur_xy):
            state.highway_excluded_transport = _capture_active_network_transport_exclusions(
                state
            )
            state.network_escape_bridge_target = None
            _reset_network_path_state(state)
            state.phase = "network_bridge_escape_check"
            log_event(
                rnd,
                uid,
                "economy",
                f"({cur_xy[0]},{cur_xy[1]})",
                "economy_network_halo_block_highway_switch_v2",
                bx=nxt[0],
                by=nxt[1],
            )
            return

        if candidate.get("mode") != "highway_direct":
            direct_bridge_target = _select_direct_bridge_sink_target(
                state,
                local_map,
                cur_xy,
                nxt,
            )
            if isinstance(direct_bridge_target, tuple) and len(direct_bridge_target) == 2:
                _sync_dynamic_blocked_tiles(state, local_map)
                state.network_escape_bridge_target = direct_bridge_target
                _reset_network_path_state(state)
                state.phase = "network_bridge_escape_execute"
                log_event(
                    rnd,
                    uid,
                    "economy",
                    f"({cur_xy[0]},{cur_xy[1]})",
                    "economy_network_bridge_sink_terminate_v2",
                    tx=direct_bridge_target[0],
                    ty=direct_bridge_target[1],
                    source="conveyor_initialisation",
                )
                return

        if (
            candidate.get("mode") != "highway_direct"
            and _is_friendly_transport_tile(local_map, nxt[0], nxt[1])
        ):
            if not _destroy_standing_transport_for_highway_switch(
                c,
                local_map,
                cur_xy,
                rnd,
                uid,
                "conveyor_initialisation",
            ):
                return

            if _destroy_friendly_broken_transport_blocker(
                c,
                state,
                local_map,
                nxt,
                rnd,
                uid,
                "conveyor_initialisation",
            ):
                _reset_network_path_state(state)
                state.phase = "network_plan_path"
                return

            is_terminal_step = (idx + 1) == (len(nodes) - 1)
            if is_terminal_step and _can_reuse_terminal_bridge(
                local_map,
                nxt,
                candidate.get("bridge_target"),
            ):
                log_event(
                    rnd,
                    uid,
                    "economy",
                    f"({cur_xy[0]},{cur_xy[1]})",
                    "economy_network_terminal_bridge_reuse_v2",
                    bx=nxt[0],
                    by=nxt[1],
                    tx=candidate["bridge_target"][0],
                    ty=candidate["bridge_target"][1],
                )
            else:
                captured = set(
                    _capture_active_network_transport_exclusions(state))
                captured.add((int(nxt[0]), int(nxt[1])))
                state.highway_excluded_transport = tuple(sorted(captured))
                state.network_escape_bridge_target = None
                _reset_network_path_state(state)
                state.phase = "network_bridge_escape_check"
                log_event(
                    rnd,
                    uid,
                    "economy",
                    f"({cur_xy[0]},{cur_xy[1]})",
                    "economy_network_friendly_transport_highway_switch_v2",
                    tx=nxt[0],
                    ty=nxt[1],
                )
                return

        build_result = _build_network_segment_transport(
            c,
            nodes[idx],
            nodes[idx],
            nxt,
            rnd,
            uid,
            state,
            conveyor_tag="economy_network_conveyor_init_v2",
            bridge_tag="economy_network_intermediate_bridge_init_v2",
        )
        if build_result == "wait_cd":
            return
        if build_result not in ("built", "already_built"):
            init_step_mode = (
                "bridge" if _is_diagonal_step(nodes[idx], nxt) else "conveyor"
            )
            if _enter_resource_wait_or_fail(
                c, state, step_mode=init_step_mode,
                resume_phase="conveyor_initialisation", owner="network",
                rnd=rnd, uid=uid, cur_xy=cur_xy,
            ):
                return
            if _is_enemy_replaceable_blocker_at(local_map, cur_xy[0], cur_xy[1]):
                _start_network_blocker_clearance(
                    state,
                    local_map,
                    cur_xy,
                    cur_xy,
                    "conveyor_initialisation",
                    "self_tile_enemy_blocker",
                )
                return
            state.phase = "network_plan_path"
            return

        _append_active_network_transport(state, nodes[idx], rnd)
        state.phase = "conveyor_execution"
        return

    if state.phase == "conveyor_execution":
        remaining = (len(nodes) - 1) - idx
        if remaining <= 1:
            state.phase = "conveyor_termination"
            return

        if _enemy_armoured_transport_on_tile(local_map, cur_xy[0], cur_xy[1]):
            _clear_network_target_state(state, clear_highway=False)
            state.phase = "network_select_candidate"
            return
        if _is_enemy_replaceable_blocker_at(local_map, cur_xy[0], cur_xy[1]):
            _start_network_blocker_clearance(
                state,
                local_map,
                cur_xy,
                cur_xy,
                "conveyor_execution",
                "self_tile_enemy_blocker",
            )
            return

        nxt = nodes[idx + 1]
        nxt2 = nodes[idx + 2]

        if _enemy_armoured_transport_on_tile(local_map, nxt[0], nxt[1]):
            _clear_network_target_state(state, clear_highway=False)
            state.phase = "network_select_candidate"
            return

        if _is_enemy_replaceable_blocker_at(local_map, nxt[0], nxt[1]):
            _start_network_blocker_clearance(
                state,
                local_map,
                cur_xy,
                nxt,
                "conveyor_execution",
                "enemy_transport_ahead",
            )
            return

        if _is_unit_halo_predicted_blocked(local_map, nxt[0], nxt[1], from_xy=cur_xy):
            state.highway_excluded_transport = _capture_active_network_transport_exclusions(
                state
            )
            state.network_escape_bridge_target = None
            _reset_network_path_state(state)
            state.phase = "network_bridge_escape_check"
            log_event(
                rnd,
                uid,
                "economy",
                f"({cur_xy[0]},{cur_xy[1]})",
                "economy_network_halo_block_highway_switch_v2",
                bx=nxt[0],
                by=nxt[1],
            )
            return

        if candidate.get("mode") != "highway_direct":
            direct_bridge_target = _select_direct_bridge_sink_target(
                state,
                local_map,
                cur_xy,
                nxt,
            )
            if isinstance(direct_bridge_target, tuple) and len(direct_bridge_target) == 2:
                _sync_dynamic_blocked_tiles(state, local_map)
                state.network_escape_bridge_target = direct_bridge_target
                _reset_network_path_state(state)
                state.phase = "network_bridge_escape_execute"
                log_event(
                    rnd,
                    uid,
                    "economy",
                    f"({cur_xy[0]},{cur_xy[1]})",
                    "economy_network_bridge_sink_terminate_v2",
                    tx=direct_bridge_target[0],
                    ty=direct_bridge_target[1],
                    source="conveyor_execution",
                )
                return

        if (
            candidate.get("mode") != "highway_direct"
            and _is_friendly_transport_tile(local_map, nxt[0], nxt[1])
        ):
            if not _destroy_standing_transport_for_highway_switch(
                c,
                local_map,
                cur_xy,
                rnd,
                uid,
                "conveyor_execution",
            ):
                return

            if _destroy_friendly_broken_transport_blocker(
                c,
                state,
                local_map,
                nxt,
                rnd,
                uid,
                "conveyor_execution",
            ):
                _reset_network_path_state(state)
                state.phase = "network_plan_path"
                return

            captured = set(_capture_active_network_transport_exclusions(state))
            captured.add((int(nxt[0]), int(nxt[1])))
            state.highway_excluded_transport = tuple(sorted(captured))
            state.network_escape_bridge_target = None
            _reset_network_path_state(state)
            state.phase = "network_bridge_escape_check"
            log_event(
                rnd,
                uid,
                "economy",
                f"({cur_xy[0]},{cur_xy[1]})",
                "economy_network_friendly_transport_highway_switch_v2",
                tx=nxt[0],
                ty=nxt[1],
            )
            return

        nxt_pos = Position(nxt[0], nxt[1])
        try:
            nxt_env = c.get_tile_env(nxt_pos)
        except GameError:
            nxt_env = None
        if nxt_env in (Environment.WALL, Environment.ORE_TITANIUM, Environment.ORE_AXIONITE):
            state.highway_excluded_transport = _capture_active_network_transport_exclusions(
                state
            )
            state.network_escape_bridge_target = None
            _reset_network_path_state(state)
            state.phase = "network_bridge_escape_check"
            log_event(
                rnd,
                uid,
                "economy",
                f"({cur_xy[0]},{cur_xy[1]})",
                "economy_network_hard_block_highway_switch_v2",
                bx=nxt[0],
                by=nxt[1],
            )
            return

        build_result = _build_network_segment_transport(
            c,
            nxt,
            nxt,
            nxt2,
            rnd,
            uid,
            state,
            conveyor_tag="economy_network_conveyor_step_v2",
            bridge_tag="economy_network_intermediate_bridge_step_v2",
        )
        if build_result == "wait_cd":
            return
        if build_result not in ("built", "already_built"):
            step_step_mode = (
                "bridge" if _is_diagonal_step(nxt, nxt2) else "conveyor"
            )
            if _enter_resource_wait_or_fail(
                c, state, step_mode=step_step_mode,
                resume_phase="conveyor_execution", owner="network",
                rnd=rnd, uid=uid, cur_xy=cur_xy,
            ):
                return
            if _is_enemy_replaceable_blocker_at(local_map, nxt[0], nxt[1]):
                _start_network_blocker_clearance(
                    state,
                    local_map,
                    cur_xy,
                    nxt,
                    "conveyor_execution",
                    "enemy_transport_ahead",
                )
                return
            state.phase = "network_plan_path"
            return

        _append_active_network_transport(state, nxt, rnd)
        move_result = _move_only_step(
            c,
            cur_xy,
            nxt,
            rnd,
            uid,
            "economy_network_move_step_v2",
        )
        if move_result == "moved":
            state.issued_move_last_tick = True
            state.expected_xy_after_move = nxt
            state.network_path_index += 1
            return
        if move_result == "blocked_hard":
            _reset_network_path_state(state)
            state.phase = "network_plan_path"
            return
        if (
            move_result == "blocked"
            and _is_unit_halo_predicted_blocked(
                local_map, nxt[0], nxt[1], from_xy=cur_xy
            )
        ):
            state.highway_excluded_transport = _capture_active_network_transport_exclusions(
                state
            )
            state.network_escape_bridge_target = None
            _reset_network_path_state(state)
            state.phase = "network_bridge_escape_check"
            log_event(
                rnd,
                uid,
                "economy",
                f"({cur_xy[0]},{cur_xy[1]})",
                "economy_network_halo_move_block_highway_switch_v2",
                bx=nxt[0],
                by=nxt[1],
            )
        return

    if state.phase == "conveyor_termination":
        if idx >= len(nodes) - 1:
            final_xy = nodes[idx]
        else:
            final_xy = nodes[idx + 1]

        if _enemy_armoured_transport_on_tile(local_map, final_xy[0], final_xy[1]):
            _clear_network_target_state(state, clear_highway=False)
            state.phase = "network_select_candidate"
            return

        if _is_enemy_replaceable_blocker_at(local_map, final_xy[0], final_xy[1]):
            _start_network_blocker_clearance(
                state,
                local_map,
                cur_xy,
                final_xy,
                "conveyor_termination",
                "enemy_transport_on_bridge_tile",
            )
            return

        bridge_result = _build_bridge_on_tile(
            c,
            final_xy,
            candidate["bridge_target"],
            rnd,
            uid,
            state=state,
        )
        if bridge_result == "wait_cd":
            return
        if bridge_result not in ("built", "already_built"):
            if _is_enemy_replaceable_blocker_at(local_map, final_xy[0], final_xy[1]):
                _start_network_blocker_clearance(
                    state,
                    local_map,
                    cur_xy,
                    final_xy,
                    "conveyor_termination",
                    "enemy_transport_on_bridge_tile",
                )
                return
            _clear_network_target_state(state, clear_highway=False)
            state.phase = "network_select_candidate"
            return

        _record_active_network_terminal_bridge(
            state,
            final_xy,
            candidate["bridge_target"],
            rnd,
        )

        if _deprioritize_successful_direct_anchor(state, candidate):
            log_event(
                rnd,
                uid,
                "economy",
                f"({cur_xy[0]},{cur_xy[1]})",
                "economy_network_direct_anchor_deprioritized_v2",
                bx=candidate["bridge_pos"][0],
                by=candidate["bridge_pos"][1],
            )

        _refresh_friendly_network_registry(state, local_map, rnd)
        _clear_network_target_state(state, clear_highway=True)
        state.active_network_id = None
        if known_ti:
            state.phase = "harvest_pick_ore"
        else:
            _resume_exploration_nearest_waypoint(state, cur_xy)
        log_event(
            rnd,
            uid,
            "economy",
            f"({cur_xy[0]},{cur_xy[1]})",
            "economy_network_complete_resume_v2",
            tx=candidate["bridge_target"][0],
            ty=candidate["bridge_target"][1],
            known_networks=len(state.network_records),
        )
        return


def _clear_network_target_state(
    state: EconomyState,
    clear_highway: bool,
    preserve_highway_excluded: bool = False,
):
    state.network_target = None
    _reset_network_path_state(state)
    state.network_escape_bridge_target = None
    if clear_highway:
        state.network_highway_pending_harvester = None
        state.network_highway_active_harvester = None
    if clear_highway or not preserve_highway_excluded:
        state.highway_excluded_transport = ()


def _reset_network_path_state(state: EconomyState):
    state.network_path_nodes = ()
    state.network_path_index = 0
    state.network_plan_session = None


def _capture_active_network_transport_exclusions(state: EconomyState):
    excluded = set()

    active_id = state.active_network_id
    if isinstance(active_id, int):
        rec = state.network_records.get(active_id)
        if isinstance(rec, dict):
            for p in rec.get("transport_chain_positions", ()):
                if isinstance(p, tuple) and len(p) == 2:
                    excluded.add((int(p[0]), int(p[1])))

    nodes = state.network_path_nodes
    upto = min(len(nodes), max(0, state.network_path_index + 1))
    for i in range(upto):
        x, y = nodes[i]
        excluded.add((int(x), int(y)))

    candidate = state.network_target
    if isinstance(candidate, dict):
        bridge_pos = candidate.get("bridge_pos")
        if isinstance(bridge_pos, tuple) and len(bridge_pos) == 2:
            excluded.add((int(bridge_pos[0]), int(bridge_pos[1])))

        source_conveyor = candidate.get("source_conveyor")
        if isinstance(source_conveyor, tuple) and len(source_conveyor) == 2:
            excluded.add((int(source_conveyor[0]), int(source_conveyor[1])))

    return tuple(sorted(excluded))


def _start_network_attack(
    state: EconomyState,
    target_xy,
    resume_phase: str,
    reason: str,
    target_queue=None,
    return_xy=None,
):
    norm_targets = []
    seen = set()
    raw_targets = target_queue if target_queue is not None else (target_xy,)
    for item in raw_targets:
        if not (isinstance(item, tuple) and len(item) == 2):
            continue
        txy = (int(item[0]), int(item[1]))
        if txy in seen:
            continue
        seen.add(txy)
        norm_targets.append(txy)

    if not norm_targets:
        norm_targets.append((int(target_xy[0]), int(target_xy[1])))

    state.attack_targets = tuple(norm_targets)
    state.attack_target_index = 0
    state.attack_target_xy = state.attack_targets[0]
    if isinstance(return_xy, tuple) and len(return_xy) == 2:
        state.attack_return_xy = (int(return_xy[0]), int(return_xy[1]))
    else:
        state.attack_return_xy = None
    state.attack_resume_phase = resume_phase
    state.attack_reason = reason
    state.phase = "network_attack_blocker"


def _clear_network_attack_substate(state: EconomyState):
    state.attack_target_xy = None
    state.attack_targets = ()
    state.attack_target_index = 0
    state.attack_return_xy = None
    state.attack_resume_phase = None
    state.attack_reason = None


def _is_enemy_directional_transport_tile(local_map, x: int, y: int) -> bool:
    rec = _known_building_at(local_map, x, y)
    if not isinstance(rec, dict):
        return False
    if rec.get("team") == getattr(local_map, "my_team", None):
        return False
    return rec.get("entity_type") in _DIRECTIONAL_TRANSPORT_ENTITY_TYPES


def _network_enemy_conveyor_target_queue(local_map, blocker_xy):
    bx, by = int(blocker_xy[0]), int(blocker_xy[1])
    targets = [(bx, by)]

    blocker_rec = _known_building_at(local_map, bx, by)
    if not isinstance(blocker_rec, dict):
        return tuple(targets)
    if blocker_rec.get("team") == getattr(local_map, "my_team", None):
        return tuple(targets)
    if blocker_rec.get("entity_type") not in _DIRECTIONAL_TRANSPORT_ENTITY_TYPES:
        return tuple(targets)

    out_dir = blocker_rec.get("direction")
    if out_dir is not None:
        ddx, ddy = out_dir.delta()
        out_xy = (bx + ddx, by + ddy)
        if _is_enemy_directional_transport_tile(local_map, out_xy[0], out_xy[1]):
            targets.append(out_xy)

    incoming = []
    for dx, dy in CARDINAL_DELTAS:
        nx = bx + dx
        ny = by + dy
        if not _is_enemy_directional_transport_tile(local_map, nx, ny):
            continue
        rec = _known_building_at(local_map, nx, ny)
        if not isinstance(rec, dict):
            continue
        direction = rec.get("direction")
        if direction is None:
            continue
        ddx, ddy = direction.delta()
        if (nx + ddx, ny + ddy) == (bx, by):
            incoming.append((nx, ny))

    if incoming:
        incoming.sort(key=lambda p: (p[0], p[1]))
        targets.append(incoming[0])

    deduped = []
    seen = set()
    for tx, ty in targets:
        txy = (int(tx), int(ty))
        if txy in seen:
            continue
        seen.add(txy)
        deduped.append(txy)
    return tuple(deduped)


def _start_network_blocker_clearance(
    state: EconomyState,
    local_map,
    cur_xy,
    blocker_xy,
    resume_phase: str,
    reason: str,
):
    targets = _network_enemy_conveyor_target_queue(local_map, blocker_xy)
    if not targets:
        targets = ((int(blocker_xy[0]), int(blocker_xy[1])),)
    _start_network_attack(
        state,
        targets[0],
        resume_phase,
        reason,
        target_queue=targets,
        return_xy=cur_xy,
    )


def _run_network_attack_blocker(
    c: Controller,
    state: EconomyState,
    local_map,
    cur_xy,
    rnd: int,
    uid: int,
):
    target_xy = state.attack_target_xy
    resume_phase = state.attack_resume_phase or "conveyor_initialisation"
    targets = tuple(state.attack_targets)
    if not targets and isinstance(target_xy, tuple) and len(target_xy) == 2:
        targets = ((int(target_xy[0]), int(target_xy[1])),)

    if not targets:
        state.phase = resume_phase
        _clear_network_attack_substate(state)
        return

    idx = int(state.attack_target_index)
    if idx < 0:
        idx = 0

    while idx < len(targets):
        tx, ty = targets[idx]
        if _is_enemy_replaceable_blocker_at(local_map, tx, ty):
            break
        idx += 1

    state.attack_target_index = idx

    if idx >= len(targets):
        return_xy = state.attack_return_xy
        if isinstance(return_xy, tuple) and len(return_xy) == 2 and cur_xy != return_xy:
            if c.get_move_cooldown() > 0:
                return

            def passable_fn(x: int, y: int) -> bool:
                return _is_general_movement_passable(local_map, x, y, return_xy)

            steps = _astar_cardinal_plan(
                local_map,
                cur_xy,
                return_xy,
                max_expansions=256,
                tile_passable_fn=passable_fn,
                max_time_us=1200,
            )
            if not steps:
                steps = _astar_cardinal_plan(
                    local_map,
                    cur_xy,
                    return_xy,
                    max_expansions=256,
                    tile_passable_fn=lambda x, y: _is_general_movement_passable(
                        local_map,
                        x,
                        y,
                        return_xy,
                        respect_halo=False,
                    ),
                    max_time_us=1200,
                )
            if not steps:
                log_event(
                    rnd,
                    uid,
                    "economy",
                    f"({cur_xy[0]},{cur_xy[1]})",
                    "economy_network_attack_return_skipped_v2",
                    reason=state.attack_reason,
                    rx=return_xy[0],
                    ry=return_xy[1],
                    detail="unreachable",
                )
                state.phase = resume_phase
                _clear_network_attack_substate(state)
                return

            nxt = steps[0]
            move_result = _execute_step_toward(
                c, local_map, cur_xy, nxt, rnd, uid)
            if move_result == "moved":
                state.issued_move_last_tick = True
                state.expected_xy_after_move = nxt
                return
            if move_result in ("wait_cd", "built"):
                return

            log_event(
                rnd,
                uid,
                "economy",
                f"({cur_xy[0]},{cur_xy[1]})",
                "economy_network_attack_return_skipped_v2",
                reason=state.attack_reason,
                rx=return_xy[0],
                ry=return_xy[1],
                detail=move_result,
            )
            state.phase = resume_phase
            _clear_network_attack_substate(state)
            return

        log_event(
            rnd,
            uid,
            "economy",
            f"({cur_xy[0]},{cur_xy[1]})",
            "economy_network_attack_cleared_v2",
            reason=state.attack_reason,
            targets=len(targets),
        )
        state.phase = resume_phase
        _clear_network_attack_substate(state)
        return

    tx, ty = targets[idx]
    state.attack_target_xy = (tx, ty)

    if cur_xy != (tx, ty):
        if c.get_move_cooldown() > 0:
            return

        def passable_fn(x: int, y: int) -> bool:
            return _is_general_movement_passable(local_map, x, y, (tx, ty))

        steps = _astar_cardinal_plan(
            local_map,
            cur_xy,
            (tx, ty),
            max_expansions=256,
            tile_passable_fn=passable_fn,
            max_time_us=1200,
        )
        if not steps:
            steps = _astar_cardinal_plan(
                local_map,
                cur_xy,
                (tx, ty),
                max_expansions=256,
                tile_passable_fn=lambda x, y: _is_general_movement_passable(
                    local_map,
                    x,
                    y,
                    (tx, ty),
                    respect_halo=False,
                ),
                max_time_us=1200,
            )
        if not steps:
            state.attack_target_index = idx + 1
            log_event(
                rnd,
                uid,
                "economy",
                f"({cur_xy[0]},{cur_xy[1]})",
                "economy_network_attack_target_skipped_v2",
                reason=state.attack_reason,
                tx=tx,
                ty=ty,
                idx=idx,
                detail="unreachable",
            )
            return

        nxt = steps[0]
        move_result = _execute_step_toward(c, local_map, cur_xy, nxt, rnd, uid)
        if move_result == "moved":
            state.issued_move_last_tick = True
            state.expected_xy_after_move = nxt
            return
        if move_result in ("wait_cd", "built"):
            return

        state.attack_target_index = idx + 1
        log_event(
            rnd,
            uid,
            "economy",
            f"({cur_xy[0]},{cur_xy[1]})",
            "economy_network_attack_target_skipped_v2",
            reason=state.attack_reason,
            tx=tx,
            ty=ty,
            idx=idx,
            detail=move_result,
        )
        return

    if c.get_action_cooldown() > 0:
        return

    cur_pos = Position(cur_xy[0], cur_xy[1])
    fired = False
    try:
        if c.can_fire(cur_pos):
            c.fire(cur_pos)
            fired = True
            log_event(
                rnd,
                uid,
                "economy",
                f"({cur_xy[0]},{cur_xy[1]})",
                "economy_network_attack_tick_v2",
                tx=tx,
                ty=ty,
                idx=idx,
                reason=state.attack_reason,
            )
    except GameError:
        pass

    if not fired:
        state.attack_target_index = idx + 1
        log_event(
            rnd,
            uid,
            "economy",
            f"({cur_xy[0]},{cur_xy[1]})",
            "economy_network_attack_target_skipped_v2",
            reason=state.attack_reason,
            tx=tx,
            ty=ty,
            idx=idx,
            detail="not_fireable",
        )

def _tile_owned_by_excluded_network(
    tile_xy,
    excluded_network_ids,
    tile_owner_index,
) -> bool:
    if not excluded_network_ids:
        return False
    if not isinstance(tile_owner_index, dict):
        return False

    tx = int(tile_xy[0])
    ty = int(tile_xy[1])
    owners = tile_owner_index.get((tx, ty))
    if not owners:
        return False

    for owner in owners:
        if owner in excluded_network_ids:
            return True
    return False


def _direct_anchor_reachable_under_congestion_mask(
    local_map,
    start_xy,
    goal_xy,
    excluded_positions,
    excluded_network_ids,
    tile_owner_index,
) -> bool:
    if not local_map.in_bounds(goal_xy[0], goal_xy[1]):
        return False
    if goal_xy in excluded_positions:
        return False
    if _tile_owned_by_excluded_network(
        goal_xy,
        excluded_network_ids,
        tile_owner_index,
    ):
        return False
    if start_xy == goal_xy:
        return True

    def passable_fn(x: int, y: int) -> bool:
        return _is_conveyor_planner_passable(
            local_map,
            x,
            y,
            goal_xy,
            exclude_positions=excluded_positions,
            exclude_network_ids=excluded_network_ids,
            tile_owner_index=tile_owner_index,
        )

    probe_steps = _astar_cardinal_plan(
        local_map,
        start_xy,
        goal_xy,
        max_expansions=_DIRECT_MASK_PREFILTER_MAX_EXPANSIONS,
        tile_passable_fn=passable_fn,
        max_time_us=_DIRECT_MASK_PREFILTER_TIMEOUT_US,
    )
    return bool(probe_steps)


def _collect_direct_anchor_candidates(state: EconomyState, cur_xy, rnd: int):
    _prune_unreachable_direct_deferrals(state, rnd)

    invalid_positions = {
        (int(p[0]), int(p[1]))
        for p in state.network_invalid_bridge_positions
        if isinstance(p, tuple) and len(p) == 2
    }

    cx, cy = state.core_xy
    core_tiles = [
        (cx + dx, cy + dy)
        for dx in (-1, 0, 1)
        for dy in (-1, 0, 1)
    ]

    avail = state.direct_anchor_available
    # The last element of avail is the deprioritized (most-recently-used) anchor.
    deprioritized = None
    if avail:
        last = avail[-1]
        if isinstance(last, tuple) and len(last) == 2:
            deprioritized = (int(last[0]), int(last[1]))

    deferred = state.network_unreachable_direct_until
    px, py = int(cur_xy[0]), int(cur_xy[1])

    normal = []
    deprior_entry = None

    for bx, by in avail:
        if (bx, by) in invalid_positions:
            continue
        hold_until = deferred.get((bx, by))
        if isinstance(hold_until, int) and hold_until > rnd:
            continue
        tx, ty = _nearest_core_tile_target((bx, by), core_tiles)
        if ((tx - bx) * (tx - bx) + (ty - by) * (ty - by)) > 9:
            continue
        dx, dy = bx - px, by - py
        dist_sq = dx * dx + dy * dy
        entry = (dist_sq, bx, by, tx, ty)
        if deprioritized is not None and (bx, by) == deprioritized:
            deprior_entry = entry
        else:
            normal.append(entry)

    normal.sort(key=lambda t: t[0])
    if deprior_entry is not None:
        normal.append(deprior_entry)
    return tuple(normal)


def _select_direct_bridge_candidate(state: EconomyState, _local_map, cur_xy, rnd: int):
    candidates = _collect_direct_anchor_candidates(state, cur_xy, rnd)
    if not candidates:
        return None

    for _, bx, by, tx, ty in candidates:
        return {
            "mode": "direct",
            "bridge_pos": (bx, by),
            "bridge_target": (tx, ty),
            "source_network_key": None,
            "source_conveyor": None,
        }

    return None


def _deprioritize_successful_direct_anchor(state: EconomyState, candidate) -> bool:
    if not isinstance(candidate, dict):
        return False
    if str(candidate.get("mode") or "") != "direct":
        return False

    bridge_pos = candidate.get("bridge_pos")
    if not (isinstance(bridge_pos, tuple) and len(bridge_pos) == 2):
        return False

    anchor = (int(bridge_pos[0]), int(bridge_pos[1]))
    ordered = []
    for p in state.direct_anchor_available:
        if not (isinstance(p, tuple) and len(p) == 2):
            continue
        norm = (int(p[0]), int(p[1]))
        if norm not in ordered:
            ordered.append(norm)

    if anchor not in ordered:
        return False

    ordered = [p for p in ordered if p != anchor]
    ordered.append(anchor)
    state.direct_anchor_available = tuple(ordered)
    return True


def _prune_unreachable_direct_deferrals(state: EconomyState, rnd: int):
    pending = state.network_unreachable_direct_until
    if not isinstance(pending, dict) or not pending:
        return

    for anchor_xy, hold_until in tuple(pending.items()):
        if not isinstance(hold_until, int) or hold_until <= rnd:
            pending.pop(anchor_xy, None)


def _defer_unreachable_direct_candidate(state: EconomyState, bridge_pos, rnd: int):
    _ = rnd
    if not (isinstance(bridge_pos, tuple) and len(bridge_pos) == 2):
        return False

    bx = int(bridge_pos[0])
    by = int(bridge_pos[1])
    anchor = (bx, by)
    state.network_unreachable_direct_until.pop(anchor, None)
    state.network_invalid_bridge_positions.add(anchor)
    return True


def _select_lidar_bridge_candidate(
    state: EconomyState,
    local_map,
    cur_xy,
    rnd=None,
    uid=None,
):
    mask_relaxed = False

    endpoint_xy = _first_lidar_transport_hit(
        state,
        local_map,
        cur_xy,
        rnd,
        uid,
        blocked_tiles=(),
        blocked_networks=(),
        tile_owners=None,
    )

    if endpoint_xy is None:
        return None

    known_targets = _known_friendly_bridge_targets(local_map)
    if not _bridge_target_region_clear(endpoint_xy, known_targets, radius_cheb=1):
        original_endpoint = endpoint_xy
        endpoint_xy = _nearest_uncongested_transport_target(
            state,
            local_map,
            endpoint_xy,
            known_targets,
        )
        if isinstance(rnd, int) and isinstance(uid, int):
            adjusted_tx = -1
            adjusted_ty = -1
            if isinstance(endpoint_xy, tuple) and len(endpoint_xy) == 2:
                adjusted_tx = int(endpoint_xy[0])
                adjusted_ty = int(endpoint_xy[1])

            log_event(
                rnd,
                uid,
                "economy",
                f"({cur_xy[0]},{cur_xy[1]})",
                "economy_network_lidar_target_adjust_v2",
                from_tx=original_endpoint[0],
                from_ty=original_endpoint[1],
                to_tx=adjusted_tx,
                to_ty=adjusted_ty,
            )
        if endpoint_xy is None:
            return None

    bridge_xy = None
    cur_override = False
    cx = int(cur_xy[0])
    cy = int(cur_xy[1])
    tx = int(endpoint_xy[0])
    ty = int(endpoint_xy[1])
    if (
        (cx, cy) != (tx, ty)
        and ((tx - cx) * (tx - cx) + (ty - cy) * (ty - cy)) <= 9
        and _is_bridge_build_tile_viable(local_map, cx, cy)
    ):
        # In lidar mode, prefer stamping the bridge on the current tile first.
        # This avoids move->highway-switch loops when adjacent tiles are congested.
        bridge_xy = (cx, cy)
        cur_override = True

    if bridge_xy is None:
        bridge_xy = _pick_bridge_build_tile_for_target(
            state,
            local_map,
            cur_xy,
            endpoint_xy,
        )

    if bridge_xy is None:
        if isinstance(rnd, int) and isinstance(uid, int):
            log_event(
                rnd,
                uid,
                "economy",
                f"({cur_xy[0]},{cur_xy[1]})",
                "economy_network_lidar_bridge_pick_failed_v2",
                tx=endpoint_xy[0],
                ty=endpoint_xy[1],
                mask_relaxed=1 if mask_relaxed else 0,
                cur_override=1 if cur_override else 0,
            )
        return None

    if isinstance(rnd, int) and isinstance(uid, int):
        log_event(
            rnd,
            uid,
            "economy",
            f"({cur_xy[0]},{cur_xy[1]})",
            "economy_network_lidar_candidate_v2",
            bx=bridge_xy[0],
            by=bridge_xy[1],
            tx=endpoint_xy[0],
            ty=endpoint_xy[1],
            mask_relaxed=1 if mask_relaxed else 0,
            cur_override=1 if cur_override else 0,
        )

    return {
        "mode": "lidar_indirect",
        "bridge_pos": bridge_xy,
        "bridge_target": endpoint_xy,
        "source_network_key": None,
        "source_conveyor": endpoint_xy,
    }


def _first_lidar_transport_hit(
    state: EconomyState,
    local_map,
    origin_xy,
    rnd=None,
    uid=None,
    blocked_tiles=(),
    blocked_networks=(),
    tile_owners=None,
):
    ox, oy = origin_xy
    my_team = getattr(local_map, "my_team", None)

    active_transport_positions = set()
    active_id = getattr(state, "active_network_id", None)
    if isinstance(active_id, int):
        rec = state.network_records.get(active_id)
        if isinstance(rec, dict):
            for p in rec.get("transport_chain_positions", ()):
                if isinstance(p, tuple) and len(p) == 2:
                    active_transport_positions.add((int(p[0]), int(p[1])))

    if isinstance(rnd, int) and isinstance(uid, int):
        log_event(
            rnd,
            uid,
            "economy",
            f"({ox},{oy})",
            "economy_network_lidar_scan_start_v2",
            rays=len(_LIDAR_RAYS_8),
            active_excluded=len(active_transport_positions),
        )

    best = None
    best_key = None
    scan_hits = []
    hit_count = 0
    blocked_hit_count = 0

    blocked_tiles_set = set(blocked_tiles)
    blocked_networks_set = set(blocked_networks)
    for ray_rank, (dx, dy) in enumerate(_LIDAR_RAYS_8):
        step = 0
        x = ox
        y = oy
        while step < 64:
            step += 1
            x += dx
            y += dy
            if not local_map.in_bounds(x, y):
                break

            rec = _known_building_at(local_map, x, y)
            if isinstance(rec, dict):
                if (
                    rec.get("team") == my_team
                    and rec.get("entity_type") in _TRANSPORT_ENTITY_TYPES
                ):
                    if (x, y) in active_transport_positions:
                        continue
                    if (x, y) in blocked_tiles_set:
                        blocked_hit_count += 1
                        continue
                    if tile_owners is not None and blocked_networks_set:
                        owners = tile_owners.get((x, y), set())
                        if any(owner in blocked_networks_set for owner in owners):
                            blocked_hit_count += 1
                            continue
                    key = (step, ray_rank, x, y)
                    hit_count += 1
                    if len(scan_hits) < _LIDAR_DEBUG_MAX_HITS_PER_SCAN:
                        scan_hits.append(f"r{ray_rank}:s{step}@({x},{y})")
                    if best_key is None or key < best_key:
                        best_key = key
                        best = (x, y)
                    break

    if isinstance(rnd, int) and isinstance(uid, int):
        log_event(
            rnd,
            uid,
            "economy",
            f"({ox},{oy})",
            "economy_network_lidar_scan_result_v2",
            hit_count=hit_count,
            blocked_hits=blocked_hit_count,
            best_x=(best[0] if isinstance(best, tuple)
                    and len(best) == 2 else -1),
            best_y=(best[1] if isinstance(best, tuple)
                    and len(best) == 2 else -1),
            hits=";".join(scan_hits),
        )

    return best


def _nearest_uncongested_transport_target(
    state: EconomyState,
    local_map,
    endpoint_xy,
    known_targets,
    blocked_tiles=(),
    blocked_networks=(),
    tile_owners=None,
):
    my_team = getattr(local_map, "my_team", None)
    ex, ey = endpoint_xy

    best = None
    best_key = None
    blocked_tiles_set = set(blocked_tiles)
    blocked_networks_set = set(blocked_networks)
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
        if rec.get("entity_type") not in _TRANSPORT_ENTITY_TYPES:
            continue

        pos = rec.get("position")
        if not (isinstance(pos, tuple) and len(pos) == 2):
            continue
        tx = int(pos[0])
        ty = int(pos[1])
        if max(abs(tx - ex), abs(ty - ey)) > 1:
            continue
        if (tx, ty) in blocked_tiles_set:
            continue
        if tile_owners is not None and blocked_networks_set:
            owners = tile_owners.get((tx, ty), set())
            if any(owner in blocked_networks_set for owner in owners):
                continue
        if not _bridge_target_region_clear((tx, ty), known_targets, radius_cheb=1):
            continue

        key = (
            abs(tx - ex) + abs(ty - ey),
            _planner_tile_soft_cost(state, tx, ty),
            tx,
            ty,
        )
        if best_key is None or key < best_key:
            best_key = key
            best = (tx, ty)

    return best


def _pick_bridge_build_tile_for_target(
    state: EconomyState,
    local_map,
    cur_xy,
    target_xy,
    blocked_tiles=(),
    blocked_networks=(),
    tile_owners=None,
):
    tx, ty = target_xy
    blocked_tiles_set = set(blocked_tiles)
    blocked_networks_set = set(blocked_networks)
    best = None
    best_key = None
    for dx, dy in _BRIDGE_TARGET_OFFSETS:
        bx = tx - dx
        by = ty - dy
        if not local_map.in_bounds(bx, by):
            continue
        if not _tile_is_known(local_map, bx, by):
            continue

        tile = local_map.get(bx, by)
        if tile in (MAP_OBSTACLE, MAP_ORE_TITANIUM, MAP_ORE_AXIONITE):
            continue

        rec = _known_building_at(local_map, bx, by)
        if isinstance(rec, dict):
            etype = rec.get("entity_type")
            team = rec.get("team")
            if etype not in (EntityType.ROAD, EntityType.MARKER):
                if not (
                    team == getattr(local_map, "my_team", None)
                    and etype in _TRANSPORT_ENTITY_TYPES
                ):
                    continue

        if (bx, by) in blocked_tiles_set:
            continue
        if tile_owners is not None and blocked_networks_set:
            owners = tile_owners.get((bx, by), set())
            if any(owner in blocked_networks_set for owner in owners):
                continue

        key = (
            _manhattan(cur_xy, (bx, by)) +
            (_planner_tile_soft_cost(state, bx, by) * 2),
            abs(bx - tx) + abs(by - ty),
            bx,
            by,
        )
        if best_key is None or key < best_key:
            best_key = key
            best = (bx, by)

    return best


def _recompute_direct_anchor_availability(state: EconomyState, local_map):
    cx, cy = state.core_xy
    available = []
    blocked = set()
    invalid_positions = {
        (int(p[0]), int(p[1]))
        for p in state.network_invalid_bridge_positions
        if isinstance(p, tuple) and len(p) == 2
    }

    for bx, by in _core_bridge_anchor_candidates(cx, cy):
        if (bx, by) in invalid_positions:
            blocked.add((bx, by))
            continue

        if not local_map.in_bounds(bx, by):
            blocked.add((bx, by))
            continue

        # Unknown tiles stay available until contradicted by fresh vision.
        if not _tile_is_known(local_map, bx, by):
            available.append((bx, by))
            continue

        tile = local_map.get(bx, by)
        if tile in (MAP_OBSTACLE, MAP_ORE_TITANIUM, MAP_ORE_AXIONITE):
            blocked.add((bx, by))
            continue

        rec = _known_building_at(local_map, bx, by)
        if isinstance(rec, dict):
            etype = rec.get("entity_type")
            if etype not in (
                EntityType.ROAD,
                EntityType.MARKER,
                EntityType.CONVEYOR,
                EntityType.SPLITTER,
                EntityType.ARMOURED_CONVEYOR,
                EntityType.BRIDGE,
            ):
                blocked.add((bx, by))
                continue

        available.append((bx, by))

    available_set = set(available)
    ordered = []
    for p in state.direct_anchor_available:
        if not (isinstance(p, tuple) and len(p) == 2):
            continue
        norm = (int(p[0]), int(p[1]))
        if norm in available_set and norm not in ordered:
            ordered.append(norm)

    for p in sorted(available):
        if p not in ordered:
            ordered.append(p)

    state.direct_anchor_available = tuple(ordered)
    state.direct_anchor_blocked = blocked


def _refresh_entity_ownership_db(state: EconomyState, local_map, rnd: int):
    if state.entity_db_round == rnd:
        return
    state.entity_db_round = rnd

    built_transport_ids = set()
    external_transport_ids = set()
    stolen_transport_ids = {
        int(i)
        for i in state.transport_ids_stolen
        if isinstance(i, int)
    }
    stolen_transport_positions = {
        (int(p[0]), int(p[1]))
        for p in state.transport_stolen_positions
        if isinstance(p, tuple) and len(p) == 2
    }

    built_harvester_ids = set()
    external_harvester_ids = set()
    stolen_harvester_ids = {
        int(i)
        for i in state.harvester_ids_stolen
        if isinstance(i, int)
    }
    stolen_harvester_positions = {
        (int(p[0]), int(p[1]))
        for p in state.harvester_stolen_positions
        if isinstance(p, tuple) and len(p) == 2
    }

    entities = getattr(local_map, "entities", None)
    my_team = getattr(local_map, "my_team", None)
    if isinstance(entities, dict):
        for rec in entities.values():
            if not isinstance(rec, dict):
                continue
            if not rec.get("alive", False):
                continue

            etype = rec.get("entity_type")
            team = rec.get("team")
            entity_id = _entity_id_from_rec(rec)
            pos = rec.get("position")
            if isinstance(pos, tuple) and len(pos) == 2:
                pxy = (int(pos[0]), int(pos[1]))
            else:
                pxy = None

            if etype in _TRANSPORT_ENTITY_TYPES:
                built_by_me = (
                    (isinstance(entity_id, int)
                     and entity_id in state.built_entity_ids)
                    or (pxy in state.built_transport_positions)
                )

                if built_by_me:
                    if isinstance(entity_id, int):
                        built_transport_ids.add(entity_id)
                    continue

                if team == my_team:
                    if pxy is not None and pxy in stolen_transport_positions:
                        if isinstance(entity_id, int):
                            stolen_transport_ids.add(entity_id)
                            external_transport_ids.add(entity_id)
                        continue

                    if isinstance(entity_id, int):
                        external_transport_ids.add(entity_id)
                    continue

                if isinstance(entity_id, int) and entity_id in stolen_transport_ids:
                    external_transport_ids.add(entity_id)
                    if pxy is not None:
                        stolen_transport_positions.add(pxy)
                continue

            if etype == EntityType.HARVESTER:
                built_by_me = (
                    (isinstance(entity_id, int)
                     and entity_id in state.built_entity_ids)
                    or (pxy in state.built_harvester_positions)
                )

                if built_by_me:
                    if isinstance(entity_id, int):
                        built_harvester_ids.add(entity_id)
                    continue

                if team == my_team:
                    if isinstance(entity_id, int):
                        external_harvester_ids.add(entity_id)
                    continue

                if isinstance(entity_id, int) and entity_id in stolen_harvester_ids:
                    external_harvester_ids.add(entity_id)
                    if pxy is not None:
                        stolen_harvester_positions.add(pxy)
                    continue

                if pxy is not None and pxy in stolen_harvester_positions:
                    if isinstance(entity_id, int):
                        stolen_harvester_ids.add(entity_id)
                        external_harvester_ids.add(entity_id)

    state.transport_ids_built = built_transport_ids
    state.transport_ids_external_friendly = external_transport_ids
    state.transport_ids_stolen = stolen_transport_ids
    state.transport_ids_friendly = (
        built_transport_ids | external_transport_ids | stolen_transport_ids
    )
    state.transport_stolen_positions = stolen_transport_positions

    state.harvester_ids_built = built_harvester_ids
    state.harvester_ids_external_friendly = external_harvester_ids
    state.harvester_ids_stolen = stolen_harvester_ids
    state.harvester_ids_friendly = (
        built_harvester_ids | external_harvester_ids | stolen_harvester_ids
    )
    state.harvester_stolen_positions = stolen_harvester_positions


def _refresh_friendly_network_registry(state: EconomyState, local_map, rnd: int):
    if state.last_registry_round == rnd:
        return
    state.last_registry_round = rnd

    my_team = getattr(local_map, "my_team", None)
    stolen_harvester_ids = set(state.harvester_ids_stolen)
    stolen_harvester_positions = set(state.harvester_stolen_positions)
    observed_ids = set()

    # ---- Axionite networks ----
    # An axionite network is rooted at a foundry. The foundry's output bridge
    # (a bridge placed on a cardinal neighbour of the foundry, targeting a
    # tile within r²≤9) delivers refined axionite to the "root" tile, from
    # which a conveyor chain runs to a terminal bridge near the core. We
    # trace each friendly foundry so that other bots' planners — which iterate
    # state.network_records to find splice targets — can splice into axionite
    # networks the same way they splice into titanium networks, even when
    # a sibling bot built it (visible through local_map.entities via vision).
    friendly_foundries = getattr(local_map, "friendly_foundries", None)
    if isinstance(friendly_foundries, set):
        for fx, fy in sorted(friendly_foundries):
            foundry_rec = _known_building_at(local_map, fx, fy)
            if not isinstance(foundry_rec, dict):
                continue
            if foundry_rec.get("entity_type") != EntityType.FOUNDRY:
                continue
            if foundry_rec.get("team") != my_team:
                continue

            # Locate the foundry's output bridge among cardinal neighbours.
            # Any friendly bridge on a cardinal tile qualifies; its target is
            # the "root" position of the axionite network's transport chain.
            ax_traces = []
            for ddx, ddy in CARDINAL_DELTAS:
                bx, by = fx + ddx, fy + ddy
                brec = _known_building_at(local_map, bx, by)
                if not isinstance(brec, dict):
                    continue
                if brec.get("entity_type") != EntityType.BRIDGE:
                    continue
                if brec.get("team") != my_team:
                    continue
                btarget = brec.get("bridge_target")
                if not (isinstance(btarget, tuple) and len(btarget) == 2):
                    continue
                root_xy = (int(btarget[0]), int(btarget[1]))
                # If the bridge targets the foundry itself, it's an input
                # bridge (titanium or raw axionite feeder), not an output.
                if root_xy == (fx, fy):
                    continue
                trace = _trace_friendly_transport_chain_partial(
                    local_map, root_xy)
                if trace is None:
                    continue
                trace["axionite_source_foundry"] = (int(fx), int(fy))
                trace["axionite_root_xy"] = root_xy
                trace["axionite_output_bridge"] = (int(bx), int(by))
                ax_traces.append(trace)

            if not ax_traces:
                continue

            complete = [t for t in ax_traces if t.get("complete", False)]
            if complete:
                best_trace = min(
                    complete,
                    key=lambda t: (
                        len(t["chain"]),
                        t["terminal_bridge"][0],
                        t["terminal_bridge"][1],
                    ),
                )
            else:
                best_trace = max(
                    ax_traces,
                    key=lambda t: (
                        len(t["chain"]),
                        t["chain"][-1][0] if t["chain"] else 0,
                        t["chain"][-1][1] if t["chain"] else 0,
                    ),
                )

            # Reuse existing registry keyed by terminal bridge position when
            # available; otherwise synthesise a fresh id. Store under the
            # foundry position in network_id_by_harvester so existing splice
            # lookups (which treat it as an opaque source tile) find it.
            network_id = None
            terminal = best_trace.get("terminal_bridge")
            if isinstance(terminal, tuple) and len(terminal) == 2:
                network_id = state.network_id_by_terminal.get(terminal)
            if network_id is None:
                network_id = state.network_id_by_harvester.get((int(fx), int(fy)))
            if network_id is None:
                network_id = state.next_network_id
                state.next_network_id += 1

            # Feed the axionite trace into the same recorder used for
            # titanium networks. The "harvester" slot holds the foundry's
            # position here — a slight naming mismatch but the downstream
            # splice logic only cares about tile identity.
            obs = {
                "harvester": (int(fx), int(fy)),
                "chain": best_trace.get("chain", ()),
                "conveyor_memory": best_trace.get("conveyor_memory", ()),
                "complete": best_trace.get("complete", False),
                "terminal_bridge": best_trace.get("terminal_bridge"),
                "bridge_target": best_trace.get("bridge_target"),
                "source_type": "axionite",
                "axionite_root_xy": best_trace.get("axionite_root_xy"),
                "axionite_output_bridge": best_trace.get("axionite_output_bridge"),
            }
            _record_network_observation(state, network_id, obs, rnd)
            observed_ids.add(network_id)

    for hx, hy in sorted(getattr(local_map, "titanium_harvesters", set())):
        harvester_rec = _known_building_at(local_map, hx, hy)
        if not isinstance(harvester_rec, dict):
            continue
        if harvester_rec.get("entity_type") != EntityType.HARVESTER:
            continue

        harvester_id = _entity_id_from_rec(harvester_rec)
        trusted_stolen = (
            (isinstance(harvester_id, int) and harvester_id in stolen_harvester_ids)
            or ((hx, hy) in stolen_harvester_positions)
        )
        if harvester_rec.get("team") != my_team and not trusted_stolen:
            continue

        traces = []
        for dx, dy in _ADJACENT_DELTAS_8:
            start_xy = (hx + dx, hy + dy)
            trace = _trace_friendly_transport_chain_partial(
                local_map, start_xy)
            if trace is None:
                continue
            trace["harvester"] = (hx, hy)
            traces.append(trace)

        if not traces:
            continue

        complete = [t for t in traces if t.get("complete", False)]
        if complete:
            best_trace = min(
                complete,
                key=lambda t: (
                    len(t["chain"]),
                    t["terminal_bridge"][0],
                    t["terminal_bridge"][1],
                ),
            )
        else:
            best_trace = max(
                traces,
                key=lambda t: (
                    len(t["chain"]),
                    t["chain"][-1][0] if t["chain"] else 0,
                    t["chain"][-1][1] if t["chain"] else 0,
                ),
            )

        network_id = None
        terminal = best_trace.get("terminal_bridge")
        if isinstance(terminal, tuple) and len(terminal) == 2:
            network_id = state.network_id_by_terminal.get(terminal)
        if network_id is None:
            network_id = state.network_id_by_harvester.get((hx, hy))
        if network_id is None:
            network_id = state.next_network_id
            state.next_network_id += 1

        _record_network_observation(state, network_id, best_trace, rnd)
        observed_ids.add(network_id)

    for network_id, rec in tuple(state.network_records.items()):
        if rec.get("last_seen_round") == rnd:
            continue
        if _network_record_contradicted(local_map, rec, state=state):
            _clear_broken_network_tracking_for_network(state, network_id)
            state.network_records.pop(network_id, None)
            hp = rec.get("harvester_pos")
            if hp in state.network_id_by_harvester:
                state.network_id_by_harvester.pop(hp, None)
            tp = rec.get("terminal_bridge_pos")
            if tp in state.network_id_by_terminal:
                state.network_id_by_terminal.pop(tp, None)
            state.provisional_network_ids.discard(network_id)


def _trace_friendly_transport_chain_partial(local_map, start_xy):
    sx, sy = int(start_xy[0]), int(start_xy[1])
    if not local_map.in_bounds(sx, sy):
        return None
    if not _tile_is_known(local_map, sx, sy):
        return None

    chain = []
    conveyor_memory = []
    seen = set()
    cur_x, cur_y = sx, sy
    steps = 0
    my_team = getattr(local_map, "my_team", None)

    while steps < 192:
        if not local_map.in_bounds(cur_x, cur_y):
            break
        if not _tile_is_known(local_map, cur_x, cur_y):
            break
        if (cur_x, cur_y) in seen:
            break
        seen.add((cur_x, cur_y))

        rec = _known_building_at(local_map, cur_x, cur_y)
        if not isinstance(rec, dict):
            break
        if rec.get("team") != my_team:
            break

        etype = rec.get("entity_type")
        if etype == EntityType.BRIDGE:
            chain.append((cur_x, cur_y))
            target = rec.get("bridge_target")
            if isinstance(target, tuple) and len(target) == 2:
                return {
                    "chain": tuple(chain),
                    "conveyor_memory": tuple(conveyor_memory),
                    "terminal_bridge": (cur_x, cur_y),
                    "bridge_target": (int(target[0]), int(target[1])),
                    "complete": True,
                }
            break

        if etype not in (
            EntityType.CONVEYOR,
            EntityType.SPLITTER,
            EntityType.ARMOURED_CONVEYOR,
        ):
            break

        direction = rec.get("direction")
        if direction is None:
            break

        chain.append((cur_x, cur_y))
        if etype == EntityType.CONVEYOR:
            direction_name = direction.name
            conveyor_memory.append(
                (
                    cur_x,
                    cur_y,
                    direction_name,
                    _entity_id_from_rec(rec),
                )
            )

        ddx, ddy = direction.delta()
        cur_x += ddx
        cur_y += ddy
        steps += 1

    if not chain:
        return None
    return {
        "chain": tuple(chain),
        "conveyor_memory": tuple(conveyor_memory),
        "terminal_bridge": None,
        "bridge_target": None,
        "complete": False,
    }


def _ensure_active_network_id_for_harvester(
    state: EconomyState,
    harvester_xy,
    rnd: int,
) -> int:
    hx = int(harvester_xy[0])
    hy = int(harvester_xy[1])
    hxy = (hx, hy)

    network_id = state.network_id_by_harvester.get(hxy)
    if network_id is None:
        network_id = state.next_network_id
        state.next_network_id += 1

    rec = state.network_records.get(network_id)
    old_hxy = None
    if isinstance(rec, dict):
        old_hp = rec.get("harvester_pos")
        if isinstance(old_hp, tuple) and len(old_hp) == 2:
            old_hxy = (int(old_hp[0]), int(old_hp[1]))
    if rec is None:
        rec = {
            "id": network_id,
            "harvester_pos": hxy,
            "terminal_bridge_pos": None,
            "bridge_target_pos": None,
            "transport_chain_positions": (),
            "conveyor_memory": (),
            "complete": False,
            "built_by_me": False,
            "last_seen_round": rnd,
        }
    else:
        rec["harvester_pos"] = hxy
        rec["last_seen_round"] = rnd

    state.network_records[network_id] = rec
    if (
        isinstance(old_hxy, tuple)
        and old_hxy != hxy
        and state.network_id_by_harvester.get(old_hxy) == network_id
    ):
        state.network_id_by_harvester.pop(old_hxy, None)
    state.network_id_by_harvester[hxy] = network_id
    if rec.get("complete", False):
        state.provisional_network_ids.discard(network_id)
    else:
        state.provisional_network_ids.add(network_id)

    return network_id


def _append_active_network_transport(state: EconomyState, tile_xy, rnd: int):
    if not isinstance(state.active_network_id, int):
        return

    rec = state.network_records.get(state.active_network_id)
    if rec is None:
        rec = {
            "id": state.active_network_id,
            "harvester_pos": None,
            "terminal_bridge_pos": None,
            "bridge_target_pos": None,
            "transport_chain_positions": (),
            "conveyor_memory": (),
            "complete": False,
            "built_by_me": False,
            "last_seen_round": rnd,
        }

    px = int(tile_xy[0])
    py = int(tile_xy[1])
    chain = [
        (int(p[0]), int(p[1]))
        for p in rec.get("transport_chain_positions", ())
        if isinstance(p, tuple) and len(p) == 2
    ]
    if (px, py) not in chain:
        chain.append((px, py))

    rec["transport_chain_positions"] = tuple(chain)
    rec["complete"] = bool(rec.get("complete", False))
    rec["last_seen_round"] = rnd

    state.network_records[state.active_network_id] = rec
    if rec.get("complete", False):
        state.provisional_network_ids.discard(state.active_network_id)
    else:
        state.provisional_network_ids.add(state.active_network_id)

    _remove_tile_from_broken_tracking(state, (px, py))


def _record_active_network_terminal_bridge(
    state: EconomyState,
    bridge_xy,
    target_xy,
    rnd: int,
):
    if not isinstance(state.active_network_id, int):
        _hp = getattr(state, "network_highway_pending_harvester", None)
        if isinstance(_hp, tuple) and len(_hp) == 2:
            state.active_network_id = _ensure_active_network_id_for_harvester(
                state, _hp, rnd)
    if not isinstance(state.active_network_id, int):
        return

    rec = state.network_records.get(state.active_network_id)
    if rec is None:
        rec = {
            "id": state.active_network_id,
            "harvester_pos": None,
            "terminal_bridge_pos": None,
            "bridge_target_pos": None,
            "transport_chain_positions": (),
            "conveyor_memory": (),
            "complete": False,
            "built_by_me": False,
            "last_seen_round": rnd,
        }

    bxy = (int(bridge_xy[0]), int(bridge_xy[1]))
    txy = (int(target_xy[0]), int(target_xy[1]))

    rec["terminal_bridge_pos"] = bxy
    rec["bridge_target_pos"] = txy
    rec["complete"] = True
    rec["last_seen_round"] = rnd

    state.network_records[state.active_network_id] = rec
    state.network_id_by_terminal[bxy] = state.active_network_id
    state.provisional_network_ids.discard(state.active_network_id)
    _remove_tile_from_broken_tracking(state, bxy)
    _clear_broken_network_tracking_for_network(state, state.active_network_id)


def _record_network_observation(state: EconomyState, network_id: int, obs, rnd: int):
    rec = state.network_records.get(network_id)
    if rec is None:
        rec = {
            "id": network_id,
            "harvester_pos": None,
            "terminal_bridge_pos": None,
            "bridge_target_pos": None,
            "transport_chain_positions": (),
            "conveyor_memory": (),
            "complete": False,
            "built_by_me": False,
            "last_seen_round": -1,
        }

    old_hxy = None
    old_hp = rec.get("harvester_pos")
    if isinstance(old_hp, tuple) and len(old_hp) == 2:
        old_hxy = (int(old_hp[0]), int(old_hp[1]))

    harvester_pos = obs.get("harvester")
    if isinstance(harvester_pos, tuple) and len(harvester_pos) == 2:
        harvester_pos = (int(harvester_pos[0]), int(harvester_pos[1]))
    else:
        harvester_pos = None
    chain = tuple(obs.get("chain", ()))
    conveyor_memory = _normalise_conveyor_memory_entries(
        obs.get("conveyor_memory", ())
    )
    complete = bool(obs.get("complete", False))
    terminal = obs.get("terminal_bridge")
    bridge_target = obs.get("bridge_target")

    rec["harvester_pos"] = harvester_pos
    if complete or not rec.get("complete", False):
        new_chain_set = {
            (int(p[0]), int(p[1]))
            for p in chain
            if isinstance(p, tuple) and len(p) == 2
        }
        if rec.get("complete", False):
            # Merge with existing positions: highway-switch tiles added via
            # _append_active_network_transport may not appear in a short retrace
            # but must stay in the index so bridge-target lookups can find them.
            old_chain_set = {
                (int(p[0]), int(p[1]))
                for p in rec.get("transport_chain_positions", ())
                if isinstance(p, tuple) and len(p) == 2
            }
            rec["transport_chain_positions"] = tuple(sorted(old_chain_set | new_chain_set))
        else:
            rec["transport_chain_positions"] = tuple(sorted(new_chain_set))
        if conveyor_memory:
            rec["conveyor_memory"] = conveyor_memory
        elif complete:
            rec["conveyor_memory"] = ()
        elif "conveyor_memory" not in rec:
            rec["conveyor_memory"] = ()
    elif "conveyor_memory" not in rec:
        rec["conveyor_memory"] = ()
    rec["last_seen_round"] = rnd

    if complete:
        rec["complete"] = True
        rec["terminal_bridge_pos"] = terminal
        if bridge_target is not None:
            rec["bridge_target_pos"] = bridge_target
        elif rec.get("bridge_target_pos") is None:
            rec["bridge_target_pos"] = bridge_target
    elif not rec.get("complete", False):
        rec["terminal_bridge_pos"] = terminal
        rec["bridge_target_pos"] = bridge_target

    chain_built = True
    chain_for_built = tuple(rec.get("transport_chain_positions", ()))
    for p in chain_for_built:
        if p not in state.built_transport_positions:
            chain_built = False
            break
    rec["built_by_me"] = bool(
        harvester_pos in state.built_harvester_positions and chain_built
    )

    state.network_records[network_id] = rec

    if isinstance(harvester_pos, tuple) and len(harvester_pos) == 2:
        if (
            isinstance(old_hxy, tuple)
            and old_hxy != harvester_pos
            and state.network_id_by_harvester.get(old_hxy) == network_id
        ):
            state.network_id_by_harvester.pop(old_hxy, None)
        state.network_id_by_harvester[harvester_pos] = network_id
    if rec.get("complete") and isinstance(rec.get("terminal_bridge_pos"), tuple):
        state.network_id_by_terminal[rec["terminal_bridge_pos"]] = network_id
        state.provisional_network_ids.discard(network_id)
    else:
        state.provisional_network_ids.add(network_id)


def _network_record_contradicted(local_map, rec, state: EconomyState | None = None) -> bool:
    hp = rec.get("harvester_pos")
    if isinstance(hp, tuple) and len(hp) == 2 and _tile_is_known(local_map, hp[0], hp[1]):
        if local_map.is_visible(hp[0], hp[1]):
            hrec = _known_building_at(local_map, hp[0], hp[1])
            if not isinstance(hrec, dict):
                return True
            trusted_stolen = False
            if state is not None:
                if hp in state.harvester_stolen_positions:
                    trusted_stolen = True
                hid = _entity_id_from_rec(hrec)
                if isinstance(hid, int) and hid in state.harvester_ids_stolen:
                    trusted_stolen = True

            if hrec.get("team") != getattr(local_map, "my_team", None) and not trusted_stolen:
                return True
            if hrec.get("entity_type") != EntityType.HARVESTER:
                return True

    tp = rec.get("terminal_bridge_pos")
    if rec.get("complete", False) and isinstance(tp, tuple) and len(tp) == 2:
        if _tile_is_known(local_map, tp[0], tp[1]) and local_map.is_visible(tp[0], tp[1]):
            trec = _known_building_at(local_map, tp[0], tp[1])
            if not isinstance(trec, dict):
                return True
            if trec.get("team") != getattr(local_map, "my_team", None):
                return True
            if trec.get("entity_type") != EntityType.BRIDGE:
                return True

    # Keep complete network memories even when interior conveyors are replaced by
    # enemies so repair can retrace and rebuild from remembered structure.
    if not rec.get("complete", False):
        for x, y in rec.get("transport_chain_positions", ()):
            if not _tile_is_known(local_map, x, y):
                continue
            if not local_map.is_visible(x, y):
                continue
            crec = _known_building_at(local_map, x, y)
            if not isinstance(crec, dict):
                return True
            if crec.get("team") != getattr(local_map, "my_team", None):
                return True
            if crec.get("entity_type") not in _TRANSPORT_ENTITY_TYPES:
                return True

    return False


def _planner_tile_soft_cost(state: EconomyState, x: int, y: int) -> int:
    return int(state.turret_threat_cost.get((x, y), 0))


def _is_friendly_transport_tile(local_map, x: int, y: int) -> bool:
    rec = _known_building_at(local_map, x, y)
    if not isinstance(rec, dict):
        return False
    if rec.get("team") != getattr(local_map, "my_team", None):
        return False
    return rec.get("entity_type") in _TRANSPORT_ENTITY_TYPES


def _select_direct_bridge_sink_target(state: EconomyState, local_map, source_xy, sink_xy):
    sx = int(source_xy[0])
    sy = int(source_xy[1])
    tx = int(sink_xy[0])
    ty = int(sink_xy[1])

    if not _is_bridge_build_tile_viable(local_map, sx, sy):
        return None

    dist_sq = (tx - sx) * (tx - sx) + (ty - sy) * (ty - sy)
    if dist_sq <= 0 or dist_sq > 9:
        return None

    rec = _known_building_at(local_map, tx, ty)
    if not isinstance(rec, dict):
        return None
    if rec.get("team") != getattr(local_map, "my_team", None):
        return None
    if rec.get("entity_type") != EntityType.BRIDGE:
        return None

    owner_index = _build_network_tile_owner_index(state)
    owner_ids = owner_index.get((tx, ty), set())
    source_network_id = _resolve_source_network_id(state)
    if isinstance(source_network_id, int) and source_network_id in owner_ids:
        return None

    return (tx, ty)


def _can_reuse_terminal_bridge(local_map, bridge_xy, target_xy) -> bool:
    rec = _known_building_at(local_map, bridge_xy[0], bridge_xy[1])
    if not isinstance(rec, dict):
        return False
    if rec.get("team") != getattr(local_map, "my_team", None):
        return False
    if rec.get("entity_type") != EntityType.BRIDGE:
        return False

    # If we have known bridge target metadata, require a target match.
    if isinstance(target_xy, tuple) and len(target_xy) == 2:
        existing_target = rec.get("bridge_target")
        if isinstance(existing_target, tuple) and len(existing_target) == 2:
            return (
                int(existing_target[0]),
                int(existing_target[1]),
            ) == (
                int(target_xy[0]),
                int(target_xy[1]),
            )

    return True


def _build_network_tile_owner_index(state: EconomyState, complete_only: bool = False):
    owners = {}

    records = getattr(state, "network_records", None)
    if not isinstance(records, dict):
        return owners

    for network_id, rec in records.items():
        if not isinstance(network_id, int):
            continue
        if not isinstance(rec, dict):
            continue
        if complete_only and not rec.get("complete", False):
            continue

        hpos = rec.get("harvester_pos")
        if isinstance(hpos, tuple) and len(hpos) == 2:
            key = (int(hpos[0]), int(hpos[1]))
            bucket = owners.get(key)
            if bucket is None:
                owners[key] = {network_id}
            else:
                bucket.add(network_id)

        for p in rec.get("transport_chain_positions", ()):
            if not (isinstance(p, tuple) and len(p) == 2):
                continue
            key = (int(p[0]), int(p[1]))
            bucket = owners.get(key)
            if bucket is None:
                owners[key] = {network_id}
            else:
                bucket.add(network_id)

        tpos = rec.get("terminal_bridge_pos")
        if isinstance(tpos, tuple) and len(tpos) == 2:
            key = (int(tpos[0]), int(tpos[1]))
            bucket = owners.get(key)
            if bucket is None:
                owners[key] = {network_id}
            else:
                bucket.add(network_id)

    return owners


def _resolve_source_network_id(state: EconomyState, harvester_xy=None):
    active_id = getattr(state, "active_network_id", None)
    if isinstance(active_id, int):
        return active_id

    if isinstance(harvester_xy, tuple) and len(harvester_xy) == 2:
        hx = int(harvester_xy[0])
        hy = int(harvester_xy[1])
        by_harvester = getattr(state, "network_id_by_harvester", None)
        if isinstance(by_harvester, dict):
            network_id = by_harvester.get((hx, hy))
            if isinstance(network_id, int):
                return network_id

    return None


def _is_core_graph_target(state: EconomyState, target_xy) -> bool:
    if not (isinstance(target_xy, tuple) and len(target_xy) == 2):
        return False
    tx = int(target_xy[0])
    ty = int(target_xy[1])
    cx, cy = state.core_xy
    return abs(tx - cx) <= 1 and abs(ty - cy) <= 1


def _compact_xy_pairs(values, max_items: int = 16) -> str:
    pts = []
    for p in values:
        if isinstance(p, tuple) and len(p) == 2:
            pts.append((int(p[0]), int(p[1])))

    if not pts:
        return ""

    pts.sort()
    shown = pts[:max_items]
    out = ",".join(f"({x},{y})" for x, y in shown)
    if len(pts) > max_items:
        out += f",...(+{len(pts) - max_items})"
    return out


def _format_adjacency_snapshot(
    local_map,
    focus_tiles,
    max_tiles: int = 6,
    max_neighbors: int = 6,
) -> str:
    entries = []
    seen = set()
    blocked_lookup = getattr(local_map, "dynamic_blocked_tiles", set())

    for tile in focus_tiles:
        if len(entries) >= max_tiles:
            break
        if not (isinstance(tile, tuple) and len(tile) == 2):
            continue

        tx = int(tile[0])
        ty = int(tile[1])
        key = (tx, ty)
        if key in seen:
            continue
        seen.add(key)

        neighbors = ()
        if hasattr(local_map, "get_adjacency_neighbors"):
            neighbors = local_map.get_adjacency_neighbors(tx, ty)

        compact_neighbors = []
        for nx, ny in neighbors[:max_neighbors]:
            compact_neighbors.append(f"({int(nx)},{int(ny)})")
        neighbor_dump = ",".join(compact_neighbors)
        if len(neighbors) > max_neighbors:
            neighbor_dump += f",...(+{len(neighbors) - max_neighbors})"

        is_blocked = 1 if key in blocked_lookup else 0
        entries.append(
            f"({tx},{ty})|blocked={is_blocked}|deg={len(neighbors)}|n=[{neighbor_dump}]"
        )

    return ";".join(entries)


def _compact_sorted_ints(values, max_items: int = 24) -> str:
    vals = sorted(int(v) for v in values if isinstance(v, int))
    if not vals:
        return ""
    if len(vals) > max_items:
        head = ",".join(str(v) for v in vals[:max_items])
        return f"{head},...(+{len(vals) - max_items})"
    return ",".join(str(v) for v in vals)


def _compact_edges(edges, max_items: int = 32) -> str:
    if not edges:
        return ""
    out = []
    for src, dst in edges[:max_items]:
        out.append(f"{src}->{dst}")
    if len(edges) > max_items:
        out.append(f"...(+{len(edges) - max_items})")
    return ",".join(out)


def _compact_int_map(mapping: dict, max_items: int = 24) -> str:
    if not isinstance(mapping, dict) or not mapping:
        return ""
    items = sorted(
        (int(k), int(v))
        for k, v in mapping.items()
        if isinstance(k, int) and isinstance(v, int)
    )
    if not items:
        return ""

    out = []
    for k, v in items[:max_items]:
        out.append(f"{k}:{v}")
    if len(items) > max_items:
        out.append(f"...(+{len(items) - max_items})")
    return ",".join(out)



def _is_enemy_replaceable_blocker_at(local_map, x: int, y: int) -> bool:
    rec = _known_building_at(local_map, x, y)
    if not isinstance(rec, dict):
        return False
    if rec.get("team") == getattr(local_map, "my_team", None):
        return False
    return rec.get("entity_type") in _REPLACEABLE_ENEMY_BLOCKERS


def _select_bridge_escape_target_v2(
    state: EconomyState,
    local_map,
    source_xy,
    core_xy,
    exclude_positions=(),
    congestion_excluded_tiles=(),
    congestion_excluded_ids=(),
    allowed_network_id: int | None = None,
    source_network_id_override: int | None = None,
    selection_stats=None,
):
    sx, sy = source_xy
    if not _is_bridge_build_tile_viable(local_map, sx, sy):
        return None

    excluded = set()
    for p in exclude_positions:
        if isinstance(p, tuple) and len(p) == 2:
            excluded.add((int(p[0]), int(p[1])))

    blocked_tiles = set()
    for p in congestion_excluded_tiles:
        if isinstance(p, tuple) and len(p) == 2:
            blocked_tiles.add((int(p[0]), int(p[1])))

    blocked_networks = set(
        int(network_id)
        for network_id in congestion_excluded_ids
        if isinstance(network_id, int)
    )

    rejected_tiles = 0
    rejected_networks = 0

    my_team = getattr(local_map, "my_team", None)
    known_bridge_targets = _known_friendly_bridge_targets(local_map)
    tile_owners = _build_network_tile_owner_index(state)
    source_network_id = (
        int(source_network_id_override)
        if isinstance(source_network_id_override, int)
        else _resolve_source_network_id(state)
    )
    allowed_network_id = (
        int(allowed_network_id)
        if isinstance(allowed_network_id, int)
        else None
    )
    cx, cy = core_xy
    best_spaced = None
    best_spaced_key = None
    best_relaxed = None
    best_relaxed_key = None
    best_bridge = None
    best_bridge_key = None

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
        etype = rec.get("entity_type")
        is_transport = etype in (
            EntityType.CONVEYOR,
            EntityType.SPLITTER,
            EntityType.ARMOURED_CONVEYOR,
        )
        is_bridge = etype == EntityType.BRIDGE
        if not (is_transport or is_bridge):
            continue

        pos = rec.get("position")
        if not (isinstance(pos, tuple) and len(pos) == 2):
            continue

        tx = int(pos[0])
        ty = int(pos[1])
        if (tx, ty) == (sx, sy):
            continue
        if (tx, ty) in excluded:
            continue
        if (tx, ty) in blocked_tiles:
            rejected_tiles += 1
            continue

        owner_ids = tile_owners.get((tx, ty))
        if not owner_ids:
            continue
        if isinstance(source_network_id, int) and source_network_id in owner_ids:
            continue
        if isinstance(allowed_network_id, int) and allowed_network_id not in owner_ids:
            continue
        if blocked_networks and any(owner in blocked_networks for owner in owner_ids):
            rejected_networks += 1
            continue

        dist_sq = (tx - sx) * (tx - sx) + (ty - sy) * (ty - sy)
        if dist_sq <= 0 or dist_sq > 9:
            continue

        key = (
            dist_sq,
            abs(tx - cx) + abs(ty - cy),
            _planner_tile_soft_cost(state, tx, ty),
            tx,
            ty,
        )
        if is_transport:
            if _bridge_target_region_clear((tx, ty), known_bridge_targets, radius_cheb=1):
                if best_spaced_key is None or key < best_spaced_key:
                    best_spaced_key = key
                    best_spaced = (tx, ty)
            else:
                if best_relaxed_key is None or key < best_relaxed_key:
                    best_relaxed_key = key
                    best_relaxed = (tx, ty)
            continue

        # Last resort: allow connecting a bridge-to-bridge sink to break deadlocks.
        if best_bridge_key is None or key < best_bridge_key:
            best_bridge_key = key
            best_bridge = (tx, ty)

    # Secondary pass: check state.network_records for known positions not
    # currently visible. This handles the case where the highway target is
    # known (built this session) but outside the current vision range.
    if best_spaced is None and best_relaxed is None:
        records = getattr(state, "network_records", None)
        if isinstance(records, dict):
            for network_id, rec in records.items():
                if not isinstance(rec, dict):
                    continue
                if not rec.get("complete", False):
                    continue
                if isinstance(source_network_id, int) and source_network_id == network_id:
                    continue
                if isinstance(allowed_network_id, int) and allowed_network_id != network_id:
                    continue
                if network_id in blocked_networks:
                    continue
                for p in rec.get("transport_chain_positions", ()):
                    if not (isinstance(p, tuple) and len(p) == 2):
                        continue
                    tx, ty = int(p[0]), int(p[1])
                    if (tx, ty) == (sx, sy) or (tx, ty) in excluded or (tx, ty) in blocked_tiles:
                        continue
                    dist_sq = (tx - sx) * (tx - sx) + (ty - sy) * (ty - sy)
                    if dist_sq <= 0 or dist_sq > 9:
                        continue
                    key = (
                        dist_sq,
                        abs(tx - cx) + abs(ty - cy),
                        _planner_tile_soft_cost(state, tx, ty),
                        tx,
                        ty,
                    )
                    if _bridge_target_region_clear((tx, ty), known_bridge_targets, radius_cheb=1):
                        if best_spaced_key is None or key < best_spaced_key:
                            best_spaced_key = key
                            best_spaced = (tx, ty)
                    else:
                        if best_relaxed_key is None or key < best_relaxed_key:
                            best_relaxed_key = key
                            best_relaxed = (tx, ty)

    if best_spaced is not None:
        if isinstance(selection_stats, dict):
            selection_stats["rejected_tiles"] = rejected_tiles
            selection_stats["rejected_networks"] = rejected_networks
        return best_spaced
    if best_relaxed is not None:
        if isinstance(selection_stats, dict):
            selection_stats["rejected_tiles"] = rejected_tiles
            selection_stats["rejected_networks"] = rejected_networks
        return best_relaxed
    if isinstance(selection_stats, dict):
        selection_stats["rejected_tiles"] = rejected_tiles
        selection_stats["rejected_networks"] = rejected_networks
    return best_bridge
# ===== MAP & BUILDING UTILITIES =====
# Pure map query helpers, building construction helpers (conveyor, bridge,
# road), enemy harvester takeover logic, and state-transition helpers
# (_resume_exploration_*, _refresh_*, _track_symmetry_*, turret threats, etc.)




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
    return local_map.get_known_building(x, y)


def _known_unit_at(local_map, x: int, y: int):
    return local_map.get_known_unit(x, y)


def _is_unit_halo_blocked(local_map, x: int, y: int) -> bool:
    # Never block our own tile (set by run_economy each tick).
    self_xy = local_map.planner_self_xy
    if self_xy[0] == x and self_xy[1] == y:
        return False

    # Always block tiles that are actually occupied.
    if local_map.get_known_unit(x, y) is not None:
        return True

    # Halo blocking can be toggled off during pre-launch queuing.
    if not local_map.enable_unit_halo_planning:
        return False

    return (x, y) in local_map.unit_halo_blocked_tiles


def _is_unit_halo_predicted_blocked(local_map, x: int, y: int, from_xy=None) -> bool:
    _ = from_xy
    # Highway-switch decisions should only react to the current halo/occupancy
    # model, not predictive expansion, to avoid false-positive switch loops.
    return _is_unit_halo_blocked(local_map, x, y)


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

        if not _bridge_target_region_clear((tx, ty), known_bridge_targets, radius_cheb=1):
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
    state: EconomyState,
    local_map,
    harvester_xy,
    bridge_build_xy,
    core_xy,
    exclude_positions=(),
    congestion_excluded_tiles=(),
    congestion_excluded_ids=(),
    allowed_network_id: int | None = None,
    source_network_id_override: int | None = None,
    selection_stats=None,
):
    hx, hy = int(harvester_xy[0]), int(harvester_xy[1])
    bx, by = int(bridge_build_xy[0]), int(bridge_build_xy[1])

    if not _is_bridge_build_tile_viable(local_map, bx, by):
        return None

    my_team = getattr(local_map, "my_team", None)
    known_bridge_targets = _known_friendly_bridge_targets(local_map)
    tile_owners = _build_network_tile_owner_index(state)
    source_network_id = (
        int(source_network_id_override)
        if isinstance(source_network_id_override, int)
        else _resolve_source_network_id(state, (hx, hy))
    )
    allowed_network_id = (
        int(allowed_network_id)
        if isinstance(allowed_network_id, int)
        else None
    )
    cx, cy = int(core_xy[0]), int(core_xy[1])
    blocked_tiles = set()
    excluded_positions = set()
    for p in exclude_positions:
        if isinstance(p, tuple) and len(p) == 2:
            excluded_positions.add((int(p[0]), int(p[1])))
    for p in congestion_excluded_tiles:
        if isinstance(p, tuple) and len(p) == 2:
            blocked_tiles.add((int(p[0]), int(p[1])))
    blocked_networks = set(
        int(network_id)
        for network_id in congestion_excluded_ids
        if isinstance(network_id, int)
    )

    rejected_tiles = 0
    rejected_networks = 0

    best_spaced = None
    best_spaced_key = None
    best_relaxed = None
    best_relaxed_key = None
    best_bridge = None
    best_bridge_key = None

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
        etype = rec.get("entity_type")
        is_transport = etype in (
            EntityType.CONVEYOR,
            EntityType.ARMOURED_CONVEYOR,
            EntityType.SPLITTER,
        )
        is_bridge = etype == EntityType.BRIDGE
        if not (is_transport or is_bridge):
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
        if (tx, ty) in excluded_positions:
            continue
        if (tx, ty) in blocked_tiles:
            rejected_tiles += 1
            continue

        owner_ids = tile_owners.get((tx, ty))
        if not owner_ids:
            continue
        if isinstance(source_network_id, int) and source_network_id in owner_ids:
            continue
        if isinstance(allowed_network_id, int) and allowed_network_id not in owner_ids:
            continue
        if blocked_networks and any(owner in blocked_networks for owner in owner_ids):
            rejected_networks += 1
            continue

        dist_sq = (tx - bx) * (tx - bx) + (ty - by) * (ty - by)
        if dist_sq <= 0 or dist_sq > 9:
            continue

        key = (
            max(abs(tx - hx), abs(ty - hy)),
            abs(tx - hx) + abs(ty - hy),
            abs(tx - cx) + abs(ty - cy),
            dist_sq,
            tx,
            ty,
        )
        if is_transport:
            if _bridge_target_region_clear((tx, ty), known_bridge_targets, radius_cheb=1):
                if best_spaced_key is None or key < best_spaced_key:
                    best_spaced_key = key
                    best_spaced = (tx, ty)
            else:
                if best_relaxed_key is None or key < best_relaxed_key:
                    best_relaxed_key = key
                    best_relaxed = (tx, ty)
            continue

        # Last resort for dense lane topologies.
        if best_bridge_key is None or key < best_bridge_key:
            best_bridge_key = key
            best_bridge = (tx, ty)

    if best_spaced is not None:
        if isinstance(selection_stats, dict):
            selection_stats["rejected_tiles"] = rejected_tiles
            selection_stats["rejected_networks"] = rejected_networks
            selection_stats["highway_local_fallback"] = 0
        return best_spaced
    if best_relaxed is not None:
        if isinstance(selection_stats, dict):
            selection_stats["rejected_tiles"] = rejected_tiles
            selection_stats["rejected_networks"] = rejected_networks
            selection_stats["highway_local_fallback"] = 0
        return best_relaxed

    # Highway mode can be stale (e.g. after long reroutes). If no harvester-local
    # sink is available, fall back to local bridge-escape targeting around the
    # current tile so we can still break out of a friendly transport collision.
    local_fallback = _select_bridge_escape_target_v2(
        state,
        local_map,
        (bx, by),
        (cx, cy),
        exclude_positions=excluded_positions,
        congestion_excluded_tiles=congestion_excluded_tiles,
        congestion_excluded_ids=congestion_excluded_ids,
        allowed_network_id=allowed_network_id,
        source_network_id_override=source_network_id,
        selection_stats=None,
    )
    if local_fallback is not None:
        if isinstance(selection_stats, dict):
            selection_stats["rejected_tiles"] = rejected_tiles
            selection_stats["rejected_networks"] = rejected_networks
            selection_stats["highway_local_fallback"] = 1
        return local_fallback

    if isinstance(selection_stats, dict):
        selection_stats["rejected_tiles"] = rejected_tiles
        selection_stats["rejected_networks"] = rejected_networks
        selection_stats["highway_local_fallback"] = 0
    return best_bridge


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
        team = rec.get("team")
        if etype == EntityType.MARKER:
            return True
        if team != getattr(local_map, "my_team", None):
            return False
        if etype not in (
            EntityType.ROAD,
            EntityType.CONVEYOR,
            EntityType.SPLITTER,
            EntityType.ARMOURED_CONVEYOR,
            EntityType.BRIDGE,
        ):
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


def _is_conveyor_planner_passable(
    local_map,
    x: int,
    y: int,
    goal_xy,
    exclude_positions=(),
    exclude_network_ids=(),
    tile_owner_index=None,
    block_friendly_infra: bool = False,
) -> bool:
    if not local_map.in_bounds(x, y):
        return False

    dynamic_blocked = getattr(local_map, "dynamic_blocked_tiles", None)
    if isinstance(dynamic_blocked, set) and (x, y) in dynamic_blocked and (x, y) != goal_xy:
        return False

    if _is_unit_halo_blocked(local_map, x, y):
        return False

    if local_map.get(x, y) in (MAP_ORE_TITANIUM, MAP_ORE_AXIONITE):
        return False

    cur = (x, y)
    if exclude_positions:
        if cur != goal_xy:
            if isinstance(exclude_positions, set):
                if cur in exclude_positions:
                    return False
            else:
                for p in exclude_positions:
                    if isinstance(p, tuple) and len(p) == 2 and cur == (int(p[0]), int(p[1])):
                        return False

    if exclude_network_ids and isinstance(tile_owner_index, dict):
        excluded_network_set = exclude_network_ids
        if not isinstance(excluded_network_set, set):
            excluded_network_set = {
                int(network_id)
                for network_id in exclude_network_ids
                if isinstance(network_id, int)
            }
        if excluded_network_set:
            owners = tile_owner_index.get(cur)
            if owners:
                for owner in owners:
                    if owner in excluded_network_set:
                        return False

    rec = _known_building_at(local_map, x, y)
    my_team = getattr(local_map, "my_team", None)
    if isinstance(rec, dict):
        etype = rec.get("entity_type")
        team = rec.get("team")

        if etype == EntityType.ARMOURED_CONVEYOR and team != my_team:
            return False

        if block_friendly_infra and team == my_team and etype in (
            EntityType.CONVEYOR,
            EntityType.ARMOURED_CONVEYOR,
            EntityType.SPLITTER,
            EntityType.BRIDGE,
        ):
            return False

        if etype in (
            EntityType.CONVEYOR,
            EntityType.ARMOURED_CONVEYOR,
            EntityType.SPLITTER,
            EntityType.BRIDGE,
            EntityType.ROAD,
            EntityType.MARKER,
        ):
            return True

        if etype == EntityType.CORE and team == my_team:
            return True

        return (x, y) == goal_xy and etype in (
            EntityType.CONVEYOR,
            EntityType.SPLITTER,
            EntityType.BRIDGE,
        )

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
    if etype in (
        EntityType.ROAD,
        EntityType.CONVEYOR,
        EntityType.SPLITTER,
        EntityType.BRIDGE,
    ):
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


def _build_network_segment_transport(
    c: Controller,
    build_xy,
    from_xy,
    to_xy,
    rnd: int,
    uid: int,
    state: EconomyState | None,
    conveyor_tag: str,
    bridge_tag: str,
) -> str:
    if not _is_adjacent_step(from_xy, to_xy):
        return "invalid"

    if _is_diagonal_step(from_xy, to_xy):
        result = _build_bridge_on_tile(
            c,
            build_xy,
            to_xy,
            rnd,
            uid,
            state=state,
        )
        if result in ("built", "already_built"):
            log_event(
                rnd,
                uid,
                "economy",
                f"({build_xy[0]},{build_xy[1]})",
                bridge_tag,
                tx=to_xy[0],
                ty=to_xy[1],
            )
        return result

    out_dir = _direction_from_delta(from_xy, to_xy)
    return _build_conveyor_on_tile(
        c,
        build_xy,
        out_dir,
        rnd,
        uid,
        conveyor_tag,
        state=state,
    )


def _build_conveyor_on_tile(
    c: Controller,
    build_xy,
    out_dir: Direction,
    rnd: int,
    uid: int,
    tag: str,
    state: EconomyState | None = None,
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

        new_id = c.build_conveyor(bp, out_dir)
        if state is not None:
            if isinstance(new_id, int):
                state.built_entity_ids.add(new_id)
            state.built_transport_positions.add((build_xy[0], build_xy[1]))
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


def _build_bridge_on_tile(
    c: Controller,
    bridge_xy,
    target_xy,
    rnd: int,
    uid: int,
    state: EconomyState | None = None,
    require_target_match_for_existing: bool = False,
    replace_mismatched_existing_bridge: bool = False,
) -> str:
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
                    if match_target:
                        return "already_built"
                    if require_target_match_for_existing and not match_target:
                        return "invalid"
                    if not replace_mismatched_existing_bridge:
                        return "already_built"

                    if c.get_action_cooldown() > 0:
                        return "wait_cd"

                    try:
                        if c.can_destroy(bp):
                            c.destroy(bp)
                            existing = None
                        else:
                            return "invalid"
                    except GameError:
                        return "invalid"
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
        new_id = c.build_bridge(bp, tp)
        if state is not None:
            if isinstance(new_id, int):
                state.built_entity_ids.add(new_id)
            state.built_transport_positions.add((bridge_xy[0], bridge_xy[1]))
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
    move_dir = _DIRECTION_BY_DELTA.get((dx, dy))
    if move_dir is None:
        return "invalid"

    nxt_pos = Position(nxt_xy[0], nxt_xy[1])
    try:
        nxt_env = c.get_tile_env(nxt_pos)
        if nxt_env in (Environment.WALL, Environment.ORE_TITANIUM, Environment.ORE_AXIONITE):
            return "blocked_hard"
    except GameError:
        pass

    try:
        if not c.can_move(move_dir):
            try:
                nxt_env = c.get_tile_env(nxt_pos)
                if nxt_env in (
                    Environment.WALL,
                    Environment.ORE_TITANIUM,
                    Environment.ORE_AXIONITE,
                ):
                    return "blocked_hard"
            except GameError:
                pass
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
        try:
            nxt_env = c.get_tile_env(nxt_pos)
            if nxt_env in (Environment.WALL, Environment.ORE_TITANIUM, Environment.ORE_AXIONITE):
                return "blocked_hard"
        except GameError:
            pass
        return "blocked"


def _network_fallback_to_next_objective(state: EconomyState, cur_xy, known_ti):
    state.network_target = None
    state.network_path_nodes = ()
    state.network_path_index = 0
    state.network_escape_bridge_target = None
    state.network_highway_pending_harvester = None
    state.network_highway_active_harvester = None
    state.highway_excluded_transport = ()
    state.active_network_id = None
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


def _is_outside_core_ore_exclusion(
    ore_xy,
    core_xy,
    min_core_cheb: int | None,
) -> bool:
    if core_xy is None or min_core_cheb is None:
        return True
    return _chebyshev(core_xy, ore_xy) >= int(min_core_cheb)


def _known_unharvested_titanium(
    local_map,
    core_xy=None,
    min_core_cheb: int | None = None,
):
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
            ore_xy = (int(item[0]), int(item[1]))
            if not _is_outside_core_ore_exclusion(
                ore_xy,
                core_xy,
                min_core_cheb,
            ):
                continue
            out.append(ore_xy)
        return out

    out = []
    for x, y in getattr(local_map, "titanium_unharvested", set()):
        if not _tile_is_known(local_map, x, y):
            continue
        ore_xy = (x, y)
        if not _is_outside_core_ore_exclusion(
            ore_xy,
            core_xy,
            min_core_cheb,
        ):
            continue
        out.append(ore_xy)
    return out


def _known_unharvested_titanium_unblocked(
    local_map,
    blocked_ores,
    core_xy=None,
    min_core_cheb: int | None = None,
):
    known_ti = _known_unharvested_titanium(
        local_map,
        core_xy=core_xy,
        min_core_cheb=min_core_cheb,
    )
    if not blocked_ores:
        return known_ti
    return [ore for ore in known_ti if ore not in blocked_ores]


def _is_builder_directly_walkable_tile(local_map, x: int, y: int, respect_halo: bool = True) -> bool:
    if not local_map.in_bounds(x, y):
        return False
    if local_map.get_known_unit(x, y) is not None:
        return False
    dynamic_blocked = getattr(local_map, "dynamic_blocked_tiles", None)
    if dynamic_blocked and (x, y) in dynamic_blocked:
        return False
    if respect_halo and _is_unit_halo_blocked(local_map, x, y):
        return False
    rec = local_map.get_known_building(x, y)
    if rec is not None:
        etype = rec.get("entity_type")
        if etype == EntityType.ROAD or etype in _TRANSPORT_ENTITY_TYPES:
            return True
        if etype == EntityType.CORE and rec.get("team") == local_map.my_team:
            return True
        return False
    return local_map.get(x, y) in WALKABLE_TILES


def _is_general_movement_passable(
    local_map,
    x: int,
    y: int,
    goal_xy=None,
    respect_halo: bool = True,
) -> bool:
    if not local_map.in_bounds(x, y):
        return False
    if local_map.get_known_unit(x, y) is not None:
        return False
    dynamic_blocked = getattr(local_map, "dynamic_blocked_tiles", None)
    if dynamic_blocked and (x, y) in dynamic_blocked and (goal_xy is None or (x, y) != goal_xy):
        return False
    if respect_halo and _is_unit_halo_blocked(local_map, x, y):
        return False
    tile = local_map.get(x, y)
    if tile in (MAP_OBSTACLE, MAP_ORE_TITANIUM, MAP_ORE_AXIONITE):
        return False
    rec = local_map.get_known_building(x, y)
    if rec is not None:
        etype = rec.get("entity_type")
        if etype == EntityType.ROAD or etype in _TRANSPORT_ENTITY_TYPES:
            return True
        if etype == EntityType.CORE and rec.get("team") == local_map.my_team:
            return True
        return False
    return tile in PASSABLE_TILES


def _entity_id_from_rec(rec) -> int | None:
    if not isinstance(rec, dict):
        return None
    entity_id = rec.get("id")
    if isinstance(entity_id, int):
        return int(entity_id)
    return None


def _known_enemy_titanium_harvesters(
    local_map,
    state: EconomyState | None = None,
    core_xy=None,
    min_core_cheb: int | None = None,
):
    entities = getattr(local_map, "entities", None)
    if not isinstance(entities, dict):
        return []

    harvested_positions = {
        (int(p[0]), int(p[1]))
        for p in getattr(local_map, "titanium_harvesters", set())
        if isinstance(p, tuple) and len(p) == 2
    }
    my_team = getattr(local_map, "my_team", None)
    out = []

    for rec in entities.values():
        if not isinstance(rec, dict):
            continue
        if not rec.get("alive", False):
            continue
        if rec.get("entity_type") != EntityType.HARVESTER:
            continue
        if rec.get("team") == my_team:
            continue

        pos = rec.get("position")
        if not (isinstance(pos, tuple) and len(pos) == 2):
            continue

        hxy = (int(pos[0]), int(pos[1]))
        if hxy not in harvested_positions:
            continue
        if not _is_outside_core_ore_exclusion(
            hxy,
            core_xy,
            min_core_cheb,
        ):
            continue

        hid = _entity_id_from_rec(rec)

        if state is not None:
            if _harvester_already_known_stolen(state, hxy, hid):
                continue

            if _enemy_harvester_should_be_assumed_stolen(local_map, hxy):
                _mark_harvester_and_adjacent_transports_stolen(
                    state,
                    local_map,
                    hxy,
                    hid,
                )
                continue

        out.append((hid, hxy))

    out.sort(
        key=lambda item: (
            item[1][0],
            item[1][1],
            -1 if item[0] is None else item[0],
        )
    )
    return out


def _enemy_titanium_harvester_id_at(local_map, ore_xy) -> int | None:
    ox, oy = int(ore_xy[0]), int(ore_xy[1])
    rec = _known_building_at(local_map, ox, oy)
    if not isinstance(rec, dict):
        return None
    if rec.get("entity_type") != EntityType.HARVESTER:
        return None
    if rec.get("team") == getattr(local_map, "my_team", None):
        return None

    harvested_positions = getattr(local_map, "titanium_harvesters", set())
    if (ox, oy) not in harvested_positions:
        return None

    return _entity_id_from_rec(rec)


def _is_enemy_titanium_harvester_at(local_map, ore_xy) -> bool:
    ox, oy = int(ore_xy[0]), int(ore_xy[1])
    rec = _known_building_at(local_map, ox, oy)
    if not isinstance(rec, dict):
        return False
    if rec.get("entity_type") != EntityType.HARVESTER:
        return False
    if rec.get("team") == getattr(local_map, "my_team", None):
        return False
    return (ox, oy) in getattr(local_map, "titanium_harvesters", set())


def _adjacent_transport_snapshot(local_map, center_xy):
    cx, cy = int(center_xy[0]), int(center_xy[1])
    my_team = getattr(local_map, "my_team", None)
    friendly_positions = []
    friendly_ids = []
    enemy_positions = []

    for dx, dy in _ADJACENT_DELTAS_8:
        nx = cx + dx
        ny = cy + dy
        if not local_map.in_bounds(nx, ny):
            continue
        rec = _known_building_at(local_map, nx, ny)
        if not isinstance(rec, dict):
            continue
        if rec.get("entity_type") not in _TRANSPORT_ENTITY_TYPES:
            continue

        if rec.get("team") == my_team:
            friendly_positions.append((nx, ny))
            fid = _entity_id_from_rec(rec)
            if isinstance(fid, int):
                friendly_ids.append(fid)
        else:
            enemy_positions.append((nx, ny))

    return tuple(friendly_positions), tuple(friendly_ids), tuple(enemy_positions)


def _harvester_already_known_stolen(
    state: EconomyState,
    harvester_xy,
    harvester_id: int | None,
) -> bool:
    hx = int(harvester_xy[0])
    hy = int(harvester_xy[1])
    if (hx, hy) in state.harvester_stolen_positions:
        return True
    if isinstance(harvester_id, int) and harvester_id in state.harvester_ids_stolen:
        return True
    return False


def _enemy_harvester_should_be_assumed_stolen(local_map, harvester_xy) -> bool:
    friendly_positions, _, enemy_positions = _adjacent_transport_snapshot(
        local_map,
        harvester_xy,
    )
    return bool(friendly_positions) and not enemy_positions


def _mark_harvester_and_adjacent_transports_stolen(
    state: EconomyState,
    local_map,
    harvester_xy,
    harvester_id: int | None = None,
):
    hx, hy = int(harvester_xy[0]), int(harvester_xy[1])
    state.harvester_stolen_positions.add((hx, hy))

    if isinstance(harvester_id, int):
        state.harvester_ids_stolen.add(harvester_id)
        state.harvester_ids_external_friendly.add(harvester_id)
        state.harvester_ids_friendly.add(harvester_id)

    friendly_positions, friendly_ids, _ = _adjacent_transport_snapshot(
        local_map,
        (hx, hy),
    )
    for pxy in friendly_positions:
        state.transport_stolen_positions.add((int(pxy[0]), int(pxy[1])))
    for tid in friendly_ids:
        state.transport_ids_stolen.add(int(tid))
        state.transport_ids_external_friendly.add(int(tid))
        state.transport_ids_friendly.add(int(tid))


def _adjacent_enemy_conveyor_targets(local_map, center_xy):
    return _adjacent_enemy_repair_blocker_targets(local_map, center_xy)


def _pick_enemy_harvester_takeover_goal(local_map, ore_xy, cur_xy):
    ox, oy = int(ore_xy[0]), int(ore_xy[1])
    candidates = []
    for dx, dy in CARDINAL_DELTAS:
        gx = ox + dx
        gy = oy + dy
        if not local_map.in_bounds(gx, gy):
            continue
        if not _is_general_movement_passable(local_map, gx, gy, (gx, gy)):
            continue

        direct_penalty = 0 if _is_builder_directly_walkable_tile(
            local_map, gx, gy) else 1
        candidates.append(
            (
                direct_penalty,
                _manhattan(cur_xy, (gx, gy)),
                gx,
                gy,
            )
        )

    if not candidates:
        return None

    _, _, gx, gy = min(candidates)
    return (gx, gy)


def _run_enemy_harvester_takeover(
    c: Controller,
    state: EconomyState,
    local_map,
    cur_xy,
    rnd: int,
    uid: int,
):
    ore_xy = state.harvest_takeover_ore_xy
    if not (isinstance(ore_xy, tuple) and len(ore_xy) == 2):
        state.harvest_takeover_harvester_id = None
        state.harvest_takeover_ore_xy = None
        state.harvest_ore_xy = None
        state.harvest_goal_xy = None
        state.plan_steps = ()
        state.plan_index = 0
        state.defer_step_once = False
        state.phase = "harvest_pick_ore"
        return

    ox, oy = int(ore_xy[0]), int(ore_xy[1])
    ore_xy = (ox, oy)

    if not _is_enemy_titanium_harvester_at(local_map, ore_xy):
        state.harvest_takeover_harvester_id = None
        state.harvest_takeover_ore_xy = None
        state.harvest_ore_xy = None
        state.harvest_goal_xy = None
        state.plan_steps = ()
        state.plan_index = 0
        state.defer_step_once = False
        state.phase = "harvest_pick_ore"
        log_event(
            rnd,
            uid,
            "economy",
            f"({cur_xy[0]},{cur_xy[1]})",
            "economy_harvester_takeover_target_lost",
            ox=ox,
            oy=oy,
        )
        return

    if state.phase == "harvest_takeover_pick_goal":
        goal_xy = _pick_enemy_harvester_takeover_goal(
            local_map, ore_xy, cur_xy)
        if goal_xy is None:
            return

        state.harvest_goal_xy = goal_xy
        state.plan_steps = ()
        state.plan_index = 0
        state.defer_step_once = False
        state.phase = "harvest_takeover_plan_goal"
        log_event(
            rnd,
            uid,
            "economy",
            f"({cur_xy[0]},{cur_xy[1]})",
            "economy_harvester_takeover_goal_selected",
            ox=ox,
            oy=oy,
            gx=goal_xy[0],
            gy=goal_xy[1],
        )
        return

    goal_xy = state.harvest_goal_xy
    if not (isinstance(goal_xy, tuple) and len(goal_xy) == 2):
        state.phase = "harvest_takeover_pick_goal"
        return

    if state.phase == "harvest_takeover_plan_goal":
        if cur_xy == goal_xy:
            state.phase = "harvest_takeover_attack_start"
            state.plan_steps = ()
            state.plan_index = 0
            state.defer_step_once = False
            return

        def passable_fn(x: int, y: int) -> bool:
            return _is_general_movement_passable(local_map, x, y, goal_xy)

        steps = _astar_cardinal_plan(
            local_map,
            cur_xy,
            goal_xy,
            max_expansions=768,
            tile_passable_fn=passable_fn,
        )
        if not steps:
            return

        state.plan_steps = steps
        state.plan_index = 0
        state.defer_step_once = True
        state.phase = "harvest_takeover_follow_plan"
        log_event(
            rnd,
            uid,
            "economy",
            f"({cur_xy[0]},{cur_xy[1]})",
            "economy_harvester_takeover_plan_ready",
            ox=ox,
            oy=oy,
            gx=goal_xy[0],
            gy=goal_xy[1],
            steps=len(steps),
        )
        return

    if state.phase == "harvest_takeover_follow_plan":
        if cur_xy == goal_xy:
            state.plan_steps = ()
            state.plan_index = 0
            state.defer_step_once = False
            state.phase = "harvest_takeover_attack_start"
            return

        if _manhattan(goal_xy, ore_xy) != 1:
            state.harvest_goal_xy = None
            state.plan_steps = ()
            state.plan_index = 0
            state.defer_step_once = False
            state.phase = "harvest_takeover_pick_goal"
            return

        if state.defer_step_once:
            state.defer_step_once = False
            return

        if state.plan_index >= len(state.plan_steps):
            state.phase = "harvest_takeover_plan_goal"
            return

        nxt = state.plan_steps[state.plan_index]
        if not _is_adjacent_step(cur_xy, nxt):
            state.phase = "harvest_takeover_plan_goal"
            return

        result = _execute_step_toward(c, local_map, cur_xy, nxt, rnd, uid)
        if result == "moved":
            state.issued_move_last_tick = True
            state.expected_xy_after_move = nxt
            state.plan_index += 1
            return
        if result in ("built", "wait_cd"):
            return

        state.plan_steps = ()
        state.plan_index = 0
        state.defer_step_once = False
        state.phase = "harvest_takeover_plan_goal"
        return

    if state.phase == "harvest_takeover_attack_start":
        if _manhattan(cur_xy, ore_xy) != 1:
            state.harvest_goal_xy = None
            state.plan_steps = ()
            state.plan_index = 0
            state.defer_step_once = False
            state.phase = "harvest_takeover_pick_goal"
            return

        attack_targets = _adjacent_enemy_conveyor_targets(local_map, ore_xy)
        if attack_targets:
            _start_network_attack(
                state,
                attack_targets[0],
                "harvest_takeover_finalize",
                "enemy_harvester_takeover",
                target_queue=attack_targets,
                return_xy=cur_xy,
            )
            log_event(
                rnd,
                uid,
                "economy",
                f"({cur_xy[0]},{cur_xy[1]})",
                "economy_harvester_takeover_attack_start",
                ox=ox,
                oy=oy,
                hid=(state.harvest_takeover_harvester_id if isinstance(
                    state.harvest_takeover_harvester_id, int) else -1),
                targets=len(attack_targets),
            )
            return

        state.phase = "harvest_takeover_finalize"
        log_event(
            rnd,
            uid,
            "economy",
            f"({cur_xy[0]},{cur_xy[1]})",
            "economy_harvester_takeover_no_adjacent_targets",
            ox=ox,
            oy=oy,
            hid=(state.harvest_takeover_harvester_id if isinstance(
                state.harvest_takeover_harvester_id, int) else -1),
        )
        return


def _start_enemy_harvester_takeover(
    state: EconomyState,
    cur_xy,
    ore_xy,
    harvester_id: int | None,
    rnd: int,
    uid: int,
):
    ox, oy = int(ore_xy[0]), int(ore_xy[1])
    state.harvest_takeover_harvester_id = harvester_id if isinstance(
        harvester_id, int) else None
    state.harvest_takeover_ore_xy = (ox, oy)
    state.harvest_ore_xy = (ox, oy)
    state.harvest_goal_xy = None
    state.plan_steps = ()
    state.plan_index = 0
    state.defer_step_once = False
    state.phase = "harvest_takeover_pick_goal"
    log_event(
        rnd,
        uid,
        "economy",
        f"({cur_xy[0]},{cur_xy[1]})",
        "economy_harvester_takeover_target_selected",
        ox=ox,
        oy=oy,
        hid=(state.harvest_takeover_harvester_id if isinstance(
            state.harvest_takeover_harvester_id, int) else -1),
    )


def _finalize_harvester_takeover(
    state: EconomyState,
    local_map,
    cur_xy,
    rnd: int,
    uid: int,
):
    ore_xy = state.harvest_takeover_ore_xy
    if not (isinstance(ore_xy, tuple) and len(ore_xy) == 2):
        state.harvest_takeover_harvester_id = None
        state.harvest_takeover_ore_xy = None
        state.phase = "harvest_pick_ore"
        return

    ox, oy = int(ore_xy[0]), int(ore_xy[1])
    rec = _known_building_at(local_map, ox, oy)
    if not isinstance(rec, dict) or rec.get("entity_type") != EntityType.HARVESTER:
        state.harvest_takeover_harvester_id = None
        state.harvest_takeover_ore_xy = None
        state.harvest_ore_xy = None
        state.harvest_goal_xy = None
        state.phase = "harvest_pick_ore"
        log_event(
            rnd,
            uid,
            "economy",
            f"({cur_xy[0]},{cur_xy[1]})",
            "economy_harvester_takeover_aborted",
            ox=ox,
            oy=oy,
        )
        return

    takeover_id = state.harvest_takeover_harvester_id
    if not isinstance(takeover_id, int):
        takeover_id = _entity_id_from_rec(rec)
    _mark_harvester_and_adjacent_transports_stolen(
        state,
        local_map,
        (ox, oy),
        takeover_id,
    )

    state.phase = "network_select_candidate"
    state.network_wait_logged = False
    state.network_target = None
    _reset_network_path_state(state)
    state.network_escape_bridge_target = None
    state.network_highway_pending_harvester = (ox, oy)
    state.network_highway_active_harvester = None
    state.active_network_id = _ensure_active_network_id_for_harvester(
        state,
        (ox, oy),
        rnd,
    )
    state.harvest_blocked_ores.discard((ox, oy))
    state.harvest_goal_xy = None
    state.harvest_ore_xy = None
    state.harvest_takeover_harvester_id = None
    state.harvest_takeover_ore_xy = None
    state.plan_steps = ()
    state.plan_index = 0
    state.defer_step_once = False

    log_event(
        rnd,
        uid,
        "economy",
        f"({cur_xy[0]},{cur_xy[1]})",
        "economy_harvester_takeover_committed",
        ox=ox,
        oy=oy,
        hid=(takeover_id if isinstance(takeover_id, int) else -1),
    )
    log_event(
        rnd,
        uid,
        "economy",
        f"({cur_xy[0]},{cur_xy[1]})",
        "economy_network_highway_fallback_pending_v2",
        hx=ox,
        hy=oy,
    )


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
    state.harvest_takeover_harvester_id = None
    state.harvest_takeover_ore_xy = None
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


def _refresh_turret_threat_costs(state: EconomyState, local_map, rnd: int):
    if state.turret_threat_round == rnd:
        return

    state.turret_threat_round = rnd
    state.turret_threat_cost = {}
    state.turret_threat_sources = {}

    entities = getattr(local_map, "entities", None)
    my_team = getattr(local_map, "my_team", None)
    if not isinstance(entities, dict):
        return

    for rec in entities.values():
        if not isinstance(rec, dict):
            continue
        if not rec.get("alive", False):
            continue
        if rec.get("team") == my_team:
            continue

        etype = rec.get("entity_type")
        if etype not in (
            EntityType.GUNNER,
            EntityType.SENTINEL,
            EntityType.BREACH,
            EntityType.LAUNCHER,
        ):
            continue

        pos = rec.get("position")
        if not (isinstance(pos, tuple) and len(pos) == 2):
            continue

        tx = int(pos[0])
        ty = int(pos[1])
        if not local_map.in_bounds(tx, ty):
            continue

        for x, y in _iter_enemy_turret_threat_tiles(local_map, rec):
            dist_sq = (x - tx) * (x - tx) + (y - ty) * (y - ty)
            base = _TURRET_SOFT_COST.get(etype, 8)
            penalty = base - (dist_sq // 6)
            if penalty < 1:
                penalty = 1
            state.turret_threat_cost[(x, y)] = state.turret_threat_cost.get(
                (x, y), 0
            ) + penalty
            state.turret_threat_sources[(x, y)] = (
                state.turret_threat_sources.get((x, y), 0) + 1
            )


def _iter_enemy_turret_threat_tiles(local_map, rec):
    etype = rec.get("entity_type")
    pos = rec.get("position")
    if not (isinstance(pos, tuple) and len(pos) == 2):
        return ()

    tx = int(pos[0])
    ty = int(pos[1])

    direction = rec.get("direction")
    if etype == EntityType.LAUNCHER:
        return _iter_circle_tiles(local_map, tx, ty, 26, include_origin=False)

    if direction is None:
        r2 = 13
        if etype == EntityType.SENTINEL:
            r2 = 32
        return _iter_circle_tiles(local_map, tx, ty, r2, include_origin=False)

    dx, dy = direction.delta()

    if etype == EntityType.GUNNER:
        out = []
        k = 1
        while True:
            x = tx + (dx * k)
            y = ty + (dy * k)
            if not local_map.in_bounds(x, y):
                break
            if ((x - tx) * (x - tx) + (y - ty) * (y - ty)) > 13:
                break
            out.append((x, y))
            k += 1
        return tuple(out)

    if etype == EntityType.SENTINEL:
        out = set()
        k = 1
        while True:
            cx = tx + (dx * k)
            cy = ty + (dy * k)
            if not local_map.in_bounds(cx, cy):
                break
            if ((cx - tx) * (cx - tx) + (cy - ty) * (cy - ty)) > 32:
                break
            for ox in (-1, 0, 1):
                for oy in (-1, 0, 1):
                    x = cx + ox
                    y = cy + oy
                    if not local_map.in_bounds(x, y):
                        continue
                    if ((x - tx) * (x - tx) + (y - ty) * (y - ty)) > 32:
                        continue
                    out.add((x, y))
            k += 1
        return tuple(sorted(out))

    if etype == EntityType.BREACH:
        out = []
        for x in range(tx - 4, tx + 5):
            for y in range(ty - 4, ty + 5):
                if not local_map.in_bounds(x, y):
                    continue
                ddx = x - tx
                ddy = y - ty
                dist_sq = (ddx * ddx) + (ddy * ddy)
                if dist_sq == 0 or dist_sq > 13:
                    continue
                dot = (ddx * dx) + (ddy * dy)
                if dot < 0:
                    continue
                out.append((x, y))
        return tuple(out)

    return ()


def _iter_circle_tiles(local_map, cx: int, cy: int, r_sq: int, include_origin: bool):
    out = []
    axis = int(math.isqrt(max(0, r_sq)))
    for x in range(cx - axis - 1, cx + axis + 2):
        for y in range(cy - axis - 1, cy + axis + 2):
            if not local_map.in_bounds(x, y):
                continue
            dist_sq = ((x - cx) * (x - cx)) + ((y - cy) * (y - cy))
            if dist_sq > r_sq:
                continue
            if dist_sq == 0 and not include_origin:
                continue
            out.append((x, y))
    return tuple(out)


def _outside_future_barrier_ring(core_xy, tile_xy, min_cheb: int = _NO_LAUNCHER_ESCAPE_MIN_CHEB) -> bool:
    return _chebyshev(core_xy, tile_xy) >= int(min_cheb)


def _collect_barrier_area_tiles(local_map, core_xy, max_cheb: int):
    cx = int(core_xy[0])
    cy = int(core_xy[1])
    out = []
    for x in range(cx - max_cheb, cx + max_cheb + 1):
        for y in range(cy - max_cheb, cy + max_cheb + 1):
            if not local_map.in_bounds(x, y):
                continue
            if _chebyshev(core_xy, (x, y)) <= max_cheb:
                out.append((x, y))
    out.sort()
    return tuple(out)


def _compose_dynamic_blocked_tiles(state: EconomyState):
    blocked_tiles = set()
    blocked_tiles.update(
        (int(p[0]), int(p[1]))
        for p in state.highway_excluded_transport
        if isinstance(p, tuple) and len(p) == 2
    )
    if state.barrier_area_block_active:
        blocked_tiles.update(
            (int(p[0]), int(p[1]))
            for p in state.barrier_area_block_tiles
            if isinstance(p, tuple) and len(p) == 2
        )
    blocked_tiles.update(state.axionite_ti_route_blocked_tiles)
    return blocked_tiles


def _sync_dynamic_blocked_tiles(state: EconomyState, local_map):
    blocked_tiles = _compose_dynamic_blocked_tiles(state)
    changed_tiles = 0
    rebuilt_tiles = 0
    if hasattr(local_map, "set_dynamic_blocked_tiles"):
        changed_tiles, rebuilt_tiles = local_map.set_dynamic_blocked_tiles(
            tuple(sorted(blocked_tiles))
        )
    return blocked_tiles, changed_tiles, rebuilt_tiles


def _update_postlaunch_barrier_blocking(
    state: EconomyState,
    local_map,
    cur_xy,
    rnd: int,
    uid: int,
):
    if state.barrier_area_block_active:
        return

    if state.phase not in _POST_LAUNCH_PHASES:
        return

    if _chebyshev(state.core_xy, cur_xy) < _NO_LAUNCHER_ESCAPE_MIN_CHEB:
        return

    state.barrier_area_block_tiles = _collect_barrier_area_tiles(
        local_map,
        state.core_xy,
        _BARRIER_AREA_CHEB,
    )
    state.barrier_area_block_active = True
    blocked_tiles, changed_tiles, rebuilt_tiles = _sync_dynamic_blocked_tiles(
        state,
        local_map,
    )
    log_event(
        rnd,
        uid,
        "economy",
        f"({cur_xy[0]},{cur_xy[1]})",
        "economy_barrier_area_block_enabled",
        cheb=_chebyshev(state.core_xy, cur_xy),
        blocked_tiles=len(state.barrier_area_block_tiles),
        dynamic_total=len(blocked_tiles),
        changed=changed_tiles,
        rebuilt=rebuilt_tiles,
    )


# ===== PRE-LAUNCH NAVIGATION =====
# Bot movement before the launcher fires it to its destination.
# Phases: seek_launcher → plan_to_launcher → follow_plan → wait_to_launch
#         prelaunch_escape_pick_goal → prelaunch_escape_plan → prelaunch_escape_follow

def _reset_prelaunch_navigation_state(state: EconomyState):
    state.goal_xy = None
    state.plan_steps = ()
    state.plan_index = 0
    state.defer_step_once = False
    state.blocked_ticks = 0


def _pick_prelaunch_escape_goal(local_map, core_xy, cur_xy):
    if _outside_future_barrier_ring(core_xy, cur_xy, min_cheb=_NO_LAUNCHER_ESCAPE_MIN_CHEB):
        return (int(cur_xy[0]), int(cur_xy[1]))

    in_bounds = getattr(local_map, "in_bounds", None)
    if not callable(in_bounds):
        return None

    start = (int(cur_xy[0]), int(cur_xy[1]))
    frontier = deque([start])
    seen = {start}

    while frontier:
        x, y = frontier.popleft()
        if _outside_future_barrier_ring(core_xy, (x, y), min_cheb=_NO_LAUNCHER_ESCAPE_MIN_CHEB):
            return (x, y)

        for dx, dy in _ADJACENT_DELTAS_8:
            nx = x + dx
            ny = y + dy
            nxy = (nx, ny)
            if nxy in seen:
                continue
            if not in_bounds(nx, ny):
                continue
            if not _is_general_movement_passable(local_map, nx, ny):
                continue

            seen.add(nxy)
            frontier.append(nxy)

    return None


def _run_prelaunch_no_launcher(
    c: Controller,
    state: EconomyState,
    local_map,
    cur_xy,
    rnd: int,
    uid: int,
):
    if _maybe_build_core_launcher(c, state, local_map, cur_xy, rnd, uid):
        return
    if _maybe_build_core_perimeter_turret(c, state, local_map, cur_xy, rnd, uid):
        return

    if _outside_future_barrier_ring(state.core_xy, cur_xy, min_cheb=_NO_LAUNCHER_ESCAPE_MIN_CHEB):
        if state.phase not in _POST_LAUNCH_PHASES:
            _reset_prelaunch_navigation_state(state)
            state.wait_logged = False
            state.phase = "explore_generate_waypoints"
            log_event(
                rnd,
                uid,
                "economy",
                f"({cur_xy[0]},{cur_xy[1]})",
                "economy_no_launcher_escape_complete_enter_explore",
                cheb=_chebyshev(state.core_xy, cur_xy),
            )
        return

    if state.phase in ("seek_launcher", "plan_to_launcher", "follow_plan", "wait_to_launch"):
        _reset_prelaunch_navigation_state(state)
        state.phase = "prelaunch_escape_pick_goal"
        log_event(
            rnd,
            uid,
            "economy",
            f"({cur_xy[0]},{cur_xy[1]})",
            "economy_no_launcher_escape_begin",
            cheb=_chebyshev(state.core_xy, cur_xy),
        )
        return

    if state.phase == "prelaunch_escape_pick_goal":
        goal_xy = _pick_prelaunch_escape_goal(local_map, state.core_xy, cur_xy)
        if goal_xy is None:
            state.blocked_ticks += 1
            if state.blocked_ticks == 1 or (state.blocked_ticks % 20) == 0:
                log_event(
                    rnd,
                    uid,
                    "economy",
                    f"({cur_xy[0]},{cur_xy[1]})",
                    "economy_no_launcher_escape_pending",
                    cheb=_chebyshev(state.core_xy, cur_xy),
                    blocked=state.blocked_ticks,
                )
            return

        state.goal_xy = goal_xy
        state.blocked_ticks = 0
        state.phase = "prelaunch_escape_plan"
        log_event(
            rnd,
            uid,
            "economy",
            f"({cur_xy[0]},{cur_xy[1]})",
            "economy_no_launcher_escape_goal_selected",
            gx=goal_xy[0],
            gy=goal_xy[1],
            cheb=_chebyshev(state.core_xy, goal_xy),
        )
        return

    if state.phase == "prelaunch_escape_plan":
        goal_xy = state.goal_xy
        if not (isinstance(goal_xy, tuple) and len(goal_xy) == 2):
            state.phase = "prelaunch_escape_pick_goal"
            return

        if cur_xy == goal_xy:
            state.phase = "prelaunch_escape_pick_goal"
            return

        plan_budget = 512

        def passable_fn(x: int, y: int) -> bool:
            return _is_general_movement_passable(local_map, x, y, goal_xy)

        steps = _astar_cardinal_plan(
            local_map,
            cur_xy,
            goal_xy,
            max_expansions=plan_budget,
            tile_passable_fn=passable_fn,
        )

        if not steps:
            state.blocked_ticks += 1
            if state.blocked_ticks >= 2:
                state.goal_xy = None
                state.blocked_ticks = 0
                state.phase = "prelaunch_escape_pick_goal"
            return

        state.plan_steps = steps
        state.plan_index = 0
        state.defer_step_once = True
        state.blocked_ticks = 0
        state.phase = "prelaunch_escape_follow"
        log_event(
            rnd,
            uid,
            "economy",
            f"({cur_xy[0]},{cur_xy[1]})",
            "economy_no_launcher_escape_plan_ready",
            gx=goal_xy[0],
            gy=goal_xy[1],
            steps=len(steps),
            budget=plan_budget,
        )
        return

    if state.phase == "prelaunch_escape_follow":
        goal_xy = state.goal_xy
        if not (isinstance(goal_xy, tuple) and len(goal_xy) == 2):
            state.phase = "prelaunch_escape_pick_goal"
            return

        if cur_xy == goal_xy:
            _reset_prelaunch_navigation_state(state)
            state.phase = "prelaunch_escape_pick_goal"
            return

        if state.defer_step_once:
            state.defer_step_once = False
            return

        if state.plan_index >= len(state.plan_steps):
            state.phase = "prelaunch_escape_plan"
            return

        nxt = state.plan_steps[state.plan_index]
        if not _is_adjacent_step(cur_xy, nxt):
            state.phase = "prelaunch_escape_plan"
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

        state.blocked_ticks += 1
        if state.blocked_ticks >= 2:
            state.plan_steps = ()
            state.plan_index = 0
            state.defer_step_once = False
            state.phase = "prelaunch_escape_plan"
            state.blocked_ticks = 0
        return

    if state.phase not in _POST_LAUNCH_PHASES:
        state.phase = "prelaunch_escape_pick_goal"


def _maybe_build_core_launcher(
    c: Controller,
    state: EconomyState,
    local_map,
    cur_xy,
    rnd: int,
    uid: int,
) -> bool:
    if state.core_launcher_built:
        return False
    if rnd < 700 or _team_titanium(c) < 2800:
        return False
    if rnd - state.core_launcher_last_attempt_round < 12:
        return False
    if c.get_action_cooldown() > 0:
        return False
    if _chebyshev(state.core_xy, cur_xy) > 2:
        state.core_launcher_built = True
        return False
    if _has_core_launcher(c, state.core_xy):
        state.core_launcher_built = True
        return False

    try:
        cost_ti, _ = c.get_launcher_cost()
    except GameError:
        cost_ti = 20
    if _team_titanium(c) < cost_ti + 2500:
        return False

    state.core_launcher_last_attempt_round = rnd
    for target_xy in _core_launcher_candidates(local_map, state.core_xy, cur_xy):
        target = Position(target_xy[0], target_xy[1])
        try:
            if not c.can_build_launcher(target):
                continue
            c.build_launcher(target)
            state.core_launcher_built = True
            state.launcher_xy = target_xy
            log_event(
                rnd,
                uid,
                "economy",
                f"({cur_xy[0]},{cur_xy[1]})",
                "economy_core_launcher_built",
                tx=target_xy[0],
                ty=target_xy[1],
            )
            return True
        except GameError:
            continue

    return False


def _has_core_launcher(c: Controller, core_xy) -> bool:
    cx, cy = int(core_xy[0]), int(core_xy[1])
    my_team = c.get_team()
    for bid in c.get_nearby_buildings():
        try:
            if c.get_entity_type(bid) != EntityType.LAUNCHER:
                continue
            if c.get_team(bid) != my_team:
                continue
            p = c.get_position(bid)
            if max(abs(p.x - cx), abs(p.y - cy)) <= 3:
                return True
        except GameError:
            continue
    return False


def _core_launcher_candidates(local_map, core_xy, cur_xy):
    cx, cy = int(core_xy[0]), int(core_xy[1])
    candidates = []
    for dx, dy in ((2, 0), (-2, 0), (0, 2), (0, -2)):
        tx = cx + dx
        ty = cy + dy
        if not local_map.in_bounds(tx, ty):
            continue
        if (tx, ty) == (cx, cy - 2):
            continue
        ddx = tx - cur_xy[0]
        ddy = ty - cur_xy[1]
        if ddx * ddx + ddy * ddy > ACTION_RADIUS_SQ:
            continue
        tile = local_map.get(tx, ty)
        if tile not in (MAP_FREE, MAP_ROAD):
            continue
        candidates.append((abs(ddx) + abs(ddy), abs(dx) + abs(dy), tx, ty))

    candidates.sort()
    return tuple((tx, ty) for _a, _b, tx, ty in candidates)


def _maybe_build_core_perimeter_turret(
    c: Controller,
    state: EconomyState,
    local_map,
    cur_xy,
    rnd: int,
    uid: int,
) -> bool:
    """Let newly spawned economy bots add a tiny core turret perimeter.

    The gate is deliberately conservative: the first cheap gunner waits until
    the initial economy has had time to stand up, and the heavier sentinel only
    appears in the high-pool late game.
    """
    if state.core_turret_built:
        return False
    if rnd - state.core_turret_last_attempt_round < 12:
        return False
    if c.get_action_cooldown() > 0:
        return False
    if _chebyshev(state.core_xy, cur_xy) > 2:
        state.core_turret_built = True
        return False

    turret_count = _count_core_perimeter_turrets(c, state.core_xy)
    if turret_count >= 2:
        state.core_turret_built = True
        return False

    ti = _team_titanium(c)
    if turret_count >= 1 or rnd < 1000 or ti < 2800:
        return False
    try:
        cost_ti, _ = c.get_sentinel_cost()
    except GameError:
        cost_ti = 30
    if ti < cost_ti + 2500:
        return False
    turret_type = EntityType.SENTINEL
    build_name = "sentinel"

    state.core_turret_last_attempt_round = rnd
    for target_xy, direction in _core_perimeter_turret_candidates(
        local_map,
        state.core_xy,
        cur_xy,
    ):
        target = Position(target_xy[0], target_xy[1])
        try:
            if turret_type == EntityType.SENTINEL:
                if not c.can_build_sentinel(target, direction):
                    continue
                c.build_sentinel(target, direction)
            else:
                if not c.can_build_gunner(target, direction):
                    continue
                c.build_gunner(target, direction)
            state.core_turret_built = True
            log_event(
                rnd,
                uid,
                "economy",
                f"({cur_xy[0]},{cur_xy[1]})",
                "economy_core_turret_built",
                kind=build_name,
                tx=target_xy[0],
                ty=target_xy[1],
                direction=direction.name,
                existing=turret_count,
            )
            return True
        except GameError:
            continue

    return False


def _count_core_perimeter_turrets(c: Controller, core_xy) -> int:
    cx, cy = int(core_xy[0]), int(core_xy[1])
    count = 0
    my_team = c.get_team()
    for bid in c.get_nearby_buildings():
        try:
            etype = c.get_entity_type(bid)
            if etype not in (EntityType.GUNNER, EntityType.SENTINEL):
                continue
            if c.get_team(bid) != my_team:
                continue
            p = c.get_position(bid)
            if max(abs(p.x - cx), abs(p.y - cy)) <= 4:
                count += 1
        except GameError:
            continue
    return count


def _core_perimeter_turret_candidates(local_map, core_xy, cur_xy):
    cx, cy = int(core_xy[0]), int(core_xy[1])
    candidates = []
    for dx in (-3, -2, -1, 0, 1, 2, 3):
        for dy in (-3, -2, -1, 0, 1, 2, 3):
            if max(abs(dx), abs(dy)) not in (2, 3):
                continue
            tx = cx + dx
            ty = cy + dy
            if not local_map.in_bounds(tx, ty):
                continue
            if (tx, ty) == (cx, cy - 2):
                continue
            ddx = tx - cur_xy[0]
            ddy = ty - cur_xy[1]
            if ddx * ddx + ddy * ddy > ACTION_RADIUS_SQ:
                continue
            tile = local_map.get(tx, ty)
            if tile not in (MAP_FREE, MAP_ROAD):
                continue
            ndx = 0 if dx == 0 else (1 if dx > 0 else -1)
            ndy = 0 if dy == 0 else (1 if dy > 0 else -1)
            direction = _DIRECTION_BY_DELTA.get((ndx, ndy))
            if direction is None or direction == Direction.CENTRE:
                continue
            key = (
                0 if max(abs(dx), abs(dy)) == 2 else 1,
                abs(dx) + abs(dy),
                abs(ddx) + abs(ddy),
                tx,
                ty,
            )
            candidates.append((key, (tx, ty), direction))

    candidates.sort(key=lambda item: item[0])
    return tuple((xy, direction) for _key, xy, direction in candidates)


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


def _execute_step_toward(c: Controller, local_map, cur_xy, nxt_xy, rnd: int, uid: int):
    cx, cy = cur_xy
    nx, ny = nxt_xy
    dx = nx - cx
    dy = ny - cy
    move_dir = _DIRECTION_BY_DELTA.get((dx, dy))
    if move_dir is None:
        return "move_blocked"

    nxt_pos = Position(nx, ny)
    try:
        nxt_env = c.get_tile_env(nxt_pos)
        if nxt_env in (Environment.WALL, Environment.ORE_TITANIUM, Environment.ORE_AXIONITE):
            return "road_invalid"
    except GameError:
        pass

    needs_road = not _is_builder_directly_walkable_tile(local_map, nx, ny)

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
            existing = c.get_tile_building_id(nxt_pos)
            if existing is not None:
                try:
                    existing_type = c.get_entity_type(existing)
                except GameError:
                    return "road_invalid"
                if existing_type != EntityType.MARKER:
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
