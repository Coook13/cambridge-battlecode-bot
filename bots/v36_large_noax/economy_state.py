import heapq
import math
import time
from collections import deque

from cambc import Controller, Direction, EntityType, Environment, GameError, Position

from constants import (ACTION_RADIUS_SQ, CARDINAL_DELTAS, MAP_FREE,
                       MAP_OBSTACLE, MAP_ORE_AXIONITE, MAP_ORE_TITANIUM,
                       MAP_ROAD, PASSABLE_TILES, WALKABLE_TILES)
from logger import log_event

# Movement planning supports all 8 directions.
_DIRECTION_BY_DELTA = {
    (0, -1): Direction.NORTH,
    (1, -1): Direction.NORTHEAST,
    (1, 0): Direction.EAST,
    (1, 1): Direction.SOUTHEAST,
    (0, 1): Direction.SOUTH,
    (-1, 1): Direction.SOUTHWEST,
    (-1, 0): Direction.WEST,
    (-1, -1): Direction.NORTHWEST,
}
_CARDINAL_DIRECTION_BY_DELTA = {
    (0, -1): Direction.NORTH,
    (1, 0): Direction.EAST,
    (0, 1): Direction.SOUTH,
    (-1, 0): Direction.WEST,
}
_CARDINAL_DIRECTION_BY_NAME = {
    direction.name: direction
    for direction in _CARDINAL_DIRECTION_BY_DELTA.values()
}

_ADJACENT_DELTAS_8 = (
    (-1, -1), (0, -1), (1, -1),
    (-1, 0),           (1, 0),
    (-1, 1),  (0, 1),  (1, 1),
)

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
_NETWORK_ASTAR_RESUME_STEP_EXPANSIONS = 256
_NETWORK_SELECT_TIMEOUT_US = 0
_NETWORK_TIMEOUT_RETRY_COOLDOWN_ROUNDS = 96
_NETWORK_UNREACHABLE_DIRECT_RETRY_COOLDOWN_ROUNDS = 48
_DIRECT_MASK_PREFILTER_MAX_EXPANSIONS = 768
_DIRECT_MASK_PREFILTER_TIMEOUT_US = 1200
_INDIRECT_TARGET_DENSITY_RADIUS_CHEB = 3
_INDIRECT_TARGET_MAX_BLOCKED_DENSITY = 0.68
_INDIRECT_TARGET_MIN_KNOWN_TILES = 24

_TRANSPORT_ENTITY_TYPES = (
    EntityType.CONVEYOR,
    EntityType.SPLITTER,
    EntityType.ARMOURED_CONVEYOR,
    EntityType.BRIDGE,
)
_DIRECTIONAL_TRANSPORT_ENTITY_TYPES = (
    EntityType.CONVEYOR,
    EntityType.SPLITTER,
)
_REPLACEABLE_ENEMY_BLOCKERS = (
    EntityType.ROAD,
    EntityType.CONVEYOR,
    EntityType.SPLITTER,
    EntityType.BRIDGE,
)
_LIDAR_RAYS_8 = (
    (0, -1),
    (1, -1),
    (1, 0),
    (1, 1),
    (0, 1),
    (-1, 1),
    (-1, 0),
    (-1, -1),
)
_LIDAR_DEBUG_MAX_HITS_PER_SCAN = 16
_TURRET_SOFT_COST = {
    EntityType.GUNNER: 20,
    EntityType.SENTINEL: 14,
    EntityType.BREACH: 28,
    EntityType.LAUNCHER: 8,
}


_POST_LAUNCH_PHASES = {
    "launched",
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
    "harvest_takeover_pick_goal",
    "harvest_takeover_plan_goal",
    "harvest_takeover_follow_plan",
    "harvest_takeover_attack_start",
    "harvest_takeover_finalize",
    "network_wait",
    "network_select_candidate",
    "network_plan_path",
    "network_plan_path_resume",
    "network_bridge_escape_check",
    "network_bridge_escape_execute",
    "network_attack_blocker",
    "conveyor_initialisation",
    "conveyor_execution",
    "conveyor_termination",
    "harvest_confirm_other_bot_building",
    "axionite_fallback_pick_consistent",
    "axionite_fallback_plan_goto",
    "axionite_fallback_goto",
    "axionite_fallback_clear_side",
    "axionite_fallback_replace_splitter",
    "axionite_fallback_replan",
    "repair_pick_target",
    "repair_plan_harvester",
    "repair_follow_harvester",
    "repair_attack_adjacent",
    "repair_attack_path",
    "repair_rebuild_plan",
    "repair_rebuild_follow",
    "repair_rebuild_build",
    "axionite_enter",
    "axionite_pick_ore",
    "axionite_pick_goal",
    "axionite_plan_goal",
    "axionite_follow_goal",
    "axionite_build_harvester",
    "axionite_pick_foundry",
    "axionite_step_off_foundry_tile",
    "axionite_pick_foundry_goal",
    "axionite_plan_foundry_goal",
    "axionite_follow_foundry_goal",
    "axionite_build_foundry",
    "axionite_pick_root",
    "axionite_pick_root_bridge",
    "axionite_plan_root_bridge",
    "axionite_follow_root_bridge",
    "axionite_build_root_bridge",
    "axionite_pick_titanium_pair",
    "axionite_pick_ti_route",
    "axionite_build_ti_route",
    "axionite_fallback_ti_pick_ore",
    "axionite_fallback_ti_pick_goal",
    "axionite_fallback_ti_plan_goal",
    "axionite_fallback_ti_follow_goal",
    "axionite_fallback_ti_build_harvester",
    "axionite_pick_core_route",
    "axionite_build_core_route",
    "axionite_done",
    "axionite_sabotage_pick",
    "axionite_sabotage_pick_goal",
    "axionite_sabotage_plan_goal",
    "axionite_sabotage_follow_goal",
    "axionite_sabotage_attack_start",
    "axionite_sabotage_finalize",
    "axionite_wait_resources",
    "network_wait_resources",
}

_NETWORK_BUILD_PHASES = {
    "network_wait",
    "network_select_candidate",
    "network_plan_path",
    "network_plan_path_resume",
    "network_bridge_escape_check",
    "network_bridge_escape_execute",
    "network_attack_blocker",
    "conveyor_initialisation",
    "conveyor_execution",
    "conveyor_termination",
    "network_wait_resources",
}

_HARVEST_TITANIUM_PHASES = {
    "harvest_enter",
    "harvest_pick_ore",
    "harvest_pick_goal",
    "harvest_plan_goal",
    "harvest_follow_plan",
    "harvest_build",
    "harvest_confirm_other_bot_building",
    "harvest_takeover_pick_goal",
    "harvest_takeover_plan_goal",
    "harvest_takeover_follow_plan",
    "harvest_takeover_attack_start",
    "harvest_takeover_finalize",
    "network_wait",
    "network_select_candidate",
    "network_plan_path",
    "network_plan_path_resume",
    "network_bridge_escape_check",
    "network_bridge_escape_execute",
    "network_attack_blocker",
    "conveyor_initialisation",
    "conveyor_execution",
    "conveyor_termination",
    "network_wait_resources",
}

_HARVEST_TAKEOVER_PHASES = {
    "harvest_takeover_pick_goal",
    "harvest_takeover_plan_goal",
    "harvest_takeover_follow_plan",
    "harvest_takeover_attack_start",
}

_HARVEST_AXIONITE_PHASES = {
    "axionite_enter",
    "axionite_pick_ore",
    "axionite_pick_goal",
    "axionite_plan_goal",
    "axionite_follow_goal",
    "axionite_build_harvester",
    "axionite_pick_foundry",
    "axionite_step_off_foundry_tile",
    "axionite_pick_foundry_goal",
    "axionite_plan_foundry_goal",
    "axionite_follow_foundry_goal",
    "axionite_build_foundry",
    "axionite_pick_root",
    "axionite_pick_root_bridge",
    "axionite_plan_root_bridge",
    "axionite_follow_root_bridge",
    "axionite_build_root_bridge",
    "axionite_pick_titanium_pair",
    "axionite_pick_ti_route",
    "axionite_build_ti_route",
    "axionite_fallback_ti_pick_ore",
    "axionite_fallback_ti_pick_goal",
    "axionite_fallback_ti_plan_goal",
    "axionite_fallback_ti_follow_goal",
    "axionite_fallback_ti_build_harvester",
    "axionite_pick_core_route",
    "axionite_build_core_route",
    "axionite_fallback_pick_consistent",
    "axionite_fallback_plan_goto",
    "axionite_fallback_goto",
    "axionite_fallback_clear_side",
    "axionite_fallback_replace_splitter",
    "axionite_fallback_replan",
    "axionite_done",
    "axionite_wait_resources",
}

_SABOTAGE_AXIONITE_PHASES = {
    "axionite_sabotage_pick",
    "axionite_sabotage_pick_goal",
    "axionite_sabotage_plan_goal",
    "axionite_sabotage_follow_goal",
    "axionite_sabotage_attack_start",
    "axionite_sabotage_finalize",
}

_PRELAUNCH_NO_LAUNCHER_PHASES = {
    "seek_launcher",
    "plan_to_launcher",
    "follow_plan",
    "wait_to_launch",
    "prelaunch_escape_pick_goal",
    "prelaunch_escape_plan",
    "prelaunch_escape_follow",
}

_REPAIR_PHASES = {
    "repair_pick_target",
    "repair_plan_harvester",
    "repair_follow_harvester",
    "repair_attack_adjacent",
    "repair_attack_path",
    "repair_rebuild_plan",
    "repair_rebuild_follow",
    "repair_rebuild_build",
}

_REPAIR_SCAN_PHASES = {
    "patrol_generate_waypoints",
    "patrol_replace_waypoint",
    "patrol_plan_waypoint",
    "patrol_follow_plan",
}
_REPAIR_SCAN_NETWORKS_PER_ROUND = 3
_REPAIR_MAX_PLAN_RETRIES = 12
_REPAIR_HANDOFF_REENQUEUE_COOLDOWN_ROUNDS = 24
_ORE_CORE_EXCLUSION_CHEB = 4
_ORE_CORE_EXCLUSION_MIN_CHEB = _ORE_CORE_EXCLUSION_CHEB
_BARRIER_AREA_CHEB = 4
_NO_LAUNCHER_ESCAPE_MIN_CHEB = _BARRIER_AREA_CHEB + 1
_AXIONITE_FOUNDRY_SEARCH_RADIUS = 10
_AXIONITE_CORE_NO_CANDIDATE_LIMIT = 12


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
        "barrier_area_block_active",
        "barrier_area_block_tiles",
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
        "harvest_takeover_harvester_id",
        "harvest_takeover_ore_xy",
        "harvest_blocked_ores",
        "network_wait_logged",
        "network_target",
        "network_path_nodes",
        "network_path_index",
        "network_plan_session",
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
        "next_network_id",
        "network_records",
        "network_id_by_harvester",
        "network_id_by_terminal",
        "provisional_network_ids",
        "broken_network_ids",
        "broken_network_tiles_by_id",
        "broken_transport_tiles",
        "built_entity_ids",
        "built_transport_positions",
        "built_harvester_positions",
        "direct_anchor_available",
        "direct_anchor_blocked",
        "attack_target_xy",
        "attack_targets",
        "attack_target_index",
        "attack_return_xy",
        "attack_resume_phase",
        "attack_reason",
        "active_network_id",
        "highway_excluded_transport",
        "turret_threat_round",
        "turret_threat_cost",
        "turret_threat_sources",
        "last_registry_round",
        "network_unreachable_direct_until",
        "entity_db_round",
        "transport_ids_built",
        "transport_ids_external_friendly",
        "transport_ids_stolen",
        "transport_ids_friendly",
        "transport_stolen_positions",
        "harvester_ids_built",
        "harvester_ids_external_friendly",
        "harvester_ids_stolen",
        "harvester_ids_friendly",
        "harvester_stolen_positions",
        "repair_scan_cursor",
        "repair_pending_conveyors",
        "repair_handoff_cooldown_until",
        "repair_enqueue_round",
        "repair_resume_phase",
        "repair_target_xy",
        "repair_target_direction_name",
        "repair_target_network_id",
        "repair_target_expected_id",
        "repair_harvester_goal_xy",
        "repair_rebuild_sequence",
        "repair_rebuild_index",
        "repair_rebuild_avoid_tiles",
        "axionite_ctx",
        "axionite_ti_route_blocked_tiles",
        "axionite_core_link_blocked_tiles",
        "axionite_core_link_active",
        "axionite_core_link_root_xy",
        "axionite_fallback_splitter_xy",
        "axionite_fallback_splitter_dir_name",
        "axionite_fallback_side_tap_xy",
        "axionite_fallback_attempted",
        "axionite_fallback_blocked_tiles",
        "axionite_fallback_foundry_xy",
        "axionite_fallback_ore_xy",
        "axionite_fallback_return_xy",
        "harvest_build_pending_ore_xy",
        "harvest_build_confirm_deadline",
        "harvest_deprioritised_ores",
        "axionite_build_pending_ore_xy",
        "axionite_build_confirm_deadline",
        "resource_wait_active",
        "resource_wait_step_mode",
        "resource_wait_cost_ti",
        "resource_wait_cost_ax",
        "resource_wait_sample_round",
        "resource_wait_sample_ti",
        "resource_wait_sample_ax",
        "resource_wait_resume_phase",
        "resource_wait_owner",
        "resource_wait_started_round",
        "paused_axionite_plan",
        "paused_network_plan",
        "core_turret_built",
        "core_turret_last_attempt_round",
        "core_launcher_built",
        "core_launcher_last_attempt_round",
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
        self.barrier_area_block_active = False
        self.barrier_area_block_tiles: tuple[tuple[int, int], ...] = ()

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
        self.harvest_takeover_harvester_id: int | None = None
        self.harvest_takeover_ore_xy: tuple[int, int] | None = None
        self.harvest_blocked_ores = set()
        self.network_wait_logged = False
        self.network_target: dict | None = None
        self.network_path_nodes: tuple[tuple[int, int], ...] = ()
        self.network_path_index = 0
        self.network_plan_session: dict | None = None
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

        # Deterministic friendly-network registry
        self.next_network_id = 1
        self.network_records = {}
        self.network_id_by_harvester = {}
        self.network_id_by_terminal = {}
        self.provisional_network_ids = set()
        self.broken_network_ids = set()
        self.broken_network_tiles_by_id = {}
        self.broken_transport_tiles = set()

        # Self-built tracking
        self.built_entity_ids = set()
        self.built_transport_positions = set()
        self.built_harvester_positions = set()

        # Direct bridge anchor availability (live set)
        self.direct_anchor_available: tuple[tuple[int, int], ...] = ()
        self.direct_anchor_blocked: set[tuple[int, int]] = set()

        # Dedicated attack substate
        self.attack_target_xy: tuple[int, int] | None = None
        self.attack_targets: tuple[tuple[int, int], ...] = ()
        self.attack_target_index = 0
        self.attack_return_xy: tuple[int, int] | None = None
        self.attack_resume_phase: str | None = None
        self.attack_reason: str | None = None
        self.active_network_id: int | None = None
        self.highway_excluded_transport: tuple[tuple[int, int], ...] = ()

        # Per-round turret soft-block cache
        self.turret_threat_round = -1
        self.turret_threat_cost = {}
        self.turret_threat_sources = {}

        self.last_registry_round = -1
        self.network_unreachable_direct_until = {}

        # Per-round entity ownership registry.
        self.entity_db_round = -1
        self.transport_ids_built = set()
        self.transport_ids_external_friendly = set()
        self.transport_ids_stolen = set()
        self.transport_ids_friendly = set()
        self.transport_stolen_positions = set()
        self.harvester_ids_built = set()
        self.harvester_ids_external_friendly = set()
        self.harvester_ids_stolen = set()
        self.harvester_ids_friendly = set()
        self.harvester_stolen_positions = set()

        # Conveyor integrity repair state.
        self.repair_scan_cursor = 0
        self.repair_pending_conveyors = {}
        self.repair_handoff_cooldown_until = {}
        self.repair_enqueue_round = -1
        self.repair_resume_phase: str | None = None
        self.repair_target_xy: tuple[int, int] | None = None
        self.repair_target_direction_name: str | None = None
        self.repair_target_network_id: int | None = None
        self.repair_target_expected_id: int | None = None
        self.repair_harvester_goal_xy: tuple[int, int] | None = None
        self.repair_rebuild_sequence: tuple[tuple[int, int, str, int], ...] = (
        )
        self.repair_rebuild_index = 0
        self.repair_rebuild_avoid_tiles = set()

        # Tiles belonging to completed axionite ti-routes; treated as obstacles
        # during network building planning but passable during movement.
        self.axionite_ti_route_blocked_tiles: set[tuple[int, int]] = set()

        # Extra blocked tiles for the axionite root→core link build. While the
        # axionite root→core network build is in progress, these tiles are the
        # ti→foundry feeder conveyors/bridges — treating them as obstacles
        # prevents the refined-axionite link from merging into the titanium
        # network (which would misroute refined axionite away from the core).
        self.axionite_core_link_blocked_tiles: set[tuple[int, int]] = set()
        # Marks whether the active network_* pipeline run was entered for an
        # axionite root→core link (not a titanium harvester). Used to route
        # pipeline completion/failure back to the axionite pipeline phases.
        self.axionite_core_link_active: bool = False
        # Root tile when the core-link pipeline is active. Kept here so the
        # pipeline can resume/complete independent of ctx contents.
        self.axionite_core_link_root_xy: tuple[int, int] | None = None

        # Axionite fallback (splitter-splice) state.
        self.axionite_fallback_splitter_xy: tuple[int, int] | None = None
        self.axionite_fallback_splitter_dir_name: str | None = None
        self.axionite_fallback_side_tap_xy: tuple[int, int] | None = None
        self.axionite_fallback_attempted: bool = False
        self.axionite_fallback_blocked_tiles: set[tuple[int, int]] = set()
        self.axionite_fallback_foundry_xy: tuple[int, int] | None = None
        self.axionite_fallback_ore_xy: tuple[int, int] | None = None
        self.axionite_fallback_return_xy: tuple[int, int] | None = None

        # Ore-contention confirm state (concern 4): when a build fails, next
        # round we verify whether another friendly bot built on that ore.
        self.harvest_build_pending_ore_xy: tuple[int, int] | None = None
        self.harvest_build_confirm_deadline: int = -1
        self.harvest_deprioritised_ores: set[tuple[int, int]] = set()
        self.axionite_build_pending_ore_xy: tuple[int, int] | None = None
        self.axionite_build_confirm_deadline: int = -1

        # Resource-wait: when a conveyor/bridge build fails because team Ti/Ax
        # is below the scaled cost, we enter a wait phase and sample income
        # rate instead of treating the failure as an obstacle.
        self.resource_wait_active: bool = False
        self.resource_wait_step_mode: str | None = None
        self.resource_wait_cost_ti: int = 0
        self.resource_wait_cost_ax: int = 0
        self.resource_wait_sample_round: int = -1
        self.resource_wait_sample_ti: int = -1
        self.resource_wait_sample_ax: int = -1
        self.resource_wait_resume_phase: str | None = None
        self.resource_wait_owner: str | None = None
        self.resource_wait_started_round: int = -1

        # Paused plans: when projected wait > threshold, stash the in-progress
        # plan and fall back to another objective. Resumed when resources are
        # available again.
        self.paused_axionite_plan: dict | None = None
        self.paused_network_plan: dict | None = None

        # v8_turrets: one-shot economy-built core perimeter turret attempt.
        self.core_turret_built = False
        self.core_turret_last_attempt_round = 0
        self.core_launcher_built = False
        self.core_launcher_last_attempt_round = 0

        # Axionite refining pipeline context.
        self.axionite_ctx = {
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
        }
