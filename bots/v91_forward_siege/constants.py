from cambc import Direction, EntityType, GameConstants


def _gc(name: str, default):
    """Read a GameConstants value with a safe fallback."""
    return getattr(GameConstants, name, default)


MAP_UNKNOWN = 0
MAP_FREE = 1       # empty tile (not walkable by bots)
MAP_OBSTACLE = 2   # wall (natural)

# Walkable friendly buildings (bots can move on these)
MAP_ROAD = 3
MAP_CONVEYOR = 4
MAP_SPLITTER = 5
MAP_BRIDGE = 6
MAP_ARMOURED_CONV = 7
MAP_CORE = 8

# Non-walkable friendly buildings
MAP_BARRIER = 10
MAP_HARVESTER = 11
MAP_FOUNDRY = 12
MAP_LAUNCHER = 13
MAP_GUNNER = 14
MAP_SENTINEL = 15
MAP_BREACH = 16
MAP_MARKER = 17

# Natural resource tiles (static environment)
MAP_ORE_TITANIUM = 18
MAP_ORE_AXIONITE = 19

# Enemy buildings (non-walkable, mirrored)
MAP_ENEMY = 20

# Other bots (temporary obstacle)
MAP_OTHER_BOT = 21
MAP_FRIENDLY_BOT = 22
MAP_ENEMY_UNIT = 23
MAP_ENEMY_TURRET = 24

# Map assumptions from game rules.
MAP_MIN_SIZE = 20
MAP_MAX_SIZE = 50

# Builder bots can legally stand on these tile types.
WALKABLE_TILES = {MAP_ROAD, MAP_CONVEYOR, MAP_BRIDGE,
                  MAP_ARMOURED_CONV, MAP_SPLITTER, MAP_CORE}

# Planner-friendly passability (unknown/empty are treated as potentially
# traversable for incremental map discovery).
PASSABLE_TILES = {MAP_FREE, MAP_UNKNOWN, MAP_ROAD, MAP_CONVEYOR,
                  MAP_BRIDGE, MAP_ARMOURED_CONV, MAP_SPLITTER, MAP_CORE}

# Entity-to-tile encoding used by local map trackers and debug dumps.
MAP_TILE_BY_ENTITY = {
    EntityType.ROAD: MAP_ROAD,
    EntityType.CONVEYOR: MAP_CONVEYOR,
    EntityType.SPLITTER: MAP_SPLITTER,
    EntityType.BRIDGE: MAP_BRIDGE,
    EntityType.ARMOURED_CONVEYOR: MAP_ARMOURED_CONV,
    EntityType.CORE: MAP_CORE,
    EntityType.BARRIER: MAP_BARRIER,
    EntityType.HARVESTER: MAP_HARVESTER,
    EntityType.FOUNDRY: MAP_FOUNDRY,
    EntityType.LAUNCHER: MAP_LAUNCHER,
    EntityType.GUNNER: MAP_GUNNER,
    EntityType.SENTINEL: MAP_SENTINEL,
    EntityType.BREACH: MAP_BREACH,
    EntityType.MARKER: MAP_MARKER,
}

TURRET_ENTITY_TYPES = {
    EntityType.GUNNER,
    EntityType.SENTINEL,
    EntityType.BREACH,
    EntityType.LAUNCHER,
}

BUILDING_ENTITY_TYPES = {
    EntityType.CORE,
    EntityType.GUNNER,
    EntityType.SENTINEL,
    EntityType.BREACH,
    EntityType.LAUNCHER,
    EntityType.CONVEYOR,
    EntityType.SPLITTER,
    EntityType.ARMOURED_CONVEYOR,
    EntityType.BRIDGE,
    EntityType.HARVESTER,
    EntityType.FOUNDRY,
    EntityType.ROAD,
    EntityType.BARRIER,
    EntityType.MARKER,
}

WALKABLE_BUILDING_ENTITY_TYPES = {
    EntityType.ROAD,
    EntityType.CONVEYOR,
    EntityType.SPLITTER,
    EntityType.BRIDGE,
    EntityType.ARMOURED_CONVEYOR,
}

DIRECTIONAL_ENTITY_TYPES = {
    EntityType.CONVEYOR,
    EntityType.SPLITTER,
    EntityType.ARMOURED_CONVEYOR,
    EntityType.GUNNER,
    EntityType.SENTINEL,
    EntityType.BREACH,
}

# -------------------------------
# Rules-aligned game constants
# -------------------------------

# General
MAX_TURNS = _gc("MAX_TURNS", 2000)
MAX_TEAM_UNITS = _gc("MAX_TEAM_UNITS", 50)
STACK_SIZE = _gc("STACK_SIZE", 10)
STARTING_TITANIUM = _gc("STARTING_TITANIUM", 500)
STARTING_AXIONITE = _gc("STARTING_AXIONITE", 0)
PASSIVE_TITANIUM_AMOUNT = _gc("PASSIVE_TITANIUM_AMOUNT", 10)
PASSIVE_TITANIUM_INTERVAL = _gc("PASSIVE_TITANIUM_INTERVAL", 4)
AXIONITE_CONVERSION_TITANIUM_RATE = _gc("AXIONITE_CONVERSION_TITANIUM_RATE", 4)

# Radii (squared)
ACTION_RADIUS_SQ = _gc("ACTION_RADIUS_SQ", 2)
CORE_ACTION_RADIUS_SQ = _gc("CORE_ACTION_RADIUS_SQ", 8)
CORE_SPAWNING_RADIUS_SQ = _gc("CORE_SPAWNING_RADIUS_SQ", 2)
CORE_VISION_RADIUS_SQ = _gc("CORE_VISION_RADIUS_SQ", 36)
BUILDER_BOT_VISION_RADIUS_SQ = _gc("BUILDER_BOT_VISION_RADIUS_SQ", 20)
GUNNER_VISION_RADIUS_SQ = _gc("GUNNER_VISION_RADIUS_SQ", 13)
SENTINEL_VISION_RADIUS_SQ = _gc("SENTINEL_VISION_RADIUS_SQ", 32)
BREACH_VISION_RADIUS_SQ = _gc("BREACH_VISION_RADIUS_SQ", 2)
BREACH_ATTACK_RADIUS_SQ = _gc("BREACH_ATTACK_RADIUS_SQ", 13)
LAUNCHER_VISION_RADIUS_SQ = _gc("LAUNCHER_VISION_RADIUS_SQ", 26)
BRIDGE_TARGET_RADIUS_SQ = _gc("BRIDGE_TARGET_RADIUS_SQ", 9)

# Base costs (titanium, axionite)
BUILDER_BOT_BASE_COST = _gc("BUILDER_BOT_BASE_COST", (30, 0))
CONVEYOR_BASE_COST = _gc("CONVEYOR_BASE_COST", (3, 0))
SPLITTER_BASE_COST = _gc("SPLITTER_BASE_COST", (6, 0))
BRIDGE_BASE_COST = _gc("BRIDGE_BASE_COST", (20, 0))
ARMOURED_CONVEYOR_BASE_COST = _gc("ARMOURED_CONVEYOR_BASE_COST", (5, 5))
HARVESTER_BASE_COST = _gc("HARVESTER_BASE_COST", (20, 0))
ROAD_BASE_COST = _gc("ROAD_BASE_COST", (1, 0))
BARRIER_BASE_COST = _gc("BARRIER_BASE_COST", (3, 0))
FOUNDRY_BASE_COST = _gc("FOUNDRY_BASE_COST", (40, 0))
GUNNER_BASE_COST = _gc("GUNNER_BASE_COST", (10, 0))
SENTINEL_BASE_COST = _gc("SENTINEL_BASE_COST", (30, 0))
BREACH_BASE_COST = _gc("BREACH_BASE_COST", (15, 10))
LAUNCHER_BASE_COST = _gc("LAUNCHER_BASE_COST", (20, 0))

# Max HP
CORE_MAX_HP = _gc("CORE_MAX_HP", 500)
BUILDER_BOT_MAX_HP = _gc("BUILDER_BOT_MAX_HP", 40)
CONVEYOR_MAX_HP = _gc("CONVEYOR_MAX_HP", 20)
SPLITTER_MAX_HP = _gc("SPLITTER_MAX_HP", 20)
BRIDGE_MAX_HP = _gc("BRIDGE_MAX_HP", 20)
ARMOURED_CONVEYOR_MAX_HP = _gc("ARMOURED_CONVEYOR_MAX_HP", 50)
HARVESTER_MAX_HP = _gc("HARVESTER_MAX_HP", 30)
ROAD_MAX_HP = _gc("ROAD_MAX_HP", 5)
BARRIER_MAX_HP = _gc("BARRIER_MAX_HP", 30)
FOUNDRY_MAX_HP = _gc("FOUNDRY_MAX_HP", 50)
MARKER_MAX_HP = _gc("MARKER_MAX_HP", 1)
GUNNER_MAX_HP = _gc("GUNNER_MAX_HP", 40)
SENTINEL_MAX_HP = _gc("SENTINEL_MAX_HP", 30)
BREACH_MAX_HP = _gc("BREACH_MAX_HP", 60)
LAUNCHER_MAX_HP = _gc("LAUNCHER_MAX_HP", 30)

# Combat
BUILDER_BOT_ATTACK_DAMAGE = _gc("BUILDER_BOT_ATTACK_DAMAGE", 2)
BUILDER_BOT_ATTACK_COST = _gc("BUILDER_BOT_ATTACK_COST", (2, 0))
BUILDER_BOT_HEAL_COST = _gc("BUILDER_BOT_HEAL_COST", (1, 0))
BUILDER_BOT_SELF_DESTRUCT_DAMAGE = _gc("BUILDER_BOT_SELF_DESTRUCT_DAMAGE", 0)
HEAL_AMOUNT = _gc("HEAL_AMOUNT", 4)
GUNNER_DAMAGE = _gc("GUNNER_DAMAGE", 10)
GUNNER_AXIONITE_DAMAGE = _gc("GUNNER_AXIONITE_DAMAGE", 40)
GUNNER_FIRE_COOLDOWN = _gc("GUNNER_FIRE_COOLDOWN", 1)
GUNNER_AMMO_COST = _gc("GUNNER_AMMO_COST", 2)
GUNNER_ROTATE_COST = _gc("GUNNER_ROTATE_COST", (10, 0))
GUNNER_ROTATE_COOLDOWN = _gc("GUNNER_ROTATE_COOLDOWN", 1)
SENTINEL_DAMAGE = _gc("SENTINEL_DAMAGE", 18)
SENTINEL_FIRE_COOLDOWN = _gc("SENTINEL_FIRE_COOLDOWN", 3)
SENTINEL_AMMO_COST = _gc("SENTINEL_AMMO_COST", 10)
SENTINEL_STUN_DURATION = _gc("SENTINEL_STUN_DURATION", 5)
BREACH_DAMAGE = _gc("BREACH_DAMAGE", 40)
BREACH_SPLASH_DAMAGE = _gc("BREACH_SPLASH_DAMAGE", 20)
BREACH_FIRE_COOLDOWN = _gc("BREACH_FIRE_COOLDOWN", 1)
BREACH_AMMO_COST = _gc("BREACH_AMMO_COST", 5)
LAUNCHER_FIRE_COOLDOWN = _gc("LAUNCHER_FIRE_COOLDOWN", 1)

# Scale contribution (% points, additive)
SCALE_ROAD = 0.5
SCALE_CONVEYOR = 1.0
SCALE_SPLITTER = 1.0
SCALE_ARMOURED_CONVEYOR = 1.0
SCALE_BARRIER = 1.0
SCALE_HARVESTER = 5.0
SCALE_BRIDGE = 10.0
SCALE_GUNNER = 10.0
SCALE_BREACH = 10.0
SCALE_LAUNCHER = 10.0
SCALE_BUILDER_BOT = 20.0
SCALE_SENTINEL = 20.0
SCALE_FOUNDRY = 50.0

CARDINALS = [Direction.NORTH, Direction.EAST, Direction.SOUTH, Direction.WEST]
CARDINAL_DELTAS = [(0, -1), (1, 0), (0, 1), (-1, 0)]

# Delta → Direction mapping used by cardinal movers (combat grafts).
DIRECTION_BY_DELTA = {
    (0, -1): Direction.NORTH,
    (1,  0): Direction.EAST,
    (0,  1): Direction.SOUTH,
    (-1, 0): Direction.WEST,
}

# --- Role constants for marker-based role assignment ---
ROLE_DEFENDER = 0
ROLE_ECONOMY = 1
ROLE_HEALER = 2
# Combat grafts (added by v3_merged flip).
ROLE_MINER = 3
ROLE_RAIDER = 4
ROLE_SCOUT = 5
ROLE_REPAIR_PATROL = 6

# Roles acceptable in decoded marker reads. Values outside this set are
# treated as corrupt markers and fall through to fallback logic.
VALID_ROLES = frozenset({
    ROLE_DEFENDER,
    ROLE_ECONOMY,
    ROLE_HEALER,
    ROLE_MINER,
    ROLE_RAIDER,
    ROLE_SCOUT,
    ROLE_REPAIR_PATROL,
})

# Competition toggle.
# When True, all logging/timing/debug outputs are disabled.
COMPETITION_MODE = True

# Local-map debug controls.
SHOW_MAP_DEBUG = True
# Master switch for all local-map dump output in debug log.
ENABLE_LOCAL_MAP_DUMPS = True
# Optional adjacency-list dump (requires ENABLE_LOCAL_MAP_DUMPS).
ENABLE_ADJACENCY_LIST_DUMPS = False
MAP_DEBUG_DUMP_EVERY_N = 20
MAP_DEBUG_CORE_ONLY = False
SHOW_EVENT_DEBUG = True
RESET_LOGS_ON_START = True

# Local-map update throttles by unit type.
MAP_UPDATE_EVERY_N_CORE = 1
MAP_UPDATE_EVERY_N_BUILDER = 1
MAP_UPDATE_EVERY_N_TURRET = 1

# Dedicated in-barrier marker channel tile (relative to core centre).
# Keep this tile free of permanent constructions so the core can always
# communicate the next spawn role.
MARKER_TILE_DX = 0
MARKER_TILE_DY = -2

# Ordered fallback marker positions around the core. The first entry keeps
# backward compatibility with the legacy fixed marker tile.
MARKER_TILE_CANDIDATE_DELTAS = (
    (MARKER_TILE_DX, MARKER_TILE_DY),
    (2, 0),
    (0, 2),
    (-2, 0),
    (1, -2),
    (2, -1),
    (2, 1),
    (1, 2),
    (-1, 2),
    (-2, 1),
    (-2, -1),
    (-1, -2),
    (2, -2),
    (2, 2),
    (-2, 2),
    (-2, -2),
)

# --- Message constants (high bit set to distinguish from roles) ---
MSG_DEFENSE_COMPLETE = 0x80  # defender → core: barrier wall done
# Landing pad is always at (cx+3, cy) relative to core centre


def marker_tile_candidates(cx: int, cy: int, map_w: int, map_h: int):
    """Return in-bounds marker candidate tiles around core in priority order."""
    out = []
    seen = set()
    for dx, dy in MARKER_TILE_CANDIDATE_DELTAS:
        mx = cx + dx
        my = cy + dy
        if not (0 <= mx < map_w and 0 <= my < map_h):
            continue
        key = (mx, my)
        if key in seen:
            continue
        seen.add(key)
        out.append(key)
    return tuple(out)


def encode_marker(role: int, cx: int, cy: int) -> int:
    """Encode role and core position into a 32-bit unsigned marker value."""
    return (role & 0xFF) | ((cx & 0xFF) << 8) | ((cy & 0xFF) << 16)


def decode_marker(value: int):
    """Decode a marker value into (role, cx, cy)."""
    role = value & 0xFF
    cx = (value >> 8) & 0xFF
    cy = (value >> 16) & 0xFF
    return role, cx, cy
