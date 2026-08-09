"""Per-unit local map with cached entity metadata and symmetry inference."""

import math

from cambc import Controller, EntityType, Environment, GameError, Position

from constants import (BUILDING_ENTITY_TYPES, DIRECTIONAL_ENTITY_TYPES,
                       MAP_MAX_SIZE, MAP_MIN_SIZE,
                       MAP_CORE, MAP_ENEMY, MAP_ENEMY_TURRET,
                       MAP_ENEMY_UNIT, MAP_FREE, MAP_FRIENDLY_BOT,
                       MAP_OBSTACLE, MAP_ORE_AXIONITE,
                       MAP_ORE_TITANIUM, MAP_OTHER_BOT,
                       MAP_TILE_BY_ENTITY, MAP_UNKNOWN,
                       PASSABLE_TILES,
                       TURRET_ENTITY_TYPES)


class UnitLocalMap:
    """Per-unit searchable local map with entity-id caching and symmetry inference."""

    # 8-neighborhood for builder movement and A* expansion.
    _ADJ_DELTAS_8 = (
        (-1, -1), (0, -1), (1, -1),
        (-1, 0),            (1, 0),
        (-1, 1),  (0, 1),  (1, 1),
    )

    _ADJ_DELTAS_9 = _ADJ_DELTAS_8 + ((0, 0),)
    _SYMMETRY_CHOICES = ("VERTICAL", "HORIZONTAL", "ROTATIONAL")
    _STATIC_SYMMETRY_CODES = (
        MAP_FREE,
        MAP_OBSTACLE,
        MAP_ORE_TITANIUM,
        MAP_ORE_AXIONITE,
    )

    def __init__(
        self,
        width: int,
        height: int,
        symmetry: str,
        my_team,
        infer_symmetry: bool = True,
        enable_adjacency: bool = True,
        vision_only: bool = False,
    ):
        # Coordinate frame uses top-left origin: (0, 0) is top-left.
        self.width = width
        self.height = height
        if symmetry not in self._SYMMETRY_CHOICES:
            symmetry = "ROTATIONAL"
        self.symmetry = symmetry
        self.initial_symmetry = symmetry
        self.symmetry_revision = 0
        self.my_team = my_team
        self.infer_symmetry = infer_symmetry
        self.enable_adjacency = enable_adjacency
        self.vision_only = vision_only
        self.coordinate_origin = (0, 0)
        self.view_x0 = 0
        self.view_y0 = 0
        self.assumption_violations = []
        if width != height:
            self.assumption_violations.append("non_square_map")
        if not MAP_MIN_SIZE <= width <= MAP_MAX_SIZE:
            self.assumption_violations.append("size_out_of_expected_range")

        self.env_grid: list[list[int]] = [
            [MAP_UNKNOWN for _ in range(width)] for _ in range(height)]
        self.env_observed: list[list[bool]] = [
            [False for _ in range(width)] for _ in range(height)]
        self.inferred_tile: list[list[int | None]] = [
            [None for _ in range(width)] for _ in range(height)]
        self.tile_building_id: list[list[int | None]] = [
            [None for _ in range(width)] for _ in range(height)]
        self.tile_unit_id: list[list[int | None]] = [
            [None for _ in range(width)] for _ in range(height)]
        self.visible_round: list[list[int]] = [
            [-1 for _ in range(width)] for _ in range(height)]

        self.entities = {}
        self.live_entity_ids = set()
        self.bridge_targets = {}
        self.friendly_foundries = set()

        self.friendly_core_pos = None
        self.inferred_enemy_core_center = None
        self.enemy_core_observed_center = None

        self.titanium_unharvested = set()
        self.titanium_harvested = set()
        self.axionite_unharvested = set()
        self.axionite_harvested = set()
        self.titanium_harvesters = set()
        self.axionite_harvesters = set()

        self.current_round = 0
        self._round_metrics = {}
        self.last_metrics = {}

        # Incremental adjacency cache for path planners.
        # Key: (x, y), Value: tuple[(nx, ny), ...] of passable neighbors.
        self.passability_grid = [
            [True for _ in range(width)] for _ in range(height)]
        self.adjacency = {}
        self._adj_dirty = set()
        self.dynamic_blocked_tiles = set()
        self.unit_halo_blocked_tiles = set()
        # Defer full adjacency bootstrap until the first post-spawn
        # map update to keep spawn-tick initialization lightweight.
        self._adjacency_bootstrapped = not self.enable_adjacency

    def _mirror_xy(self, x: int, y: int):
        return self._mirror_xy_for(x, y, self.symmetry)

    def _mirror_xy_for(self, x: int, y: int, symmetry: str):
        if symmetry == "VERTICAL":
            return self.width - 1 - x, y
        if symmetry == "HORIZONTAL":
            return x, self.height - 1 - y
        return self.width - 1 - x, self.height - 1 - y

    def _reset_round_metrics(self):
        self._round_metrics = {
            "map_nearby_tiles_calls": 0,
            "map_nearby_entities_calls": 0,
            "map_tile_env_calls": 0,
            "map_entity_pos_calls": 0,
            "map_entity_type_calls": 0,
            "map_entity_team_calls": 0,
            "map_direction_calls": 0,
            "map_bridge_target_calls": 0,
            "map_known_building_hits": 0,
            "map_reconciled_entities": 0,
            "map_adj_dirty_tiles": 0,
            "map_adj_rebuilt_tiles": 0,
            "map_symmetry_conflicts": 0,
            "map_symmetry_switches": 0,
        }

    def _bump(self, key: str, amount: int = 1):
        self._round_metrics[key] = self._round_metrics.get(key, 0) + amount

    def consume_round_metrics(self):
        return dict(self.last_metrics)

    def _build_initial_adjacency(self):
        for y in range(self.height):
            for x in range(self.width):
                self._adj_dirty.add((x, y))
        return self._flush_adjacency_updates()

    def _compute_tile_passability(self, x: int, y: int) -> bool:
        planner_self_xy = getattr(self, "planner_self_xy", None)
        if isinstance(planner_self_xy, tuple) and len(planner_self_xy) == 2:
            if x == int(planner_self_xy[0]) and y == int(planner_self_xy[1]):
                return True

        if (x, y) in self.dynamic_blocked_tiles:
            return False
        if (x, y) in self.unit_halo_blocked_tiles:
            return False

        # Temporary unit occupancy should block planner adjacency.
        if self.get_known_unit(x, y) is not None:
            return False

        bid = self.tile_building_id[y][x]
        if bid is not None:
            rec = self.entities.get(bid)
            if rec is not None and rec.get("alive", False):
                tile = rec["tile_code"]
            else:
                tile = MAP_OBSTACLE
        else:
            inferred = self.inferred_tile[y][x]
            tile = inferred if inferred is not None else self.env_grid[y][x]

        if tile in (MAP_ORE_TITANIUM, MAP_ORE_AXIONITE):
            return False
        return tile in PASSABLE_TILES

    def _mark_adjacency_dirty(self, x: int, y: int):
        if not self.in_bounds(x, y):
            return
        # Rebuild this tile and all neighbors because each edge is bidirectional.
        for dx, dy in self._ADJ_DELTAS_9:
            nx = x + dx
            ny = y + dy
            if self.in_bounds(nx, ny):
                self._adj_dirty.add((nx, ny))

    def _mark_adjacency_if_changed(self, x: int, y: int):
        if not self.enable_adjacency:
            return
        if not self.in_bounds(x, y):
            return
        previous = self.passability_grid[y][x]
        current = self._compute_tile_passability(x, y)
        if previous != current:
            self._mark_adjacency_dirty(x, y)

    def _rebuild_adjacency_tile(self, x: int, y: int):
        key = (x, y)
        if not self.passability_grid[y][x]:
            self.adjacency.pop(key, None)
            return

        neighbors = []
        for dx, dy in self._ADJ_DELTAS_8:
            nx = x + dx
            ny = y + dy
            if self.in_bounds(nx, ny) and self.passability_grid[ny][nx]:
                neighbors.append((nx, ny))
        self.adjacency[key] = tuple(neighbors)

    def _flush_adjacency_updates(self):
        if not self.enable_adjacency:
            self._adj_dirty.clear()
            return 0, 0
        if not self._adj_dirty:
            return 0, 0
        dirty_tiles = tuple(self._adj_dirty)
        self._adj_dirty.clear()

        for x, y in dirty_tiles:
            self.passability_grid[y][x] = self._compute_tile_passability(x, y)

        for x, y in dirty_tiles:
            self._rebuild_adjacency_tile(x, y)
        return len(dirty_tiles), len(dirty_tiles)

    def _refresh_unit_halo_blocked_tiles(self):
        new_blocked = set()
        for rec in self.entities.values():
            if not isinstance(rec, dict):
                continue
            if not rec.get("alive", False):
                continue
            if rec.get("is_building", False):
                continue

            pos = rec.get("position")
            if not (isinstance(pos, tuple) and len(pos) == 2):
                continue
            ux = int(pos[0])
            uy = int(pos[1])

            for dx in (-1, 0, 1):
                for dy in (-1, 0, 1):
                    tx = ux + dx
                    ty = uy + dy
                    if self.in_bounds(tx, ty):
                        new_blocked.add((tx, ty))

        if new_blocked == self.unit_halo_blocked_tiles:
            return 0

        changed = self.unit_halo_blocked_tiles.symmetric_difference(
            new_blocked)
        self.unit_halo_blocked_tiles = new_blocked

        for x, y in changed:
            self._mark_adjacency_dirty(x, y)

        return len(changed)

    def get_adjacency_neighbors(self, x: int, y: int):
        if not self.enable_adjacency:
            return ()
        cached = self.adjacency.get((x, y))
        if cached is not None:
            return cached
        if not self.in_bounds(x, y):
            return ()
        if not self._compute_tile_passability(x, y):
            return ()

        # Fallback for uncached tiles while adjacency bootstrap is deferred.
        neighbors = []
        for dx, dy in self._ADJ_DELTAS_8:
            nx = x + dx
            ny = y + dy
            if self.in_bounds(nx, ny) and self._compute_tile_passability(nx, ny):
                neighbors.append((nx, ny))
        return tuple(neighbors)

    def get_adjacency_list(self):
        if not self.enable_adjacency:
            return {}
        return self.adjacency

    def set_dynamic_blocked_tiles(self, blocked_tiles):
        normalized = set()
        for p in blocked_tiles:
            if isinstance(p, tuple) and len(p) == 2:
                normalized.add((int(p[0]), int(p[1])))

        if normalized == self.dynamic_blocked_tiles:
            return 0, 0

        changed = self.dynamic_blocked_tiles.symmetric_difference(normalized)
        self.dynamic_blocked_tiles = normalized

        for x, y in changed:
            self._mark_adjacency_dirty(x, y)

        return self._flush_adjacency_updates()

    def in_bounds(self, x: int, y: int) -> bool:
        return 0 <= x < self.width and 0 <= y < self.height

    def is_visible(self, x: int, y: int, max_age_rounds: int = 0) -> bool:
        if not self.in_bounds(x, y):
            return False
        return self.visible_round[y][x] >= (self.current_round - max_age_rounds)

    def get_known_building(self, x: int, y: int):
        if not self.in_bounds(x, y):
            return None
        bid = self.tile_building_id[y][x]
        if bid is None:
            return None
        rec = self.entities.get(bid)
        if rec is None or not rec.get("alive", False):
            return None
        return rec

    def get_known_unit(self, x: int, y: int):
        if not self.in_bounds(x, y):
            return None
        uid = self.tile_unit_id[y][x]
        if uid is None:
            return None
        rec = self.entities.get(uid)
        if rec is None or not rec.get("alive", False):
            return None
        return rec

    def is_symmetry_fully_resolved(self) -> bool:
        return self._symmetry_from_enemy_core() is not None

    def is_tile_known(self, x: int, y: int) -> bool:
        if not self.in_bounds(x, y):
            return False
        if self.env_observed[y][x]:
            return True
        if not self.infer_symmetry:
            return False
        if not self.is_symmetry_fully_resolved():
            return False
        return self.inferred_tile[y][x] is not None

    def get_known_unharvested_titanium(self):
        out = []
        for x, y in self.titanium_unharvested:
            if self.is_tile_known(x, y):
                out.append((x, y))
        return out

    def get_known_unharvested_axionite(self):
        out = []
        for x, y in self.axionite_unharvested:
            if self.is_tile_known(x, y):
                out.append((x, y))
        return out

    def set_fixed_obstacle(self, x: int, y: int):
        if not self.in_bounds(x, y):
            return
        self.env_grid[y][x] = MAP_OBSTACLE
        self.env_observed[y][x] = True
        self.inferred_tile[y][x] = None
        self._set_ore_state(x, y, MAP_OBSTACLE)
        self._mark_adjacency_if_changed(x, y)
        self._flush_adjacency_updates()

    def set_friendly_core(self, core_pos: Position):
        self.friendly_core_pos = core_pos
        self._reproject_inferred_tiles()
        self._flush_adjacency_updates()

    def get(self, x: int, y: int) -> int:
        if not self.in_bounds(x, y):
            return MAP_OBSTACLE

        uid = self.tile_unit_id[y][x]
        if uid is not None:
            rec = self.entities.get(uid)
            if rec is None:
                return MAP_OTHER_BOT
            return rec["tile_code"]

        bid = self.tile_building_id[y][x]
        if bid is not None:
            rec = self.entities.get(bid)
            if rec is None:
                return MAP_OBSTACLE
            return rec["tile_code"]

        inferred = self.inferred_tile[y][x]
        if inferred is not None:
            return inferred

        return self.env_grid[y][x]

    def ore_marker(self, x: int, y: int):
        p = (x, y)
        if p in self.titanium_harvesters:
            return "T"
        if p in self.axionite_harvesters:
            return "A"
        if p in self.titanium_unharvested or p in self.titanium_harvested:
            return "t"
        if p in self.axionite_unharvested or p in self.axionite_harvested:
            return "a"
        return None

    def conveyor_marker(self, x: int, y: int):
        if not self.in_bounds(x, y):
            return None
        bid = self.tile_building_id[y][x]
        if bid is None:
            return None
        rec = self.entities.get(bid)
        if rec is None:
            return None
        direction = rec.get("direction")
        if direction is None:
            return None
        if direction.name == "NORTH":
            return "^"
        if direction.name == "EAST":
            return ">"
        if direction.name == "SOUTH":
            return "v"
        if direction.name == "WEST":
            return "<"
        return "*"

    def is_friendly_foundry(self, x: int, y: int) -> bool:
        return (x, y) in self.friendly_foundries

    def update_from_controller(self, c: Controller):
        self.current_round = c.get_current_round()
        self._reset_round_metrics()

        try:
            my_id = c.get_id()
        except GameError:
            my_id = None

        try:
            nearby_tiles = list(c.get_nearby_tiles())
            self._bump("map_nearby_tiles_calls")
        except GameError:
            nearby_tiles = []

        visible_positions = []
        for t in nearby_tiles:
            x, y = t.x, t.y
            if not self.in_bounds(x, y):
                continue
            visible_positions.append((x, y))
            self.visible_round[y][x] = self.current_round
            if self.env_observed[y][x]:
                continue
            try:
                env = c.get_tile_env(t)
                self._bump("map_tile_env_calls")
            except GameError:
                continue
            code = self._env_to_map_code(env)
            self.env_grid[y][x] = code
            self.env_observed[y][x] = True
            if not self._is_observed_enemy_core_footprint_tile(x, y):
                self.inferred_tile[y][x] = None
            self._set_ore_state(x, y, code)
            self._mark_adjacency_if_changed(x, y)
            switched_symmetry = False
            contradicted = False
            if self.infer_symmetry and self._is_static_symmetry_code(code):
                contradicted = self._is_static_conflict_under_symmetry(
                    x,
                    y,
                    code,
                    self.symmetry,
                )
                if contradicted:
                    self._bump("map_symmetry_conflicts")
                    switched_symmetry = self._maybe_revise_symmetry(
                        pivot_static=(x, y, code),
                    )

            if (
                self.infer_symmetry
                and self._is_static_symmetry_code(code)
                and not contradicted
                and not switched_symmetry
            ):
                self._infer_static_mirror(x, y, code)

        try:
            nearby_entities = list(c.get_nearby_entities())
            self._bump("map_nearby_entities_calls")
        except GameError:
            nearby_entities = []

        for entity_id in nearby_entities:
            if my_id is not None and entity_id == my_id:
                continue

            rec = self.entities.get(entity_id)
            if rec is not None and rec["is_building"] and rec.get("position") is not None:
                px, py = rec["position"]
                if self.in_bounds(px, py) and self.visible_round[py][px] == self.current_round:
                    self.tile_building_id[py][px] = entity_id
                    rec["last_seen_round"] = self.current_round
                    rec["alive"] = True
                    self.live_entity_ids.add(entity_id)
                    self._bump("map_known_building_hits")
                    continue

            try:
                pos = c.get_position(entity_id)
                self._bump("map_entity_pos_calls")
            except GameError:
                continue

            if not self.in_bounds(pos.x, pos.y):
                continue

            if rec is None:
                rec = self._register_new_entity(c, entity_id)
                if rec is None:
                    continue

            self._place_entity(rec, pos.x, pos.y)

        self._reconcile_missing()
        self._reconcile_inferred_buildings(visible_positions)
        self._refresh_unit_halo_blocked_tiles()
        adj_dirty, adj_rebuilt = self._flush_adjacency_updates()
        if self.enable_adjacency and not self._adjacency_bootstrapped:
            # Adjacency cache creation starts only after the first map update.
            self._adjacency_bootstrapped = True
        self._round_metrics["map_adj_dirty_tiles"] = adj_dirty
        self._round_metrics["map_adj_rebuilt_tiles"] = adj_rebuilt
        self.last_metrics = dict(self._round_metrics)

    def _register_new_entity(self, c: Controller, entity_id: int):
        try:
            entity_type = c.get_entity_type(entity_id)
            self._bump("map_entity_type_calls")
            team = c.get_team(entity_id)
            self._bump("map_entity_team_calls")
        except GameError:
            return None

        rec = {
            "id": entity_id,
            "entity_type": entity_type,
            "team": team,
            "is_building": entity_type in BUILDING_ENTITY_TYPES,
            "tile_code": self._entity_tile_code(entity_type, team),
            "position": None,
            "last_seen_round": self.current_round,
            "alive": True,
            "direction": None,
            "bridge_target": None,
        }

        if entity_type in DIRECTIONAL_ENTITY_TYPES:
            try:
                rec["direction"] = c.get_direction(entity_id)
                self._bump("map_direction_calls")
            except GameError:
                pass

        if entity_type == EntityType.BRIDGE:
            try:
                target = c.get_bridge_target(entity_id)
                rec["bridge_target"] = (target.x, target.y)
                self._bump("map_bridge_target_calls")
            except GameError:
                pass

        self.entities[entity_id] = rec
        return rec

    def _place_entity(self, rec, x: int, y: int):
        entity_id = rec["id"]
        old = rec.get("position")

        if old is not None:
            ox, oy = old
            if self.in_bounds(ox, oy):
                if rec["is_building"] and self.tile_building_id[oy][ox] == entity_id:
                    self.tile_building_id[oy][ox] = None
                if (not rec["is_building"]) and self.tile_unit_id[oy][ox] == entity_id:
                    self.tile_unit_id[oy][ox] = None
                self._mark_adjacency_if_changed(ox, oy)

        if rec["is_building"]:
            self.tile_building_id[y][x] = entity_id
            if rec["entity_type"] == EntityType.CORE and rec["team"] != self.my_team:
                self._on_enemy_core_observed(x, y)
            if rec["entity_type"] == EntityType.FOUNDRY and rec["team"] == self.my_team:
                self.friendly_foundries.add((x, y))
            if rec["entity_type"] == EntityType.BRIDGE and rec["bridge_target"] is not None:
                self.bridge_targets[(x, y)] = rec["bridge_target"]
            if rec["entity_type"] == EntityType.HARVESTER:
                self._mark_harvested_by_harvester(x, y)
        else:
            self.tile_unit_id[y][x] = entity_id

        self._mark_adjacency_if_changed(x, y)

        rec["position"] = (x, y)
        rec["last_seen_round"] = self.current_round
        rec["alive"] = True
        self.live_entity_ids.add(entity_id)

        if self.inferred_tile[y][x] in (MAP_CORE, MAP_ENEMY):
            # Keep inferred enemy-core footprint pinned so transient entity
            # reconciliation cannot create holes in the known 3x3 core area.
            if not self._is_observed_enemy_core_footprint_tile(x, y):
                self.inferred_tile[y][x] = None
                self._mark_adjacency_if_changed(x, y)

    def _reconcile_missing(self):
        for entity_id in tuple(self.live_entity_ids):
            rec = self.entities.get(entity_id)
            if rec is None:
                self.live_entity_ids.discard(entity_id)
                continue
            if not rec.get("alive", False):
                self.live_entity_ids.discard(entity_id)
                continue
            if rec.get("last_seen_round") == self.current_round:
                continue
            pos = rec.get("position")
            if pos is None:
                self.live_entity_ids.discard(entity_id)
                continue

            x, y = pos
            if self.visible_round[y][x] != self.current_round:
                continue

            if rec["is_building"] and self.tile_building_id[y][x] == rec["id"]:
                self.tile_building_id[y][x] = None
            if (not rec["is_building"]) and self.tile_unit_id[y][x] == rec["id"]:
                self.tile_unit_id[y][x] = None
            rec["position"] = None
            rec["alive"] = False
            self.live_entity_ids.discard(entity_id)
            self._bump("map_reconciled_entities")
            self._mark_adjacency_if_changed(x, y)

    def _reconcile_inferred_buildings(self, visible_tiles):
        for x, y in visible_tiles:
            if (
                self.inferred_tile[y][x] in (MAP_CORE, MAP_ENEMY)
                and self.tile_building_id[y][x] is None
            ):
                if self._is_observed_enemy_core_footprint_tile(x, y):
                    continue
                self.inferred_tile[y][x] = None
                self._mark_adjacency_if_changed(x, y)

    def _entity_tile_code(self, entity_type: EntityType, team) -> int:
        if entity_type == EntityType.BUILDER_BOT:
            if team == self.my_team:
                return MAP_FRIENDLY_BOT
            return MAP_ENEMY_UNIT

        if team != self.my_team:
            if entity_type in TURRET_ENTITY_TYPES:
                return MAP_ENEMY_TURRET
            return MAP_ENEMY

        return MAP_TILE_BY_ENTITY.get(entity_type, MAP_OBSTACLE)

    def _env_to_map_code(self, env: Environment) -> int:
        if env == Environment.EMPTY:
            return MAP_FREE
        if env == Environment.WALL:
            return MAP_OBSTACLE
        if env == Environment.ORE_TITANIUM:
            return MAP_ORE_TITANIUM
        if env == Environment.ORE_AXIONITE:
            return MAP_ORE_AXIONITE
        return MAP_UNKNOWN

    def _infer_static_mirror(self, x: int, y: int, code: int):
        mx, my = self._mirror_xy(x, y)
        if not self.in_bounds(mx, my) or self.env_observed[my][mx]:
            return
        if self._is_observed_enemy_core_footprint_tile(mx, my):
            return
        # Do not override inferred core regions with mirrored free tiles.
        if code == MAP_FREE and self.inferred_tile[my][mx] in (MAP_CORE, MAP_ENEMY):
            return
        self._set_inferred_tile(mx, my, code)

    def _is_static_symmetry_code(self, code: int) -> bool:
        return code in self._STATIC_SYMMETRY_CODES

    def _is_observed_enemy_core_footprint_tile(self, x: int, y: int) -> bool:
        center = self.enemy_core_observed_center
        if center is None and self.inferred_enemy_core_center is not None:
            center = (
                self.inferred_enemy_core_center.x,
                self.inferred_enemy_core_center.y,
            )
        if center is None:
            return False
        ex, ey = center
        return abs(x - ex) <= 1 and abs(y - ey) <= 1

    def _set_inferred_tile(self, x: int, y: int, code: int, allow_observed: bool = False):
        if not self.in_bounds(x, y):
            return
        if (not allow_observed) and self.env_observed[y][x]:
            return
        if self.tile_building_id[y][x] is not None:
            return
        if self.tile_unit_id[y][x] is not None:
            return

        if self.inferred_tile[y][x] == code:
            return

        self.inferred_tile[y][x] = code
        self._set_ore_state(x, y, code)
        self._mark_adjacency_if_changed(x, y)

    def _clear_inferred_tiles(self):
        for y in range(self.height):
            for x in range(self.width):
                old = self.inferred_tile[y][x]
                if old is None:
                    continue
                self.inferred_tile[y][x] = None
                if old in (MAP_ORE_TITANIUM, MAP_ORE_AXIONITE):
                    self._set_ore_state(x, y, MAP_FREE)
                self._mark_adjacency_if_changed(x, y)

    def _reproject_inferred_tiles(self):
        self._clear_inferred_tiles()

        if self.friendly_core_pos is not None:
            cx = self.friendly_core_pos.x
            cy = self.friendly_core_pos.y
            for dy in range(-1, 2):
                for dx in range(-1, 2):
                    self._set_inferred_tile(cx + dx, cy + dy, MAP_CORE)

        if self.enemy_core_observed_center is not None:
            ex, ey = self.enemy_core_observed_center
            self.inferred_enemy_core_center = Position(ex, ey)
            self._stamp_enemy_core_footprint(ex, ey)
        elif self.infer_symmetry and self.friendly_core_pos is not None:
            ex, ey = self._mirror_xy(
                self.friendly_core_pos.x,
                self.friendly_core_pos.y,
            )
            # Guard against impossible overlap if bootstrap symmetry would mirror
            # enemy core onto our own core footprint.
            if (
                abs(ex - self.friendly_core_pos.x) <= 1
                and abs(ey - self.friendly_core_pos.y) <= 1
            ):
                self.inferred_enemy_core_center = None
            else:
                self.inferred_enemy_core_center = Position(ex, ey)
                for dy in range(-1, 2):
                    for dx in range(-1, 2):
                        self._set_inferred_tile(ex + dx, ey + dy, MAP_ENEMY)
        else:
            self.inferred_enemy_core_center = None

        if not self.infer_symmetry:
            return

        for y in range(self.height):
            for x in range(self.width):
                if not self.env_observed[y][x]:
                    continue
                code = self.env_grid[y][x]
                if not self._is_static_symmetry_code(code):
                    continue
                mx, my = self._mirror_xy(x, y)
                if not self.in_bounds(mx, my):
                    continue
                if self._is_observed_enemy_core_footprint_tile(mx, my):
                    continue
                if code == MAP_FREE and self.inferred_tile[my][mx] in (MAP_CORE, MAP_ENEMY):
                    continue
                self._set_inferred_tile(mx, my, code)

    def _is_static_conflict_under_symmetry(
        self,
        x: int,
        y: int,
        code: int,
        symmetry: str,
    ) -> bool:
        mx, my = self._mirror_xy_for(x, y, symmetry)
        if not self.in_bounds(mx, my):
            return False
        if not self.env_observed[my][mx]:
            return False
        mirror_code = self.env_grid[my][mx]
        if not self._is_static_symmetry_code(mirror_code):
            return False
        return mirror_code != code

    def _count_static_conflicts_for_symmetry(self, symmetry: str) -> int:
        conflicts = 0
        for y in range(self.height):
            for x in range(self.width):
                if not self.env_observed[y][x]:
                    continue
                code = self.env_grid[y][x]
                if not self._is_static_symmetry_code(code):
                    continue
                if self._is_static_conflict_under_symmetry(x, y, code, symmetry):
                    conflicts += 1
        return conflicts

    def _core_mismatch_for_symmetry(
        self,
        symmetry: str,
        enemy_core_center: tuple[int, int] | None = None,
    ) -> int:
        if self.friendly_core_pos is None:
            return 0

        center = enemy_core_center
        if center is None:
            center = self.enemy_core_observed_center
        if center is None:
            return 0

        ex, ey = center
        mx, my = self._mirror_xy_for(
            self.friendly_core_pos.x,
            self.friendly_core_pos.y,
            symmetry,
        )
        return 0 if (mx, my) == (ex, ey) else 1

    def _symmetry_from_enemy_core(
        self,
        enemy_core_center: tuple[int, int] | None = None,
    ) -> str | None:
        if self.friendly_core_pos is None:
            return None

        center = enemy_core_center
        if center is None:
            center = self.enemy_core_observed_center
        if center is None:
            return None

        ex, ey = center
        matches = []
        for symmetry in self._SYMMETRY_CHOICES:
            mx, my = self._mirror_xy_for(
                self.friendly_core_pos.x,
                self.friendly_core_pos.y,
                symmetry,
            )
            if (mx, my) == (ex, ey):
                matches.append(symmetry)

        if not matches:
            return None
        if len(matches) == 1:
            return matches[0]

        # Degenerate fallback (should be rare): keep deterministic order.
        if self.symmetry in matches:
            return self.symmetry
        if self.initial_symmetry in matches:
            return self.initial_symmetry
        for symmetry in self._SYMMETRY_CHOICES:
            if symmetry in matches:
                return symmetry
        return matches[0]

    def _best_alternative_symmetry(
        self,
        pivot_static: tuple[int, int, int] | None = None,
        enemy_core_center: tuple[int, int] | None = None,
    ) -> str:
        candidates = [s for s in self._SYMMETRY_CHOICES if s != self.symmetry]
        order = {name: idx for idx, name in enumerate(self._SYMMETRY_CHOICES)}

        def rank(symmetry: str):
            pivot_conflict = 0
            if pivot_static is not None:
                px, py, pcode = pivot_static
                if self._is_static_conflict_under_symmetry(px, py, pcode, symmetry):
                    pivot_conflict = 1

            core_mismatch = self._core_mismatch_for_symmetry(
                symmetry,
                enemy_core_center,
            )
            total_conflicts = self._count_static_conflicts_for_symmetry(
                symmetry)
            tie_initial = 0 if symmetry == self.initial_symmetry else 1
            return (
                pivot_conflict,
                core_mismatch,
                total_conflicts,
                tie_initial,
                order[symmetry],
            )

        return min(candidates, key=rank)

    def _maybe_revise_symmetry(
        self,
        pivot_static: tuple[int, int, int] | None = None,
        enemy_core_center: tuple[int, int] | None = None,
    ) -> bool:
        if not self.infer_symmetry:
            return False

        forced_by_core = self._symmetry_from_enemy_core(enemy_core_center)
        if forced_by_core is not None:
            if forced_by_core == self.symmetry:
                return False
            self.symmetry = forced_by_core
            self.symmetry_revision += 1
            self._bump("map_symmetry_switches")
            self._reproject_inferred_tiles()
            return True

        if self.symmetry_revision >= 2:
            return False

        current_conflict = False
        if pivot_static is not None:
            px, py, pcode = pivot_static
            current_conflict = self._is_static_conflict_under_symmetry(
                px,
                py,
                pcode,
                self.symmetry,
            )

        if self._core_mismatch_for_symmetry(self.symmetry, enemy_core_center):
            current_conflict = True

        if not current_conflict:
            return False

        best = self._best_alternative_symmetry(
            pivot_static=pivot_static,
            enemy_core_center=enemy_core_center,
        )
        if best == self.symmetry:
            return False

        self.symmetry = best
        self.symmetry_revision += 1
        self._bump("map_symmetry_switches")
        self._reproject_inferred_tiles()
        return True

    def _on_enemy_core_observed(self, x: int, y: int):
        seen_center = (x, y)
        changed_center = self.enemy_core_observed_center != seen_center
        self.enemy_core_observed_center = seen_center

        if changed_center and self._core_mismatch_for_symmetry(self.symmetry, seen_center):
            self._bump("map_symmetry_conflicts")

        switched = self._maybe_revise_symmetry(enemy_core_center=seen_center)
        if changed_center and not switched:
            self._reproject_inferred_tiles()

        self.inferred_enemy_core_center = Position(x, y)
        self._stamp_enemy_core_footprint(x, y)

    def _stamp_enemy_core_footprint(self, cx: int, cy: int):
        for dy in range(-1, 2):
            for dx in range(-1, 2):
                x = cx + dx
                y = cy + dy
                if not self.in_bounds(x, y):
                    continue
                if self.friendly_core_pos is not None:
                    if (
                        abs(x - self.friendly_core_pos.x) <= 1
                        and abs(y - self.friendly_core_pos.y) <= 1
                    ):
                        continue
                # Stamp directly so footprint tiles remain inferred even when
                # a building id is currently present on the tile.
                if self.inferred_tile[y][x] == MAP_ENEMY:
                    continue
                self.inferred_tile[y][x] = MAP_ENEMY
                self._set_ore_state(x, y, MAP_ENEMY)
                self._mark_adjacency_if_changed(x, y)

    def _set_ore_state(self, x: int, y: int, code: int):
        p = (x, y)
        if code == MAP_ORE_TITANIUM:
            self.titanium_unharvested.add(p)
            self.titanium_harvested.discard(p)
            self.axionite_unharvested.discard(p)
            self.axionite_harvested.discard(p)
            return
        if code == MAP_ORE_AXIONITE:
            self.axionite_unharvested.add(p)
            self.axionite_harvested.discard(p)
            self.titanium_unharvested.discard(p)
            self.titanium_harvested.discard(p)
            return

        if (
            p not in self.titanium_unharvested
            and p not in self.titanium_harvested
            and p not in self.axionite_unharvested
            and p not in self.axionite_harvested
            and p not in self.titanium_harvesters
            and p not in self.axionite_harvesters
        ):
            return

        self.titanium_unharvested.discard(p)
        self.titanium_harvested.discard(p)
        self.axionite_unharvested.discard(p)
        self.axionite_harvested.discard(p)
        self.titanium_harvesters.discard(p)
        self.axionite_harvesters.discard(p)

    def _mark_harvested_by_harvester(self, x: int, y: int):
        p = (x, y)
        if self.env_grid[y][x] == MAP_ORE_TITANIUM:
            self.titanium_unharvested.discard(p)
            self.titanium_harvested.add(p)
            self.titanium_harvesters.add(p)
        elif self.env_grid[y][x] == MAP_ORE_AXIONITE:
            self.axionite_unharvested.discard(p)
            self.axionite_harvested.add(p)
            self.axionite_harvesters.add(p)


class VisionLocalMap:
    """Lightweight fixed-size map window centered on an immobile unit."""

    def __init__(
        self,
        map_width: int,
        map_height: int,
        my_team,
        center_pos: Position,
        vision_radius_sq: int,
    ):
        self.global_width = map_width
        self.global_height = map_height
        self.my_team = my_team

        self.vision_radius_sq = vision_radius_sq
        self.vision_radius = self._axis_radius_from_sq(vision_radius_sq)
        self.width = (self.vision_radius * 2) + 1
        self.height = (self.vision_radius * 2) + 1

        self.view_x0 = center_pos.x - self.vision_radius
        self.view_y0 = center_pos.y - self.vision_radius
        self.coordinate_origin = (self.view_x0, self.view_y0)

        self.env_grid = [
            [MAP_UNKNOWN for _ in range(self.width)] for _ in range(self.height)]
        self.tile_building_id = [
            [None for _ in range(self.width)] for _ in range(self.height)]
        self.tile_unit_id = [
            [None for _ in range(self.width)] for _ in range(self.height)]
        self.visible_round = [
            [-1 for _ in range(self.width)] for _ in range(self.height)]

        self.entities = {}
        self.friendly_core_pos = None
        self.unit_halo_blocked_tiles = set()

        self.titanium_unharvested = set()
        self.titanium_harvested = set()
        self.axionite_unharvested = set()
        self.axionite_harvested = set()
        self.titanium_harvesters = set()
        self.axionite_harvesters = set()

        self.current_round = 0
        self._round_metrics = {}
        self.last_metrics = {}

    @staticmethod
    def _axis_radius_from_sq(value: int) -> int:
        if value <= 0:
            return 0
        return int(math.isqrt(value))

    def _reset_round_metrics(self):
        self._round_metrics = {
            "map_nearby_tiles_calls": 0,
            "map_nearby_entities_calls": 0,
            "map_tile_env_calls": 0,
            "map_entity_pos_calls": 0,
            "map_entity_type_calls": 0,
            "map_entity_team_calls": 0,
            "map_direction_calls": 0,
            "map_bridge_target_calls": 0,
            "map_known_building_hits": 0,
            "map_reconciled_entities": 0,
            "map_adj_dirty_tiles": 0,
            "map_adj_rebuilt_tiles": 0,
        }

    def _bump(self, key: str, amount: int = 1):
        self._round_metrics[key] = self._round_metrics.get(key, 0) + amount

    def consume_round_metrics(self):
        return dict(self.last_metrics)

    def _to_local(self, x: int, y: int):
        return x - self.view_x0, y - self.view_y0

    def _in_window(self, x: int, y: int) -> bool:
        lx, ly = self._to_local(x, y)
        return 0 <= lx < self.width and 0 <= ly < self.height

    def _in_map_bounds(self, x: int, y: int) -> bool:
        return 0 <= x < self.global_width and 0 <= y < self.global_height

    def in_bounds(self, x: int, y: int) -> bool:
        if not self._in_map_bounds(x, y):
            return False
        return self._in_window(x, y)

    def is_visible(self, x: int, y: int, max_age_rounds: int = 0) -> bool:
        if not self.in_bounds(x, y):
            return False
        lx, ly = self._to_local(x, y)
        return self.visible_round[ly][lx] >= (self.current_round - max_age_rounds)

    def set_friendly_core(self, core_pos: Position):
        self.friendly_core_pos = core_pos

    def get_known_building(self, x: int, y: int):
        if not self.in_bounds(x, y):
            return None
        lx, ly = self._to_local(x, y)
        bid = self.tile_building_id[ly][lx]
        if bid is None:
            return None
        rec = self.entities.get(bid)
        if rec is None or not rec.get("alive", False):
            return None
        return rec

    def get_known_unit(self, x: int, y: int):
        if not self.in_bounds(x, y):
            return None
        lx, ly = self._to_local(x, y)
        uid = self.tile_unit_id[ly][lx]
        if uid is None:
            return None
        rec = self.entities.get(uid)
        if rec is None or not rec.get("alive", False):
            return None
        return rec

    def is_symmetry_fully_resolved(self) -> bool:
        return False

    def is_tile_known(self, x: int, y: int) -> bool:
        if not self.in_bounds(x, y):
            return False
        lx, ly = self._to_local(x, y)
        return self.visible_round[ly][lx] >= 0

    def get_known_unharvested_titanium(self):
        out = []
        for x, y in self.titanium_unharvested:
            if self.is_tile_known(x, y):
                out.append((x, y))
        return out

    def get_known_unharvested_axionite(self):
        out = []
        for x, y in self.axionite_unharvested:
            if self.is_tile_known(x, y):
                out.append((x, y))
        return out

    def _clear_round_occupancy(self):
        for y in range(self.height):
            unit_row = self.tile_unit_id[y]
            building_row = self.tile_building_id[y]
            for x in range(self.width):
                unit_row[x] = None
                building_row[x] = None

    def update_from_controller(self, c: Controller):
        self.current_round = c.get_current_round()
        self._reset_round_metrics()
        self._clear_round_occupancy()

        try:
            my_id = c.get_id()
        except GameError:
            my_id = None

        try:
            nearby_tiles = list(c.get_nearby_tiles())
            self._bump("map_nearby_tiles_calls")
        except GameError:
            nearby_tiles = []

        for t in nearby_tiles:
            x, y = t.x, t.y
            if not self.in_bounds(x, y):
                continue
            lx, ly = self._to_local(x, y)
            self.visible_round[ly][lx] = self.current_round
            try:
                env = c.get_tile_env(t)
                self._bump("map_tile_env_calls")
            except GameError:
                continue
            code = self._env_to_map_code(env)
            self.env_grid[ly][lx] = code
            self._set_ore_state(x, y, code)

        try:
            nearby_entities = list(c.get_nearby_entities())
            self._bump("map_nearby_entities_calls")
        except GameError:
            nearby_entities = []

        seen_entity_ids = set()
        for entity_id in nearby_entities:
            if my_id is not None and entity_id == my_id:
                continue

            try:
                pos = c.get_position(entity_id)
                self._bump("map_entity_pos_calls")
            except GameError:
                continue

            if not self.in_bounds(pos.x, pos.y):
                continue

            try:
                entity_type = c.get_entity_type(entity_id)
                self._bump("map_entity_type_calls")
                team = c.get_team(entity_id)
                self._bump("map_entity_team_calls")
            except GameError:
                continue

            rec = {
                "id": entity_id,
                "entity_type": entity_type,
                "team": team,
                "is_building": entity_type in BUILDING_ENTITY_TYPES,
                "tile_code": self._entity_tile_code(entity_type, team),
                "position": (pos.x, pos.y),
                "last_seen_round": self.current_round,
                "alive": True,
                "direction": None,
            }
            if entity_type in DIRECTIONAL_ENTITY_TYPES:
                try:
                    rec["direction"] = c.get_direction(entity_id)
                    self._bump("map_direction_calls")
                except GameError:
                    pass

            self.entities[entity_id] = rec
            seen_entity_ids.add(entity_id)

            lx, ly = self._to_local(pos.x, pos.y)
            if rec["is_building"]:
                self.tile_building_id[ly][lx] = entity_id
                if entity_type == EntityType.HARVESTER:
                    self._mark_harvested_by_harvester(pos.x, pos.y)
            else:
                self.tile_unit_id[ly][lx] = entity_id

        for entity_id, rec in self.entities.items():
            if entity_id in seen_entity_ids:
                rec["alive"] = True
            elif rec.get("last_seen_round") != self.current_round:
                rec["alive"] = False

        halo = set()
        for rec in self.entities.values():
            if not isinstance(rec, dict):
                continue
            if not rec.get("alive", False):
                continue
            if rec.get("is_building", False):
                continue
            pos = rec.get("position")
            if not (isinstance(pos, tuple) and len(pos) == 2):
                continue

            ux = int(pos[0])
            uy = int(pos[1])
            for dx in (-1, 0, 1):
                for dy in (-1, 0, 1):
                    tx = ux + dx
                    ty = uy + dy
                    if self.in_bounds(tx, ty):
                        halo.add((tx, ty))
        self.unit_halo_blocked_tiles = halo

        self.last_metrics = dict(self._round_metrics)

    def _entity_tile_code(self, entity_type: EntityType, team) -> int:
        if entity_type == EntityType.BUILDER_BOT:
            if team == self.my_team:
                return MAP_FRIENDLY_BOT
            return MAP_ENEMY_UNIT

        if team != self.my_team:
            if entity_type in TURRET_ENTITY_TYPES:
                return MAP_ENEMY_TURRET
            return MAP_ENEMY

        return MAP_TILE_BY_ENTITY.get(entity_type, MAP_OBSTACLE)

    def _env_to_map_code(self, env: Environment) -> int:
        if env == Environment.EMPTY:
            return MAP_FREE
        if env == Environment.WALL:
            return MAP_OBSTACLE
        if env == Environment.ORE_TITANIUM:
            return MAP_ORE_TITANIUM
        if env == Environment.ORE_AXIONITE:
            return MAP_ORE_AXIONITE
        return MAP_UNKNOWN

    def _set_ore_state(self, x: int, y: int, code: int):
        p = (x, y)
        if code == MAP_ORE_TITANIUM:
            self.titanium_unharvested.add(p)
            self.titanium_harvested.discard(p)
            self.axionite_unharvested.discard(p)
            self.axionite_harvested.discard(p)
            return
        if code == MAP_ORE_AXIONITE:
            self.axionite_unharvested.add(p)
            self.axionite_harvested.discard(p)
            self.titanium_unharvested.discard(p)
            self.titanium_harvested.discard(p)
            return

        self.titanium_unharvested.discard(p)
        self.titanium_harvested.discard(p)
        self.axionite_unharvested.discard(p)
        self.axionite_harvested.discard(p)
        self.titanium_harvesters.discard(p)
        self.axionite_harvesters.discard(p)

    def _mark_harvested_by_harvester(self, x: int, y: int):
        if not self.in_bounds(x, y):
            return
        lx, ly = self._to_local(x, y)
        p = (x, y)
        if self.env_grid[ly][lx] == MAP_ORE_TITANIUM:
            self.titanium_unharvested.discard(p)
            self.titanium_harvested.add(p)
            self.titanium_harvesters.add(p)
        elif self.env_grid[ly][lx] == MAP_ORE_AXIONITE:
            self.axionite_unharvested.discard(p)
            self.axionite_harvested.add(p)
            self.axionite_harvesters.add(p)

    def get(self, x: int, y: int) -> int:
        if not self.in_bounds(x, y):
            return MAP_OBSTACLE
        lx, ly = self._to_local(x, y)

        uid = self.tile_unit_id[ly][lx]
        if uid is not None:
            rec = self.entities.get(uid)
            if rec is not None:
                return rec["tile_code"]
            return MAP_OTHER_BOT

        bid = self.tile_building_id[ly][lx]
        if bid is not None:
            rec = self.entities.get(bid)
            if rec is not None:
                return rec["tile_code"]
            return MAP_OBSTACLE

        return self.env_grid[ly][lx]

    def ore_marker(self, x: int, y: int):
        p = (x, y)
        if p in self.titanium_harvesters:
            return "T"
        if p in self.axionite_harvesters:
            return "A"
        if p in self.titanium_unharvested or p in self.titanium_harvested:
            return "t"
        if p in self.axionite_unharvested or p in self.axionite_harvested:
            return "a"
        return None

    def conveyor_marker(self, x: int, y: int):
        if not self.in_bounds(x, y):
            return None
        lx, ly = self._to_local(x, y)
        bid = self.tile_building_id[ly][lx]
        if bid is None:
            return None
        rec = self.entities.get(bid)
        if rec is None:
            return None
        direction = rec.get("direction")
        if direction is None:
            return None
        if direction.name == "NORTH":
            return "^"
        if direction.name == "EAST":
            return ">"
        if direction.name == "SOUTH":
            return "v"
        if direction.name == "WEST":
            return "<"
        return "*"
