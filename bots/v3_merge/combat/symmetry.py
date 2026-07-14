from cambc import Position


def detect_symmetry(core_pos: Position, map_width: int, map_height: int) -> str:
    """Determine symmetry using the core's relation to map mid-lines."""
    mid_x = (map_width - 1) / 2.0
    mid_y = (map_height - 1) / 2.0

    if abs(core_pos.x - mid_x) < 0.51:
        return "VERTICAL"
    if abs(core_pos.y - mid_y) < 0.51:
        return "HORIZONTAL"
    return "ROTATIONAL"


def mirror_position(pos: Position, symmetry: str, map_width: int, map_height: int) -> Position:
    if symmetry == "VERTICAL":
        return Position(map_width - 1 - pos.x, pos.y)
    if symmetry == "HORIZONTAL":
        return Position(pos.x, map_height - 1 - pos.y)

    return Position(map_width - 1 - pos.x, map_height - 1 - pos.y)


def apply_enemy_core_footprint(local_map, enemy_core: Position):
    for dy in range(-1, 2):
        for dx in range(-1, 2):
            nx, ny = enemy_core.x + dx, enemy_core.y + dy
            if 0 <= nx < local_map.width and 0 <= ny < local_map.height:
                local_map.set_fixed_obstacle(nx, ny)