from cambc import Position


def detect_symmetry(core_pos: Position, map_width: int, map_height: int) -> str:
    """Return an initial symmetry guess from core position.

    This is only a bootstrap guess; UnitLocalMap can revise symmetry later.
    Avoid guesses that would mirror the friendly core onto itself, because that
    creates impossible enemy-core overlap and corrupts early inferred tiles.
    """
    mid_x = (map_width - 1) / 2.0
    mid_y = (map_height - 1) / 2.0

    on_mid_x = abs(core_pos.x - mid_x) < 0.51
    on_mid_y = abs(core_pos.y - mid_y) < 0.51

    # If the core sits on a map mid-line, one axial symmetry maps the core to
    # itself and is therefore impossible for the enemy core location.
    # Choose the remaining axis-family guess that still places enemy opposite.
    if on_mid_x and not on_mid_y:
        return "HORIZONTAL"
    if on_mid_y and not on_mid_x:
        return "VERTICAL"

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
