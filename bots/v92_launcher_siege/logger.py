import os
import time

from constants import (COMPETITION_MODE,
                       ENABLE_ADJACENCY_LIST_DUMPS,
                       ENABLE_LOCAL_MAP_DUMPS,
                       MAP_DEBUG_DUMP_EVERY_N, SHOW_EVENT_DEBUG,
                       SHOW_MAP_DEBUG)

_ROOT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
LOG_FILE = os.path.join(_ROOT_DIR, "battlecode_debug.log")
TIMING_FILE = os.path.join(_ROOT_DIR, "battlecode_timing.log")

# Debug controls (code-level flags, no CLI required).
LOGGING_ENABLED = not COMPETITION_MODE
DEBUG_ENABLED = LOGGING_ENABLED and SHOW_EVENT_DEBUG
DEBUG_MAP_DUMPS = (
    LOGGING_ENABLED and SHOW_MAP_DEBUG and ENABLE_LOCAL_MAP_DUMPS
)
DEBUG_ADJACENCY_DUMPS = DEBUG_MAP_DUMPS and ENABLE_ADJACENCY_LIST_DUMPS
DEBUG_MAP_DUMP_EVERY_N = MAP_DEBUG_DUMP_EVERY_N
TIMING_ENABLED = LOGGING_ENABLED

_log_initialized = False
_tick_log_overhead_ns = 0


def reset_tick_log_overhead():
    global _tick_log_overhead_ns
    if not LOGGING_ENABLED:
        return
    _tick_log_overhead_ns = 0


def consume_tick_log_overhead_us() -> int:
    global _tick_log_overhead_ns
    if not LOGGING_ENABLED:
        return 0
    out = _tick_log_overhead_ns // 1000
    _tick_log_overhead_ns = 0
    return out


def _add_log_overhead(start_ns: int):
    global _tick_log_overhead_ns
    if not LOGGING_ENABLED:
        return
    _tick_log_overhead_ns += time.perf_counter_ns() - start_ns


def reset_run_logs():
    if not LOGGING_ENABLED:
        return
    with open(LOG_FILE, "w", encoding="utf-8") as f:
        f.write("=== LOG START ===\n")
    with open(TIMING_FILE, "w", encoding="utf-8") as f:
        f.write("=== TIMING LOG START ===\n")


def init_logger():
    global _log_initialized
    if not LOGGING_ENABLED or _log_initialized:
        return
    _log_initialized = True


def _should_log_map(round_num: int) -> bool:
    if not LOGGING_ENABLED or not DEBUG_MAP_DUMPS:
        return False
    if DEBUG_MAP_DUMP_EVERY_N <= 1:
        return True
    return round_num % DEBUG_MAP_DUMP_EVERY_N == 0


def log_event(round_num: int, ent_id: int, ent_type: str, pos: str, tag: str, force: bool = False, **kwargs):
    if not LOGGING_ENABLED:
        return
    if not force and not DEBUG_ENABLED:
        return
    start_ns = time.perf_counter_ns()
    init_logger()
    line = f"r={round_num} | id={ent_id} | type={ent_type} | pos={pos} | {tag}"
    if kwargs:
        extras = " | ".join(f"{k}={v}" for k, v in kwargs.items())
        line += f" | {extras}"

    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(line + "\n")
    _add_log_overhead(start_ns)


def log_local_map(round_num: int, ent_id: int, ent_type: str, pos: str, local_map, bot_pos):
    if not LOGGING_ENABLED:
        return
    if not _should_log_map(round_num):
        return
    start_ns = time.perf_counter_ns()
    init_logger()
    out = [
        f"r={round_num} | id={ent_id} | type={ent_type} | pos={pos} | local_map_dump"]

    view_x0 = getattr(local_map, "view_x0", 0)
    view_y0 = getattr(local_map, "view_y0", 0)

    header = "    " + "".join(str((view_x0 + i) % 10)
                              for i in range(local_map.width))
    out.append(header)

    map_chars = {
        0: "?",   # unknown
        1: " ",   # free (empty, not walkable)
        2: "#",   # wall
        3: "=",   # road
        4: ">",   # conveyor (overridden by conveyor_marker)
        5: "S",   # splitter
        6: "B",   # bridge
        7: "A",   # armoured conveyor
        8: "C",   # core
        10: "W",  # barrier (wall/defense)
        11: "H",  # harvester
        12: "F",  # foundry
        13: "L",  # launcher
        14: "G",  # gunner
        15: "N",  # sentinel
        16: "R",  # breach
        17: "M",  # marker
        18: "t",  # titanium ore
        19: "a",  # axionite ore
        20: "E",  # enemy building
        21: "b",  # other bot
        22: "@",  # friendly bot
        23: "e",  # enemy bot
        24: "T",  # enemy turret
    }

    for ly in range(local_map.height):
        gy = view_y0 + ly
        row_chars = []
        for lx in range(local_map.width):
            gx = view_x0 + lx
            if gx == bot_pos.x and gy == bot_pos.y:
                # Current unit marker shown distinctly from other friendly bots.
                row_chars.append("P")
                continue

            # Determine entity layers first so topmost object is rendered.
            top_unit = local_map.tile_unit_id[ly][lx] is not None
            top_building = local_map.tile_building_id[ly][lx] is not None

            val = local_map.get(gx, gy)

            if top_unit:
                row_chars.append(map_chars.get(val, "X"))
                continue

            if top_building:
                direction_char = local_map.conveyor_marker(gx, gy)
                if direction_char is not None:
                    row_chars.append(direction_char)
                else:
                    row_chars.append(map_chars.get(val, "X"))
                continue

            ore = local_map.ore_marker(gx, gy)
            if ore is not None:
                row_chars.append(ore)
                continue

            row_chars.append(map_chars.get(val, "X"))

        out.append(f"{gy:03d} " + "".join(row_chars))

    out.append(
        "ore_counts"
        + f" | ti_unharvested={len(local_map.titanium_unharvested)}"
        + f" | ti_harvested={len(local_map.titanium_harvested)}"
        + f" | ax_unharvested={len(local_map.axionite_unharvested)}"
        + f" | ax_harvested={len(local_map.axionite_harvested)}"
    )

    if DEBUG_ADJACENCY_DUMPS:
        get_adj_list = getattr(local_map, "get_adjacency_list", None)
        if callable(get_adj_list):
            adjacency = get_adj_list()
            if isinstance(adjacency, dict):
                out.append(f"adjacency_dump | nodes={len(adjacency)}")
                for (x, y), neighbors in sorted(adjacency.items()):
                    n_txt = ",".join(f"({nx},{ny})" for nx, ny in neighbors)
                    out.append(f"({x},{y}) -> [{n_txt}]")
            else:
                out.append("adjacency_dump | unsupported=1")
        else:
            out.append("adjacency_dump | unsupported=1")

    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write("\n".join(out) + "\n")
    _add_log_overhead(start_ns)


def log_timing_event(round_num: int, ent_id: int, ent_type: str, pos: str, elapsed_us: int, **kwargs):
    if not LOGGING_ENABLED or not TIMING_ENABLED:
        return
    line = (
        f"r={round_num} | id={ent_id} | type={ent_type} | pos={pos} | "
        f"tick_time_us={elapsed_us}"
    )
    if kwargs:
        extras = " | ".join(f"{k}={v}" for k, v in kwargs.items())
        line += f" | {extras}"
    with open(TIMING_FILE, "a", encoding="utf-8") as f:
        f.write(line + "\n")
