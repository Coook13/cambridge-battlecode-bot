"""Live-safe seat/map selector against xuanming_v3.

This bot intentionally composes already-tested local winners:
- v53_xuan_plus wins all tested seat-B cells against xuanming_v3.
- v62_fastwave wins seat-A default_small1/default_large1/test1.
- v47_earlier_fire wins seat-A default_medium1 and carries the #1-map
  builder-fire improvement.
- v79_large_only is the strongest broad/no1-map fallback.

The selector chooses once on the first turn using only map geometry and our
core position, then delegates permanently. Unknown live geometry goes to v79
instead of the older fragile fallbacks.
"""

from cambc import EntityType, Environment, GameError

import v47_impl
import xuan58_impl
import xuan53_impl
import xuan62_impl
import xuan78_impl
import xuan79_impl
import norush_impl


class Player:
    def __init__(self):
        self._delegate = None
        self._delegate_name = None

    def run(self, c):
        if self._delegate is None:
            self._choose_delegate(c)
        self._delegate.run(c)

    def _choose_delegate(self, c):
        try:
            width = int(c.get_map_width())
            height = int(c.get_map_height())
            core_xy = self._friendly_core_xy(c)
        except GameError:
            self._delegate_name = "xuan79_fallback"
            self._delegate = xuan79_impl.Player()
            return

        if core_xy is None:
            self._delegate_name = "xuan79_no_core"
            self._delegate = xuan79_impl.Player()
            return

        key = (width, height, core_xy[0], core_xy[1])

        # Extracted top-team replay maps. These are routed before the older
        # local default selector because the deployed v3 file has different
        # seat traps on this pool.
        if key in ((20, 20, 2, 2),):
            self._delegate_name = "v47_top_b12_game1_seat_a"
            self._delegate = v47_impl.Player()
            return

        if key in ((35, 35, 14, 17), (35, 35, 20, 17)):
            self._delegate_name = "v47_top_b12_game2"
            self._delegate = v47_impl.Player()
            return

        if key in ((49, 49, 27, 32), (49, 49, 21, 16)):
            self._delegate_name = "norush_top_b12_game3"
            self._delegate = norush_impl.Player()
            return

        if key in ((21, 29, 4, 4), (21, 29, 16, 4)):
            self._delegate_name = "v47_top_b12_game4"
            self._delegate = v47_impl.Player()
            return

        if key in ((40, 20, 1, 6), (40, 20, 38, 6)):
            self._delegate_name = "xuan79_top_b12_game5"
            self._delegate = xuan79_impl.Player()
            return

        # Exact cells where local head-to-heads found a stronger specialist.
        # Coordinates alone are ambiguous: default_small2 Team B and
        # default_medium1 Team A both sit at (10, 19).
        if key == (21, 21, 10, 1):
            self._delegate_name = "xuan58_small2_seat_a"
            self._delegate = xuan58_impl.Player()
            return

        if key == (21, 21, 10, 19):
            self._delegate_name = "xuan62_small2_seat_b"
            self._delegate = xuan62_impl.Player()
            return

        if key == (30, 30, 3, 3):
            self._delegate_name = "xuan62_medium2_seat_a"
            self._delegate = xuan62_impl.Player()
            return

        if key == (30, 30, 26, 3):
            self._delegate_name = "v47_medium2_seat_b"
            self._delegate = v47_impl.Player()
            return

        if key == (50, 30, 46, 16):
            self._delegate_name = "xuan62_large2_seat_b"
            self._delegate = xuan62_impl.Player()
            return

        if key == (50, 30, 3, 16):
            self._delegate_name = "xuan78_large2_seat_a"
            self._delegate = xuan78_impl.Player()
            return

        if key == (50, 41, 3, 39):
            self._delegate_name = "xuan79_no1_replay_seat_a"
            self._delegate = xuan79_impl.Player()
            return

        if key == (50, 41, 3, 1):
            self._delegate_name = "xuan79_no1_replay_seat_b"
            self._delegate = xuan79_impl.Player()
            return

        if key == (30, 30, 10, 19):
            self._delegate_name = "v47_medium1_seat_a"
            self._delegate = v47_impl.Player()
            return

        if core_xy in ((1, 1), (11, 25), (4, 7)):
            self._delegate_name = "xuan62_seat_a"
            self._delegate = xuan62_impl.Player()
            return

        seat_a_cores = {
            (1, 1),
            (10, 19),
            (11, 25),
            (4, 7),
            (10, 1),
            (3, 3),
            (3, 16),
            (3, 39),
        }
        area = width * height
        ti_ores, ax_ores = self._nearby_ore_counts(c)

        if area >= 1800:
            self._delegate_name = "xuan79_adaptive_large"
            self._delegate = xuan79_impl.Player()
            return

        if area >= 1100 and ax_ores >= ti_ores:
            self._delegate_name = "v47_adaptive_ax_economy"
            self._delegate = v47_impl.Player()
            return

        if core_xy not in seat_a_cores:
            self._delegate_name = "xuan53_adaptive_seat_b"
            self._delegate = xuan53_impl.Player()
            return

        self._delegate_name = "xuan79_adaptive_seat_a"
        self._delegate = xuan79_impl.Player()

    def _friendly_core_xy(self, c):
        my_team = c.get_team()
        try:
            if c.get_entity_type() == EntityType.CORE:
                p = c.get_position()
                return (p.x, p.y)
        except GameError:
            pass

        try:
            buildings = c.get_nearby_buildings()
        except GameError:
            buildings = ()

        for bid in buildings:
            try:
                if c.get_team(bid) != my_team:
                    continue
                if c.get_entity_type(bid) != EntityType.CORE:
                    continue
                p = c.get_position(bid)
                return (p.x, p.y)
            except GameError:
                continue

        return None

    def _nearby_ore_counts(self, c):
        ti = 0
        ax = 0
        try:
            tiles = c.get_nearby_tiles()
        except GameError:
            return (0, 0)
        for tp in tiles:
            try:
                env = c.get_tile_env(tp)
            except GameError:
                continue
            if env == Environment.ORE_TITANIUM:
                ti += 1
            elif env == Environment.ORE_AXIONITE:
                ax += 1
        return (ti, ax)
