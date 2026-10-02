import v47_impl
import xuan79_impl


SWITCH_ROUND = 600


class Player:
    def __init__(self):
        self._anti = xuan79_impl.Player()
        self._econ = v47_impl.Player()

    def run(self, c):
        if c.get_current_round() < SWITCH_ROUND:
            self._anti.run(c)
        else:
            self._econ.run(c)
