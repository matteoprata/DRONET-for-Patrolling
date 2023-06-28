
import numpy as np
from src.patrolling.meta_patrolling import PatrollingPolicy


class MaxAOIPolicy(PatrollingPolicy):
    name = "Go-Max-AOI"
    identifier = 1
    line_tick = "-"
    marker = ">"

    def __init__(self, patrol_drone, set_drones, set_targets):
        super().__init__(patrol_drone=patrol_drone, set_drones=set_drones, set_targets=set_targets)

    def next_visit(self):
        """ Returns the target with the oldest age. """
        chosen_target = None
        biggest_ratio = -np.inf
        for t in self.set_targets:
            temp = t.AOI_absolute()
            # set the target visited furthest in the past
            if t.lock is None and temp > biggest_ratio:
                biggest_ratio = temp
                chosen_target = t
        return chosen_target
