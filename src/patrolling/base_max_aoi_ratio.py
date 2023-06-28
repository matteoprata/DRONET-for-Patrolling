
import numpy as np
from src.patrolling.meta_patrolling import PatrollingPolicy


class MaxAOIRatioPolicy(PatrollingPolicy):
    name = "Go-Max-AOI-Ratio"
    identifier = 2
    line_tick = "-"
    marker = "<"

    def __init__(self, patrol_drone, set_drones, set_targets):
        super().__init__(patrol_drone=patrol_drone, set_drones=set_drones, set_targets=set_targets)

    def next_visit(self):
        """ Returns the target with the lowest percentage residual. """
        chosen_target = None
        least_ratio = np.inf
        for t in self.set_targets:
            temp = t.AOI_tolerance_ratio()
            # set the target visited furthest in the past and has lest tolerance
            if t.lock is None and temp < least_ratio:
                least_ratio = temp
                chosen_target = t
        return chosen_target
