
# from src.utilities.utilities import euclidean_distance
import numpy as np


class PlottingStyle:
    line_tick = "-"
    marker = "o"


class PatrollingPolicy(PlottingStyle):
    name = None
    identifier = None

    def __init__(self, patrol_drone, set_drones, set_targets):
        self.patrol_drone = patrol_drone
        self.set_drones = set_drones
        self.set_targets = set_targets

    def next_visit(self):
        pass


class RLPolicy(PatrollingPolicy):
    """ Reinforcement Learning trained policy. """
    name = "Go-RL"
    identifier = 4
    line_tick = "-"
    marker = "o"

    def __init__(self, patrol_drone, set_drones, set_targets):
        super().__init__(patrol_drone=patrol_drone, set_drones=set_drones, set_targets=set_targets)

    def next_visit(self):
        """ Returns a random target. """
        pass


class RandomPolicy(PatrollingPolicy):
    name = "Go-Random"
    identifier = 0
    line_tick = "-"
    marker = "p"

    def __init__(self, patrol_drone, set_drones, set_targets):
        super().__init__(patrol_drone=patrol_drone, set_drones=set_drones, set_targets=set_targets)

    def next_visit(self):
        """ Returns a random target. """
        target_id = self.patrol_drone.simulator.rnd_explore.randint(0, len(self.patrol_drone.simulator.environment.targets))
        target = self.patrol_drone.simulator.environment.targets[target_id]
        return target


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


class MaxSumResidualPolicy(PatrollingPolicy):
    name = "Go-Max-Residual-Ratio"
    identifier = 3
    line_tick = "-"
    marker = "+"

    def __init__(self, patrol_drone, set_drones, set_targets):
        super().__init__(patrol_drone=patrol_drone, set_drones=set_drones, set_targets=set_targets)

    def next_visit(self):
        """ Returns the target leading to the maximum minimum residual upon having reached it. """

        max_min_res_list = [np.inf] * len(self.set_targets)
        for ti, target_1 in enumerate(self.set_targets):
            if self.patrol_drone.current_target() == target_1 or target_1.lock is not None:
                continue

            rel_time_arrival = euclidean_distance(target_1.coords, self.patrol_drone.current_target().coords) / self.patrol_drone.speed
            sec_arrival = self.patrol_drone.simulator.cur_step * self.patrol_drone.simulator.ts_duration_sec + rel_time_arrival

            min_res_list = []
            for target_2 in self.set_targets:
                ls_visit = target_2.last_visit_ts * self.patrol_drone.simulator.ts_duration_sec if target_1.identifier != target_2.identifier else sec_arrival
                RES = (sec_arrival - ls_visit) / target_2.maximum_tolerated_idleness
                min_res_list.append(RES)

            # print("min ->", min_res_list, target_1.identifier)
            max_min_res_list[ti] = np.sum(min_res_list)
        max_min_res_tar = self.set_targets[np.argmin(max_min_res_list)]

        # print("max_min ->", max_min_res_list, max_min_res_tar.identifier)
        return max_min_res_tar


def euclidean_distance(p1, p2):
    """ Given points p1, p2 in R^2 it returns the norm of the vector connecting them.  """
    return np.linalg.norm(np.array(p1)-np.array(p2))
