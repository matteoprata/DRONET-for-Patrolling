
from src.utilities.utilities import euclidean_distance
import numpy as np


def max_aoi(set_targets, drone):
    """ Returns the target with the oldest age. """

    # PER MICHELE
    # aoi_list = []
    # for t in set_targets:
    #     if t.lock is None:
    #         aoi_list.append(t.AOI_absolute())
    #     else:
    #         # the target is locked, this target will be the least relevant, discard
    #         aoi_list.append(-np.inf)
    #
    # aoi_list_max = np.argmax(aoi_list)
    # return set_targets[aoi_list_max]

    # more efficient
    # TODO: handle drone battery!
    chosen_target = None
    biggest_ratio = -np.inf
    for t in set_targets:
        temp = t.AOI_absolute()
        # set the target visited furthest in the past
        if t.lock is None and temp > biggest_ratio:
            biggest_ratio = temp
            chosen_target = t
    return chosen_target

    # shorter alternative but no lock considered
    # max_aoi = np.argmax([target.AOI_absolute() for target in set_targets])
    # return set_targets[max_aoi]


def min_residual(set_targets, drone):
    """ Returns the target with the lowest percentage residual. """

    chosen_target = None
    least_ratio = np.inf

    # TODO: handle drone battery!
    for t in set_targets:
        temp = t.AOI_tolerance_ratio()
        # set the target visited furthest in the past and has lest tolerance
        if t.lock is None and temp < least_ratio:
            least_ratio = temp
            chosen_target = t
    return chosen_target

    # shorter alternative but no lock considered
    # min_res = np.argmin([target.AOI_tolerance_ratio() for target in set_targets])
    # return set_targets[min_res]


def min_sum_residual(set_targets, cur_tar, speed, cur_step, ts_duration_sec, drone):
    """ Returns the target leading to the maximum minimum residual upon having reached it. """

    # TODO: handle drone battery!
    max_min_res_list = [np.inf] * len(set_targets)
    for ti, target_1 in enumerate(set_targets):
        if cur_tar == target_1 or target_1.lock is not None:
            continue

        rel_time_arrival = euclidean_distance(target_1.coords, cur_tar.coords) / speed
        sec_arrival = cur_step * ts_duration_sec + rel_time_arrival

        min_res_list = []
        for target_2 in set_targets:
            ls_visit = target_2.last_visit_ts * ts_duration_sec if target_1.identifier != target_2.identifier else sec_arrival
            RES = (sec_arrival - ls_visit) / target_2.maximum_tolerated_idleness
            min_res_list.append(RES)

        # print("min ->", min_res_list, target_1.identifier)
        max_min_res_list[ti] = np.sum(min_res_list)
    max_min_res_tar = set_targets[np.argmin(max_min_res_list)]

    # print("max_min ->", max_min_res_list, max_min_res_tar.identifier)
    return max_min_res_tar


# IMPLEMENT your own policy here
def policy_next_target_michele(set_targets, drone):
    # TODO: handle drone battery;
    #  think of how the distance to reach a target may influence the decision of the drone and the drones energy!
    pass
