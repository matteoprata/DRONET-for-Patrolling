
from enum import Enum
import numpy as np
from src.RL.RLSate import State, FeatureFamilyName
from src.utilities.utilities import min_max_normalizer
from src.constants import RLRewardType


def __reward_preamble(state_prime: State):
    r_vals = np.array(state_prime.get_feature_by_name(FeatureFamilyName.AOIR).values(is_normalized=False))  # - SUM AOIR > 100%  [50%, 150%] -> 150%
    r = - np.sum(r_vals[r_vals >= 1])
    return r


def reward_0(sim, state_prime: State):
    r = - np.sum(state_prime.get_feature_by_name(FeatureFamilyName.AOIR).values())  # - SUM AOIR  [50%, 150%] -> 200%
    r_norm = min_max_normalizer(r, sim.rmin, sim.rmax, -1, 1, soft=True)
    return r_norm


def reward_1(sim, state_prime: State, eps=0.05):
    r_minor = - np.sum(state_prime.get_feature_by_name(FeatureFamilyName.AOIR).values())
    r = __reward_preamble(state_prime) + eps * r_minor
    r_norm = min_max_normalizer(r, sim.rmin, sim.rmax, -1, 1, soft=True)
    return r_norm


def reward_2(sim, state_prev: State, state_prime: State, a_prev: int, weight=0.3):
    pr = __reward_preamble(state_prime)
    if None in [state_prev, state_prime, a_prev]:
        return 0

    st_prev_aoi = np.array(state_prev.get_feature_by_name(FeatureFamilyName.AOIR).values(is_normalized=False))
    st_prev_dis = np.array(state_prev.get_feature_by_name(FeatureFamilyName.TIME_DISTANCES).values(is_normalized=True))

    r = 1 / (st_prev_aoi[a_prev] + st_prev_dis[a_prev]) + weight * pr if st_prev_aoi[a_prev] >= 1 else 0
    r_norm = min_max_normalizer(r, sim.rmin, sim.rmax, -1, 1, soft=True)
    return r_norm


def reward_3(sim, state_prev: State, state_prime: State, a_prev: int, weight=0.3):
    pr = __reward_preamble(state_prime)
    if None in [state_prev, state_prime, a_prev]:
        return 0

    st_prev_aoi = np.array(state_prev.get_feature_by_name(FeatureFamilyName.AOIR).values(is_normalized=False))
    st_prev_dis = np.array(state_prev.get_feature_by_name(FeatureFamilyName.TIME_DISTANCES).values(is_normalized=True))

    r = (- st_prev_aoi[a_prev] - st_prev_dis[a_prev] + weight * pr) if st_prev_aoi[a_prev] >= 1 else 0
    r_norm = min_max_normalizer(r, sim.rmin, sim.rmax, -1, 1, soft=True)
    return r_norm


def reward_map(type, sim, state_prev: State, state_prime: State, a_prev: int):

    if type == RLRewardType.REW0:
        return reward_0(sim, state_prime)

    elif type == RLRewardType.REW1:
        return reward_1(sim, state_prime)

    elif type == RLRewardType.REW2:
        return reward_2(sim, state_prev, state_prime, a_prev)

    elif type == RLRewardType.REW3:
        return reward_3(sim, state_prev, state_prime, a_prev)
