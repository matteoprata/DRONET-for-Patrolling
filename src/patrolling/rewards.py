
from src.utilities import config
from src.utilities.utilities import min_max_normalizer

def reward_0_TEMPORAL(s, a, drone, simulator, TARGET_VIOLATION_FACTOR, N_ACTIONS):
    # # REWARD TEST ATP01
    # rew = - max([min(i, self.TARGET_VIOLATION_FACTOR) for i in s_prime.aoi_idleness_ratio(False)])
    # rew += self.simulator.penalty_on_bs_expiration if s_prime.is_final else 0
    # rew = min_max_normalizer(rew,
    #                          startUB=0,
    #                          startLB=(-(self.TARGET_VIOLATION_FACTOR - self.simulator.penalty_on_bs_expiration)),
    #                          endUB=0,
    #                          endLB=-1)

    rew = 0

    time_dist_to_a = s.time_distances(False)[a]
    n_steps = max(int(time_dist_to_a / config.DELTA_DEC), 1)

    for step in range(n_steps):
        # count from the head of the arrow
        TIME = simulator.current_second() - (config.DELTA_DEC * step)

        for target in simulator.environment.targets:
            LAST_VISIT = target.last_visit_ts[drone.identifier] * simulator.ts_duration_sec
            residual = (TIME - LAST_VISIT) / target.maximum_tolerated_idleness
            residual = min(residual, TARGET_VIOLATION_FACTOR)

            rew += - residual if residual >= 1 else 0

    # rew += self.simulator.penalty_on_bs_expiration if s_prime.is_final else 0
    # rew = -100 if s.vector()[a] == 0 else rew

    # NOW NORMALIZE
    n_steps = int(simulator.max_travel_time() / config.DELTA_DEC)
    rew = min_max_normalizer(rew,
                             startLB=-TARGET_VIOLATION_FACTOR * n_steps * N_ACTIONS,  # TODO FIX ACTIONS /2
                             startUB=0,
                             endLB=-1,
                             endUB=0)

    return rew


def reward_1_EXPIRED_ONLY(s_prime, a, drone, simulator, TARGET_VIOLATION_FACTOR, N_ACTIONS):
    """ NO PENALTY TO END STATES ENDLESS BATTERY """

    rew = - sum([min(i, TARGET_VIOLATION_FACTOR) for i in s_prime.aoi_idleness_ratio(False) if i >= 1])

    rew = min_max_normalizer(rew,
                             startUB=0,
                             startLB=(-(TARGET_VIOLATION_FACTOR * N_ACTIONS)),
                             endUB=0,
                             endLB=-1)
    return rew

def reward_2_NOT_ONY_EXPIRED(s_prime, a, drone, simulator, TARGET_VIOLATION_FACTOR, N_ACTIONS):
    """ NO PENALTY TO END STATES ENDLESS BATTERY """

    rew = - sum([min(i, TARGET_VIOLATION_FACTOR) for i in s_prime.aoi_idleness_ratio(False)])

    rew = min_max_normalizer(rew,
                             startUB=0,
                             startLB=(-(TARGET_VIOLATION_FACTOR * N_ACTIONS)),
                             endUB=0,
                             endLB=-1)
    return rew


def reward_3_POSITIVE_N(s_prime, a, drone, simulator, TARGET_VIOLATION_FACTOR, N_ACTIONS):
    """ NO PENALTY TO END STATES ENDLESS BATTERY """
    rew = 0
    for i in s_prime.aoi_idleness_ratio(False):
        rew += 1 if i < 1 else - min(i, TARGET_VIOLATION_FACTOR)

    rew = min_max_normalizer(rew,
                             startUB=N_ACTIONS,
                             startLB=(-(TARGET_VIOLATION_FACTOR * N_ACTIONS)),
                             endUB=1,
                             endLB=-1)
    return rew


def reward_4_POSITIVE_N_BOTH(s_prime, a, drone, simulator, TARGET_VIOLATION_FACTOR, N_ACTIONS):
    """ NO PENALTY TO END STATES ENDLESS BATTERY """
    rew = 0
    for i in s_prime.aoi_idleness_ratio(False):
        rew += (1-i) if i < 1 else - min(i, TARGET_VIOLATION_FACTOR)

    rew = min_max_normalizer(rew,
                             startUB=N_ACTIONS,
                             startLB=(-(TARGET_VIOLATION_FACTOR * N_ACTIONS)),
                             endUB=1,
                             endLB=-1)
    return rew

def reward_5_TEMPORAL_POSITIVE_N_BOTH(s, a, drone, simulator, TARGET_VIOLATION_FACTOR, N_ACTIONS):
    rew = 0

    time_dist_to_a = s.time_distances(False)[a]
    n_steps = max(int(time_dist_to_a / config.DELTA_DEC), 1)

    for step in range(n_steps):
        # count from the head of the arrow
        TIME = simulator.current_second() - (config.DELTA_DEC * step)

        for target in simulator.environment.targets:
            LAST_VISIT = target.last_visit_ts[drone.identifier] * simulator.ts_duration_sec
            residual = (TIME - LAST_VISIT) / target.maximum_tolerated_idleness
            rew += (1-residual) if residual < 1 else - min(residual, TARGET_VIOLATION_FACTOR)

    # rew += self.simulator.penalty_on_bs_expiration if s_prime.is_final else 0
    # rew = -100 if s.vector()[a] == 0 else rew

    # NOW NORMALIZE
    n_steps = int(simulator.max_travel_time() / config.DELTA_DEC)
    rew = min_max_normalizer(rew,
                             startLB=-TARGET_VIOLATION_FACTOR * n_steps * N_ACTIONS,  # TODO FIX ACTIONS /2
                             startUB=n_steps * N_ACTIONS,
                             endLB=-1,
                             endUB=1)

    return rew