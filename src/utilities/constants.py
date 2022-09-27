from enum import Enum


class Mobility(Enum):
    FREE = 0
    FIXED_TRAJECTORIES = 1  # ignore
    RL_DECISION = 2

    RANDOM_MOVEMENT = 3
    GO_MAX_AOI = 4
    GO_MIN_RESIDUAL = 5
    GO_MIN_SUM_RESIDUAL = 6

    MICHELE = 7


PATH_STATS = "data/experiments/"