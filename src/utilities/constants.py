from enum import Enum
import multiprocessing


class Mobility(Enum):
    FREE = 0
    FIXED_TRAJECTORIES = 1  # ignore
    RL_DECISION = 2

    RANDOM_MOVEMENT = 3
    GO_MAX_AOI = 4
    GO_MIN_RESIDUAL = 5
    GO_MIN_SUM_RESIDUAL = 6

    MICHELE = 7


class IndependentVariable(Enum):
    SEED = {"ID": 1, "NAME": "Seed"}
    ALGORITHM = {"ID": 2, "NAME": "Algorithm"}

    # PICK FROM HERE:
    DRONES_SPEED = {"ID": 3, "NAME": "Drones Speed"}
    DRONES_NUMBER = {"ID": 4, "NAME": "Drones Number"}
    TARGETS_NUMBER = {"ID": 5, "NAME": "Targets Number"}
    TARGETS_TOLERANCE = {"ID": 6, "NAME": "Tolerance Factor"}


class DependentVariable(Enum):
    AOI_ABSOLUTE = {"NAME": "Absolute AOI"}
    AOI_RATIO = {"NAME": "Ratio AOI"}

    # distinct target distribution & averaged targets
    CUMULATIVE_AR =       {"NAME": "Integral AOI"}
    CUMULATIVE_DELAY_AR = {"NAME": "Cumulative Delay AOI"}
    WORST_DELAY =         {"NAME": "Worst Delay"}
    WORST_AGE =           {"NAME": "Worst Age"}
    VIOLATION_NUMBER =    {"NAME": "Number of Violations"}


class JSONFields(Enum):
    # L0
    VISIT_TIMES = "visit_times"
    SIMULATION_INFO = "info"

    # L1
    TOLERANCE = "targets_tolerance"
    EPISODE_DURATION = "episode_duration"
    TS_DURATION = "ts_duration_sec"


PATH_STATS = "data/experiments/"
N_CORES = multiprocessing.cpu_count()-2
