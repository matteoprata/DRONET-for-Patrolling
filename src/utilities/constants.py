from enum import Enum
import multiprocessing
from src.patrolling import patrolling_baselines as pbase


class Mobility(Enum):
    FREE = None
    RL_DECISION_TRAIN = None

    RANDOM_MOVEMENT     = pbase.RandomPolicy
    GO_MAX_AOI          = pbase.MaxAOIPolicy
    GO_MIN_RESIDUAL     = pbase.MaxAOIRatioPolicy
    GO_MIN_SUM_RESIDUAL = pbase.MaxSumResidualPolicy
    RL_DECISION_TEST    = pbase.RLPolicy


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


class HyperParameters(Enum):
    LR = "learning_rate"
    DISCOUNT_FACTOR = "discount_factor"
    SWAP_MODELS = "swap_models_every_decision"

    MLP_HID1 = "n_hidden_neurons_lv1"
    MLP_HID2 = "n_hidden_neurons_lv2"
    MLP_HID3 = "n_hidden_neurons_lv3"

    IS_SELF_LOOP = "is_allow_self_loop"
    BATTERY = "battery"
    EPISODE_DURATION = "episode_duration"
    N_EPISODES = "n_episodes"
    N_EPOCHS = "n_epochs"
    DURATION_EPISODE = "episode_duration"


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

PROJECT_NAME = "uavsimulator_patrolling"
