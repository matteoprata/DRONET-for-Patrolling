from enum import Enum
import multiprocessing

from src.patrolling.base_max_aoi import MaxAOIPolicy
from src.patrolling.base_random import RandomPolicy
from src.patrolling.base_max_sum_aoi_ratio import MaxSumResidualPolicy
from src.patrolling.base_max_aoi_ratio import MaxAOIRatioPolicy

from src.patrolling.base_clustering_tsp import ClusteringTSP

"""
This file contains all the constants of the sim.
"""


class LearningHyperParameters(Enum):

    REPLAY_MEMORY_DEPTH = "replay_memory_depth"
    EPSILON_DECAY = "epsilon_decay"
    LEARNING_RATE = "learning_rate"
    DISCOUNT_FACTOR = "discount_factor"
    BATCH_SIZE = "batch_size"
    SWAP_MODELS_EVERY_DECISION = "swap_models_every_decision"

    N_HIDDEN_1 = "n_hidden_neurons_lv1"
    N_HIDDEN_2 = "n_hidden_neurons_lv2"
    N_HIDDEN_3 = "n_hidden_neurons_lv3"
    N_HIDDEN_4 = "n_hidden_neurons_lv4"
    N_HIDDEN_5 = "n_hidden_neurons_lv5"

    PERCENTAGE_SWAP = "percentage_swap"
    # OPTIMIZER = "optimizer"
    # LOSS = "loss"


class ErrorType(Enum):
    STD = "std"
    STD_ERROR = "stde"


class EpisodeType(Enum):
    TRAIN = "train"
    TEST = "test"
    VAL = "validation"


class PatrollingProtocol(Enum):
    FREE = 1
    RL_DECISION_TRAIN = 0
    RL_DECISION_TEST    = 2

    RANDOM_MOVEMENT     = RandomPolicy
    GO_MAX_AOI          = MaxAOIPolicy
    GO_MIN_RESIDUAL     = MaxAOIRatioPolicy
    GO_MIN_SUM_RESIDUAL = MaxSumResidualPolicy


class PrecomputedPatrollingProtocol(Enum):
    MULTI_TSP = ClusteringTSP


class TargetFamily(Enum):
    PURPLE = 'purple'
    BLUE = 'blue'
    GREEN = 'green'


class DroneFamily(Enum):
    PURPLE = 'purple'
    BLUE = 'blue'


class IndependentVariable(Enum):
    SEED = {"ID": 1, "NAME": "Seed"}
    DRONE_PATROLLING_POLICY = {"ID": 2, "NAME": "Algorithm"}

    # PICK FROM HERE:
    DRONE_SPEED = {"ID": 3, "NAME": "Drones Speed"}
    DRONES_NUMBER = {"ID": 4, "NAME": "Drones Number"}
    TARGETS_NUMBER = {"ID": 5, "NAME": "Targets Number"}
    TARGETS_TOLERANCE_FIXED = {"ID": 7, "NAME": "Tolerance"}


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

    VAL_EPISODE_ID = "VAL_EPISODE_ID"
    VAL_EPISODE_ALGO = "VAL_EPISODE_ALGO"
    DRONE_NUMBER = "DRONE_NUMBER"
    TARGET_NUMBER = "TARGET_NUMBER"
    DRONE_SPEED = "DRONE_SPEED"
    TOLERANCE_FACTOR = "TOLERANCE_FACTOR"


class RLRewardType(Enum):
    REW0 = 1
    REW1 = 2
    REW2 = 3
    REW3 = 4


PATH_STATS = "data/experiments/"
N_CORES = multiprocessing.cpu_count()-1

from src.simulation_setup import setup01, setup02, progetto_iot_setup


class Setups(Enum):
    SETUP01 = setup01
    SETUP02 = setup02
    IOT = progetto_iot_setup


TORCH_DEVICE = 'cpu'  # 'cuda' if torch.cuda.is_available() else 'cpu'
