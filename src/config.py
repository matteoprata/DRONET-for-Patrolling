
from enum import Enum

from src.constants import PatrollingProtocol, DependentVariable
from src.utilities.utilities import euclidean_distance
import numpy as np


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

    # OPTIMIZER = "optimizer"
    # LOSS = "loss"


class Configuration:
    """ This class represent all the constants of a simulation, they vary from one run to another. """
    def __init__(self):

        self.SETUP_NAME = None

        # Frequently used parameters
        self.SIM_TS_DURATION = 0.150       # float: seconds duration of a step in seconds.
        self.SEED = 100                # int: seed of this simulation

        self.DAY = int(60 * 60 * 24 / self.SIM_TS_DURATION)
        self.HOUR = int(60 * 60 / self.SIM_TS_DURATION)
        self.MIN = int(60 / self.SIM_TS_DURATION)

        self.TARGETS_NUMBER = 5       # number of random targets in the map
        self.TARGETS_TOLERANCE = 0.1  # std % distance of tolerance generation

        self.DRONES_NUMBER = 1                                                  # int: number of drones.
        self.DRONE_SPEED = 15                                              # 15 m/s = 54 km/h   # float: m/s, drone speed.
        self.DRONE_PATROLLING_POLICY = PatrollingProtocol.RANDOM_MOVEMENT  #
        self.DRONE_MAX_ENERGY = int(2 * self.HOUR)                         # int: max energy of a drone steps

        self.N_EPOCHS = 1                           # how many times you will see the same scenario
        self.EPISODE_DURATION = int(1 * self.HOUR)  # how much time the episode lasts steps

        self.N_EPISODES_TRAIN = 0  # how many times the scenario (a.k.a. episode) changes during a simulation
        self.N_EPISODES_VAL = 0    # how many times the scenario (a.k.a. episode) changes during a simulation
        self.N_EPISODES_TEST = 1  # how many times the scenario (a.k.a. episode) changes during a simulation

        self.DELTA_DEC = 5                 # after how many seconds a new decision must take place
        self.IS_DECIDED_ON_TARGET = False  # the decision step happens on target visited (non uniformity of the decision step), or every DELTA_DEC
        self.IS_ALLOW_SELF_LOOP = False     # drone can decide to visit the same target in two consecutive decisions or not

        # algorithms to play with
        self.VALIDATION_ALGORITHMS = [
            PatrollingProtocol.GO_MIN_SUM_RESIDUAL,
            PatrollingProtocol.RANDOM_MOVEMENT,
            PatrollingProtocol.GO_MIN_RESIDUAL,
            PatrollingProtocol.GO_MAX_AOI
                                      ]

        self.VALIDATION_VARS = [
            DependentVariable.CUMULATIVE_AR,
            DependentVariable.CUMULATIVE_DELAY_AR,
            DependentVariable.WORST_DELAY,
            DependentVariable.WORST_AGE,
            DependentVariable.VIOLATION_NUMBER
        ]

        # ----------------------------- SIMULATION PARAMS ---------------------------- #

        self.ENV_WIDTH = 1500      # float: meters, width of environment
        self.ENV_HEIGHT = 1500     # float: meters, height of environment

        self.N_OBSTACLES = 0      # number of random obstacles in the map
        self.N_GRID_CELLS = 0     # number of cells in the grid

        # base station
        self.N_BASE_STATIONS = 1
        self.BASE_STATION_COORDS = [self.ENV_WIDTH / 2, 0]   # coordinates of the base staion
        self.BASE_STATION_COM_RANGE = 0                      # float: meters, communication range of the depot.

        # IMPORTANT: coordinates of the drones at the beginning, it can be NONE in that case drone will follow
        # fixed tours determined in FIXED_TOURS_DIR
        self.DRONE_COORDS = self.BASE_STATION_COORDS

        # FREE MOVEMENT
        self.DRONE_ANGLE = 0               # degrees (0, 359)
        self.DRONE_SPEED_INCREMENT = 5     # increment at every key stroke
        self.DRONE_ANGLE_INCREMENT = 45    # increment at every key stroke
        self.DRONE_COM_RANGE = 100         # float: meters, communication range of the drones.
        self.DRONE_SENSING_RANGE = 0       # float: meters, the sensing range of the drones.
        self.DRONE_MAX_BUFFER_SIZE = 0     # int: max number of packets in the buffer of a drone.
        self.DRONE_RADAR_RADIUS = 60       # meters
        self.DRONE_SENSING_HOVERING = 60*5   # seconds

        # map
        self.PLOT_TRAJECTORY_NEXT_TARGET = True   # shows the segment from the drone to its next waypoint

        # ------------------------------ CONSTANTS ------------------------------- #

        self.FIXED_TOURS_DIR = "data/tours/"  # str: the visited_targets_coordinates to the drones tours
        self.DEMO_PATH = False                # bool: whether to use handcrafted tours or not (in utilities.utilities)

        self.PLOT_SIM = False       # bool: whether to plot or not the simulation (set to false for faster experiments)
        self.WAIT_SIM_STEP = 0     # float >= 0: seconds, pauses the rendering for x seconds
        self.SKIP_SIM_STEP = 20    # int > 0 : steps, plot the simulation every x steps
        # self.DRAW_SIZE = 700       # int: size of the drawing window

        self.SAVE_PLOT = False              # bool: whether to save the plots of the simulation or not
        self.SAVE_PLOT_DIR = "data/plots/"  # string: where to save plots
        self.DRAW_SIZE = 700

        # ------------------------------- WANDB ------------------------------- #

        self.PROJECT_NAME = "RL_Patrolling"
        self.HYPER_PARAM_SEARCH_MODE = 'bayes'
        self.FUNCTION_TO_OPTIMIZE = {
            'goal': 'maximize',
            'name': "cumulative_reward"
        }

        # ASSIGNED for a run
        self.DQN_PARAMETERS = {
            LearningHyperParameters.REPLAY_MEMORY_DEPTH: 100000,
            LearningHyperParameters.EPSILON_DECAY: 0.0001,
            LearningHyperParameters.LEARNING_RATE: 0.0001,
            LearningHyperParameters.DISCOUNT_FACTOR: 1,
            LearningHyperParameters.BATCH_SIZE: 30,
            LearningHyperParameters.SWAP_MODELS_EVERY_DECISION: 100,

            LearningHyperParameters.N_HIDDEN_1: 10,
            LearningHyperParameters.N_HIDDEN_2: 0,
            LearningHyperParameters.N_HIDDEN_3: 0,
            LearningHyperParameters.N_HIDDEN_4: 0,
            LearningHyperParameters.N_HIDDEN_5: 0,

            # LearningHyperParameters.OPTIMIZER: None,
            # LearningHyperParameters.LOSS: None
        }

        # paths
        self.RL_BEST_MODEL_PATH = "data/rl/"

        # how much exploration, careful to edit
        self.ZERO_TOLERANCE = 0.1     # 10% at 80% of the simulation
        self.EXPLORE_PORTION = 0.7    # what portion of time of the simulation is spent exploring

        # variables from here
        self.LOG_STATE = False                       # print the state or not

        self.TARGET_VIOLATION_FACTOR = 100  # ?

        self.IS_RESIDUAL_REWARD = False                                        # ?
        self.PENALTY_ON_BS_EXPIRATION = - self.TARGETS_NUMBER * self.TARGET_VIOLATION_FACTOR  # reward due to the violation of the base station (i.e. the drone dies)
        self.OK_VISIT_RADIUS = 0  # radius of a target, suffices to visit it IGNORE

        self.IS_PARALLEL_EXECUTION = False
        self.TIME_DENSITY_METRICS = 5000  # density on the X axis of AOI ratio plots

        self.IS_WANDB = False
        self.WANDB_INSTANCE = None
        self.IS_HIDE_PROGRESS_BAR = False

    def conf_description(self):
        return "seed={}_nd={}_nt={}_pol={}_sp={}_tolf={}".format(self.SEED, self.DRONES_NUMBER, self.TARGETS_NUMBER,
                                                                 self.DRONE_PATROLLING_POLICY.name, self.DRONE_SPEED, self.TARGETS_TOLERANCE)

    def n_tot_episodes(self):
        return self.N_EPISODES_TRAIN + self.N_EPISODES_TEST + self.N_EPISODES_VAL

    def seconds_to_ts(self, seconds):
        return int(seconds / self.SIM_TS_DURATION)

    def ts_tp_seconds(self, tss):
        return tss * self.SIM_TS_DURATION

    def max_time_distance(self):
        return np.sqrt(self.ENV_WIDTH**2 + self.ENV_HEIGHT**2) / self.DRONE_SPEED

    def is_rl_training(self):
        return self.DRONE_PATROLLING_POLICY == PatrollingProtocol.RL_DECISION_TRAIN

    def is_rl_testing(self):
        return self.DRONE_PATROLLING_POLICY == PatrollingProtocol.RL_DECISION_TEST

    def max_times_violation(self):
        return 1000


# SWEEP
DQN_LEARNING_HYPER_PARAMETERS = {
    # "set" is the chosen value
    LearningHyperParameters.REPLAY_MEMORY_DEPTH.value: {'values': [100000]},
    LearningHyperParameters.EPSILON_DECAY.value: {'min': 0.00001, 'max': 0.001},
    LearningHyperParameters.LEARNING_RATE.value:  {'min': 0.0001, 'max': 0.001},
    LearningHyperParameters.DISCOUNT_FACTOR.value: {'values': [1, 0.8]},
    LearningHyperParameters.BATCH_SIZE.value: {'values': [32, 64]},
    LearningHyperParameters.SWAP_MODELS_EVERY_DECISION.value: {'values': [100, 500]},

    LearningHyperParameters.N_HIDDEN_1.value: {'values': [10, 20, 30]},
    LearningHyperParameters.N_HIDDEN_2.value: {'values': [0, 5, 10]},
    LearningHyperParameters.N_HIDDEN_3.value: {'values': [0]},
    LearningHyperParameters.N_HIDDEN_4.value: {'values': [0]},
    LearningHyperParameters.N_HIDDEN_5.value: {'values': [0]},

    # LearningHyperParameters.OPTIMIZER.value: {'values': ["adam"]},
    # LearningHyperParameters.LOSS.value: {'values': ["mse"]},
}
