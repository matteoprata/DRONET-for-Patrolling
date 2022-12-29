
from enum import Enum

from src.constants import PatrollingProtocol


class LearningHyperParameters(Enum):

    BETA = "beta"
    REPLAY_MEMORY_DEPTH = "replay_memory_depth"
    EPSILON_DECAY = "epsilon_decay"
    LEARNING_RATE = "learning_rate"
    DISCOUNT_FACTOR = "discount_factor"
    BATCH_SIZE = "batch_size"
    SWAP_MODELS_EVERY_DECISION = "swap_models_every_decision"

    N_HIDDEN_1 = "n_hidden_neurons_lv1"
    N_HIDDEN_2 = "n_hidden_neurons_lv2"
    N_HIDDEN_3 = "n_hidden_neurons_lv3"

    OPTIMIZER = "optimizer"
    LOSS = "loss"


DQN_LEARNING_HYPER_PARAMETERS = {
    # "set" is the chosen value
    LearningHyperParameters.BETA.value: {'values': [1]},
    LearningHyperParameters.REPLAY_MEMORY_DEPTH.value: {'values': [100000]},
    LearningHyperParameters.EPSILON_DECAY.value: {'values': [1]},
    LearningHyperParameters.LEARNING_RATE.value:  {'min': 0.0001, 'max': 0.001},
    LearningHyperParameters.DISCOUNT_FACTOR.value: {'values': [1]},
    LearningHyperParameters.BATCH_SIZE.value: {'values': [32, 64]},
    LearningHyperParameters.SWAP_MODELS_EVERY_DECISION.value: {'values': [500]},

    LearningHyperParameters.N_HIDDEN_1.value: {'values': [10]},
    LearningHyperParameters.N_HIDDEN_2.value: {'values': [1]},
    LearningHyperParameters.N_HIDDEN_3.value: {'values': [1]},

    LearningHyperParameters.OPTIMIZER.value: {'values': ["sdg"]},
    LearningHyperParameters.LOSS.value: {'values': ["mse"]},
}


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
        self.DRONE_MAX_ENERGY = int(10 * self.MIN)                         # int: max energy of a drone steps

        self.N_EPOCHS = 1                           # how many times you will see the same scenario
        self.EPISODE_DURATION = int(1 * self.HOUR)  # how much time the episode lasts steps

        self.N_EPISODES_TRAIN = 1  # how many times the scenario (a.k.a. episode) changes during a simulation
        self.N_EPISODES_VAL = 0    # how many times the scenario (a.k.a. episode) changes during a simulation
        self.N_EPISODES_TEST = 0  # how many times the scenario (a.k.a. episode) changes during a simulation

        self.DELTA_DEC = 5                 # after how many seconds a new decision must take place
        self.IS_DECIDED_ON_TARGET = False  # the decision step happens on target visited (non uniformity of the decision step), or every DELTA_DEC
        self.IS_ALLOW_SELF_LOOP = True     # drone can decide to visit the same target in two consecutive decisions or not

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
        self.FUNCTION_TO_OPTIMIZE = self.FUNCTION_TO_OPTIMIZE = {
            'goal': 'maximize',
            'name': "cumulative_reward"
        }

        self.DQN_PARAMETERS = {
            LearningHyperParameters.BETA: None,
            LearningHyperParameters.REPLAY_MEMORY_DEPTH: None,
            LearningHyperParameters.EPSILON_DECAY: None,
            LearningHyperParameters.LEARNING_RATE: None,
            LearningHyperParameters.DISCOUNT_FACTOR: None,
            LearningHyperParameters.BATCH_SIZE: None,
            LearningHyperParameters.SWAP_MODELS_EVERY_DECISION: None,

            LearningHyperParameters.N_HIDDEN_1: None,
            LearningHyperParameters.N_HIDDEN_2: None,
            LearningHyperParameters.N_HIDDEN_3: None,

            LearningHyperParameters.OPTIMIZER: None,
            LearningHyperParameters.LOSS: None
        }

        # paths
        self.RL_DATA = "data/rl/"

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
        self.IS_TRAINING_MODE = False
        self.TIME_DENSITY_METRICS = 5000  # density on the X axis of AOI ratio plots

        self.IS_WANDB = False
        self.WANDB_INSTANCE = None
        self.IS_HIDE_PROGRESS_BAR = False

    def conf_description(self):
        return "seed={}_nd={}_nt={}_pol={}_sp={}_tolf={}".format(self.SEED, self.DRONES_NUMBER, self.TARGETS_NUMBER,
                                                                 self.DRONE_PATROLLING_POLICY.name, self.DRONE_SPEED, self.TARGETS_TOLERANCE)

    def n_tot_episodes(self):
        return self.N_EPISODES_TRAIN + self.N_EPISODES_TEST + self.N_EPISODES_VAL
