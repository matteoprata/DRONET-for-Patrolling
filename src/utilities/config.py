
from enum import Enum

from src.utilities.constants import Mobility

"""
This file contains all the constants and parameters of the simulator.
It comes handy when you want to make one shot simulations, making parameters and constants vary in every
simulation. For an extensive experimental campaign read the header at src.simulator.
"""

# ----------------------------- SIMULATION PARAMS ---------------------------- #

SIM_DESCRIPTION = "default"
SIM_SEED = 100                # int: seed of this simulation
SIM_DURATION = 24000*24*10    # int: steps of simulation. (np.inf)
SIM_TS_DURATION = 0.150       # float: seconds duration of a step in seconds.

ENV_WIDTH = 1500      # float: meters, width of environment
ENV_HEIGHT = 1500     # float: meters, height of environment

N_DRONES = 1         # int: number of drones.
N_OBSTACLES = 0      # number of random obstacles in the map
N_GRID_CELLS = 0     # number of cells in the grid

# base station
N_BASE_STATIONS = 1
BASE_STATION_COORDS = [ENV_WIDTH / 2, 0]   # coordinates of the base staion
BASE_STATION_COM_RANGE = 0                 # float: meters, communication range of the depot.

# IMPORTANT: coordinates of the drones at the beginning, it can be NONE in that case drone will follow
# fixed tours determined in FIXED_TOURS_DIR
DRONE_COORDS = BASE_STATION_COORDS

# FREE MOVEMENT
DRONE_ANGLE = 0               # degrees (0, 359)
DRONE_SPEED_INCREMENT = 5     # increment at every key stroke
DRONE_ANGLE_INCREMENT = 45    # increment at every key stroke
DRONE_COM_RANGE = 100         # float: meters, communication range of the drones.
DRONE_SENSING_RANGE = 0       # float: meters, the sensing range of the drones.
DRONE_MAX_BUFFER_SIZE = 0     # int: max number of packets in the buffer of a drone.
DRONE_RADAR_RADIUS = 60       # meters

# map
PLOT_TRAJECTORY_NEXT_TARGET = True   # shows the segment from the drone to its next waypoint

# ------------------------------ CONSTANTS ------------------------------- #

FIXED_TOURS_DIR = "data/tours/"        # str: the path to the drones tours
DEMO_PATH = False                      # bool: whether to use handcrafted tours or not (in utilities.utilities)

PLOT_SIM = False      # bool: whether to plot or not the simulation (set to false for faster experiments)
WAIT_SIM_STEP = .1     # float >= 0: seconds, pauses the rendering for x seconds
SKIP_SIM_STEP = 20    # int > 0 : steps, plot the simulation every x steps
DRAW_SIZE = 700       # int: size of the drawing window

SAVE_PLOT = False              # bool: whether to save the plots of the simulation or not
SAVE_PLOT_DIR = "data/plots/"  # string: where to save plots

# ------------------------------- PATROLLING ------------------------------- #


class Time(Enum):
    """ number of steps to simulate specific time """
    DAY = int(60*60*24/SIM_TS_DURATION)
    HOUR = int(60*60/SIM_TS_DURATION)
    MIN = int(60/SIM_TS_DURATION)


LEARNING_PARAMETERS = {
    "is_pretrained": False,
    "model_name": "data/rl/model.h5",
    "beta": None,  # for continuous tasks
    "replay_memory_depth": 100000,
    "epsilon_decay": None,

    "learning_rate":  0.001,
    "discount_factor": 1,
    "batch_size": 32,
    "swap_models_every_decision": 500,
    "n_hidden_neurons_lv1": 10,
    "n_hidden_neurons_lv2": 1,
    "n_hidden_neurons_lv3": 1,
    "optimizer": "sgd",
    "loss": "mse"
}

# paths
RL_DATA = "data/rl/"
TARGETS_FILE = "data/targets/"
YAML_FILE = "wandb_sweep_bayesian.yaml"

# how much exploration, careful to edit
ZERO_TOLERANCE = 0.1     # 10% at 80% of the simulation
EXPLORE_PORTION = 0.7    # what portion of time of the simulation is spent exploring

# variables from here
DRONE_MAX_ENERGY = 3 * Time.MIN.value       # int: max energy of a drone sec
DRONE_SPEED = 15                # 54 km/h   # float: m/s, drone speed.
N_TARGETS = 5                              # number of random targets in the map

LOG_STATE = False                       # print the state or not

N_EPISODES = 10                         # how many times the scenario (a.k.a. episode) changes during a simulation
N_EPOCHS = 2                            # how many times you will see the same scenario
EPISODE_DURATION = int(1 * Time.HOUR.value)  # how much time the episode lasts

TARGET_VIOLATION_FACTOR = 5    # ?
TOLERANCE_FACTOR = 0.1  # std % distance of tolerance generation

IS_DECIDED_ON_TARGET = False  # the decision step happens on target visited (non uniformity of the decision step), or every DELTA_DEC
DELTA_DEC = 3                 # after how many seconds a new decision must take place

IS_RESIDUAL_REWARD = False                                        # ?
IS_ALLOW_SELF_LOOP = True                                         # drone can decide to visit the same target in two consecutive decisions or not
PENALTY_ON_BS_EXPIRATION = - N_TARGETS * TARGET_VIOLATION_FACTOR  # reward due to the violation of the base station (i.e. the drone dies)
OK_VISIT_RADIUS = 0  # radius of a target, suffices to visit it IGNORE

DRONE_MOBILITY = Mobility.RL_DECISION
IS_PARALLEL = True

IS_SWEEP = False
IS_TRAINING_MODE = IS_SWEEP
