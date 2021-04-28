
from enum import Enum

"""
This file contains all the constants and parameters of the simulator.
It comes handy when you want to make one shot simulations, making parameters and constants vary in every
simulation. For an extensive experimental campaign read the header at src.simulator.
"""

# ----------------------------- SIMULATION PARAMS ---------------------------- #

SIM_SEED = 24                # int: seed of this simulation.
SIM_DURATION = 24000*24*5      # int: steps of simulation. (np.inf)
SIM_TS_DURATION = 0.150     # float: seconds duration of a step in seconds.

ENV_WIDTH = 1500      # float: meters, width of environment
ENV_HEIGHT = 1500     # float: meters, height of environment

N_DRONES = 1         # int: number of drones.
N_OBSTACLES = 0      # number of random obstacles in the map
N_GRID_CELLS = 5    # number of cless in the grid

# base station
N_BASE_STATIONS = 1
BASE_STATION_COORDS = [ENV_WIDTH / 2, 0]   # coordinates of the base staion
BASE_STATION_COM_RANGE = 200               # float: meters, communication range of the depot.

# IMPORTANT: coordinates of the drones at the beginning, it can be NONE in that case drone will follow
# fixed tours determined in FIXED_TOURS_DIR
DRONE_COORDS = BASE_STATION_COORDS

DRONE_SPEED = 10              # float: m/s, drone speed.
DRONE_ANGLE = 0               # degrees (0, 359)
DRONE_SPEED_INCREMENT = 5     # increment at every key stroke
DRONE_ANGLE_INCREMENT = 45    # increment at every key stroke
DRONE_COM_RANGE = 100         # float: meters, communication range of the drones.
DRONE_SENSING_RANGE = 0       # float: meters, the sensing range of the drones.
DRONE_MAX_BUFFER_SIZE = 0     # int: max number of packets in the buffer of a drone.
DRONE_MAX_ENERGY = int(3600/4)        # int: max energy of a drone sec
DRONE_RADAR_RADIUS = 60       # meters

# map
PLOT_TRAJECTORY_NEXT_TARGET = True

# ------------------------------- CONSTANTS ------------------------------- #

FIXED_TOURS_DIR = "data/tours/"        # str: the path to the drones tours
DEMO_PATH = False                      # bool: whether to use handcrafted tours or not (in utilities.utilities)

PLOT_SIM = True       # bool: whether to plot or not the simulation (set to false for faster experiments)
WAIT_SIM_STEP = 0     # float >= 0: seconds, pauses the rendering for x seconds
SKIP_SIM_STEP = 5     # int > 0 : steps, plot the simulation every x steps
DRAW_SIZE = 700       # int: size of the drawing window

SAVE_PLOT = False              # bool: whether to save the plots of the simulation or not
SAVE_PLOT_DIR = "data/plots/"  # string: where to save plots

# ------------------------------- PATROLLING ------------------------------- #
N_TARGETS = 7       # number of random targets in the map

TARGETS = [(BASE_STATION_COORDS[0]-600, BASE_STATION_COORDS[1]+400, 100),
           (BASE_STATION_COORDS[0]-250, BASE_STATION_COORDS[1]+1100, 100),
           (BASE_STATION_COORDS[0]+250, BASE_STATION_COORDS[1]+1100, 100),
           (BASE_STATION_COORDS[0]+600, BASE_STATION_COORDS[1]+400, 100)]


class Mobility(Enum):
    FREE = 0
    PLANNED = 1
    DECIDED = 2

    RANDOM_MOVEMENT = 3
    GO_MAX_AOI = 4
    GO_MIN_RESIDUAL = 5
    GO_MIN_SUM_RESIDUAL = 6

    OPTIMAL_SOFT_CONSTRAINT = 7   # the deadline can be violated, but obj to meet most of them
    OPTIMAL_HARD_CONSTRAINT = 8   # the deadline can'T be violated

DRONE_MOBILITY = Mobility.GO_MIN_SUM_RESIDUAL
RL_DATA = "data/rl/"
RL_MODEL = ""
PRE_TRAINED = False


