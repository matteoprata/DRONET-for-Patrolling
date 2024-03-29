
import os
import sys
module_path = os.path.abspath(os.getcwd())
if module_path not in sys.path:
    sys.path.append(module_path)

import argparse
import src.constants as cst
from src.world_entities.simulator_patrolling import PatrollingSimulator
from src.config import Configuration


def main(configuration):
    configuration.run_parameters_sanity_check()
    print("\nExecuting > {}\n".format(configuration.conf_description()))
    sim = PatrollingSimulator(configuration)
    sim.run_testing_loop()
    # sim.run_training_loop()


def parser_cl_arguments(configuration: Configuration):
    """ Parses the arguments for the command line. """

    parser = argparse.ArgumentParser(description='Patrolling Simulator arguments:')

    parser.add_argument('-seed', '--SEED', default=configuration.SEED, type=int)
    parser.add_argument('-pol', '--DRONE_PATROLLING_POLICY', default=configuration.DRONE_PATROLLING_POLICY, type=str)

    parser.add_argument('-tol', '--TARGETS_TOLERANCE_FIXED', default=configuration.TARGETS_TOLERANCE_FIXED, type=float)
    parser.add_argument('-nt',  '--TARGETS_NUMBER', default=configuration.TARGETS_NUMBER, type=int)
    parser.add_argument('-nd',  '--DRONES_NUMBER', default=configuration.DRONES_NUMBER, type=int)
    parser.add_argument('-spe', '--DRONE_SPEED', default=configuration.DRONE_SPEED, type=float)
    parser.add_argument('-bat', '--DRONE_MAX_ENERGY', default=configuration.DRONE_MAX_ENERGY, type=float)

    parser.add_argument('-edu', '--EPISODE_DURATION', default=configuration.EPISODE_DURATION, type=int)
    parser.add_argument('-pl',  '--PLOT_SIM', default=configuration.PLOT_SIM, type=int)
    parser.add_argument('-mpa', '--MODEL_PATH', default='none', type=str)

    # python src/main_single_test.py -pl 1 -nd 1 -nt 5 -seed 103 -tol .1 -pol GO_MIN_RESIDUAL
    # parsing arguments from cli
    args = vars(parser.parse_args())

    print("Setting parameters...")

    configuration.SEED = args["SEED"]
    configuration.PLOT_SIM = bool(args["PLOT_SIM"])
    configuration.TARGETS_TOLERANCE_FIXED = args["TARGETS_TOLERANCE_FIXED"]
    configuration.TARGETS_NUMBER = args["TARGETS_NUMBER"]
    configuration.DRONES_NUMBER = args["DRONES_NUMBER"]
    configuration.DRONE_SPEED = args["DRONE_SPEED"]

    configuration.DRONE_MAX_ENERGY = args["DRONE_MAX_ENERGY"]
    configuration.EPISODE_DURATION = args["EPISODE_DURATION"]

    if type(args["DRONE_PATROLLING_POLICY"]) == str:
        if is_enum_key(args["DRONE_PATROLLING_POLICY"], cst.OnlinePatrollingProtocol):
            configuration.DRONE_PATROLLING_POLICY = cst.OnlinePatrollingProtocol[args["DRONE_PATROLLING_POLICY"]]
        elif is_enum_key(args["DRONE_PATROLLING_POLICY"], cst.PrecomputedPatrollingProtocol):
            configuration.DRONE_PATROLLING_POLICY = cst.PrecomputedPatrollingProtocol[args["DRONE_PATROLLING_POLICY"]]
    else:
        configuration.DRONE_PATROLLING_POLICY = args["DRONE_PATROLLING_POLICY"]  # ?

    if configuration.DRONE_PATROLLING_POLICY == cst.OnlinePatrollingProtocol.RL_DECISION_TEST:
        if os.path.exists(args["MODEL_PATH"]):
            configuration.RL_BEST_MODEL_PATH = args["MODEL_PATH"]
        else:
            print("Path {} is invalid for testing a pretrained RL model.".format(args["MODEL_PATH"]))
            exit(1)


def is_enum_key(key:str, enum):
    try:
        enum[key]
        return True
    except:
        return False


if __name__ == "__main__":
    # conf = Configuration()
    # parser_cl_arguments(conf)
    # main(conf)

    configuration = Configuration()

    configuration.SEED = 1
    configuration.PLOT_SIM = True
    configuration.TARGETS_NUMBER = 10
    configuration.DRONE_SPEED = 5
    configuration.TARGETS_POSITION_SCENARIO = cst.PositionScenario.CLUSTERED

    configuration.EPISODE_DURATION = 500
    configuration.N_EPISODES_TRAIN = 5
    configuration.N_EPISODES_TEST = 5
    configuration.N_EPISODES_VAL = 5

    main(configuration)

