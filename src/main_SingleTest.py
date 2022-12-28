
import argparse
import src.constants as cst
from src.simulation.simulator_patrolling import PatrollingSimulator
from src.config import Configuration


def main(configuration):

    print("\nExecuting > {}\n".format(configuration.conf_description()))
    sim = PatrollingSimulator(configuration)
    sim.run()


def parser_cl_arguments(configuration: Configuration):
    """ Parses the arguments for the command line. """

    parser = argparse.ArgumentParser(description='Patrolling Simulator arguments:')

    parser.add_argument('-seed', '--SEED', default=configuration.SEED, type=int)
    parser.add_argument('-pol', '--DRONE_PATROLLING_POLICY', default=configuration.DRONE_PATROLLING_POLICY, type=str)

    parser.add_argument('-tol', '--TARGETS_TOLERANCE', default=configuration.TARGETS_TOLERANCE, type=float)
    parser.add_argument('-nt',  '--TARGETS_NUMBER', default=configuration.TARGETS_NUMBER, type=int)
    parser.add_argument('-nd',  '--DRONES_NUMBER', default=configuration.DRONES_NUMBER, type=int)
    parser.add_argument('-spe', '--DRONE_SPEED', default=configuration.DRONE_SPEED, type=float)
    parser.add_argument('-bat', '--DRONE_MAX_ENERGY', default=configuration.DRONE_MAX_ENERGY, type=float)

    parser.add_argument('-ne',  '--N_EPISODES', default=configuration.N_EPISODES, type=int)
    parser.add_argument('-edu', '--EPISODE_DURATION', default=configuration.EPISODE_DURATION, type=int)
    parser.add_argument('-pl',  '--PLOT_SIM', default=configuration.PLOT_SIM, type=int)

    # python -m src.main_single_test -seed 1 -nt 10 -nd 2 -pol BASE_01 -pl 1 -ne 1
    # parsing arguments from cli
    args = vars(parser.parse_args())

    print("Setting parameters...")

    configuration.SEED = args["SEED"]
    configuration.PLOT_SIM = bool(args["PLOT_SIM"])
    configuration.TARGETS_TOLERANCE = args["TARGETS_TOLERANCE"]
    configuration.TARGETS_NUMBER = args["TARGETS_NUMBER"]
    configuration.DRONES_NUMBER = args["DRONES_NUMBER"]
    configuration.DRONE_SPEED = args["DRONE_SPEED"]

    configuration.DRONE_MAX_ENERGY = args["DRONE_MAX_ENERGY"]
    configuration.N_EPISODES = args["N_EPISODES"]
    configuration.EPISODE_DURATION = args["EPISODE_DURATION"]

    if type(args["DRONE_PATROLLING_POLICY"]) == str:
        configuration.DRONE_PATROLLING_POLICY = cst.PatrollingProtocol[args["DRONE_PATROLLING_POLICY"]]
    else:
        configuration.DRONE_PATROLLING_POLICY = args["DRONE_PATROLLING_POLICY"]


if __name__ == "__main__":
    conf = Configuration()
    parser_cl_arguments(conf)
    main(conf)
