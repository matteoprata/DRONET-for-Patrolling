

import argparse
import src.constants as cst
from src.simulation.simulator_patrolling import PatrollingSimulator
from src.config import Configuration, LearningHyperParameters, DQN_LEARNING_HYPER_PARAMETERS
import wandb
import traceback
import sys
import numpy as np
import time


def run_sweep(configuration: Configuration):
    """ Run sweeps to monitor models performances live. """

    print(DQN_LEARNING_HYPER_PARAMETERS)

    sweep_id = wandb.sweep(
        project=configuration.PROJECT_NAME,
        sweep={
            'name':       configuration.conf_description(),
            'method':     configuration.HYPER_PARAM_SEARCH_MODE,
            'metric':     configuration.FUNCTION_TO_OPTIMIZE,
            'parameters': DQN_LEARNING_HYPER_PARAMETERS
        }
    )
    wandb.agent(sweep_id, function=run)


# REMINDER: this function must take NO INPUT PARAMETERS
def run():
    try:
        cf = Configuration()
        parser_cl_arguments(cf)

        with wandb.init() as wandb_instance:

            cf.IS_WANDB = True
            cf.WANDB_INSTANCE = wandb_instance

            for param in LearningHyperParameters:
                cf.DQN_PARAMETERS[param] = wandb_instance.config[param.value]

            sim = PatrollingSimulator(cf)
            sim.run_training()
            print("DONE")

    except Exception as e:
        # exit gracefully, so wandb logs the problem
        print(traceback.print_exc(), file=sys.stderr)
        exit(1)


def parser_cl_arguments(configuration: Configuration):
    """ Parses the arguments for the command line. """

    # python -m src.main_WandbTrain -seed 10 -nd 1 -nt 10 -pl 0
    args_li = [
            ('-seed', 'SEED', int),
            ('-tol', 'TARGETS_TOLERANCE', float),
            ('-nt', 'TARGETS_NUMBER', int),
            ('-nd', 'DRONES_NUMBER', int),
            ('-spe', 'DRONE_SPEED', float),
            ('-bat', 'DRONE_MAX_ENERGY', float),
            ('-ne', 'N_EPOCHS', int),
            ('-net', 'N_EPISODES_TRAIN', int),
            ('-neb', 'N_EPISODES_VAL', int),
            ('-nes', 'N_EPISODES_TEST', int),
            ('-edu', 'EPISODE_DURATION', int),
            ('-pl', 'PLOT_SIM', int)
    ]

    parser = argparse.ArgumentParser(description='Patrolling Simulator arguments:')

    for nick, name, typ in args_li:
        parser.add_argument(nick, "--" + name, default=getattr(configuration, name), type=typ)

    args = vars(parser.parse_args())
    # parsing arguments from cli

    print("Setting parameters...")

    for nick, name, typ in args_li:
        if name == "PLOT_SIM":
            setattr(configuration, name, bool(args[name]))
        else:
            setattr(configuration, name, args[name])

    configuration.DRONE_PATROLLING_POLICY = cst.PatrollingProtocol.RL_DECISION_TRAIN
    configuration.N_EPOCHS = 50
    configuration.N_EPISODES_TRAIN = 20
    configuration.N_EPISODES_VAL = 0
    configuration.N_EPISODES_TEST = 0


if __name__ == "__main__":
    conf = Configuration()
    parser_cl_arguments(conf)
    run_sweep(conf)
