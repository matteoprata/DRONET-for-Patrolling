
import os
import sys
module_path = os.path.abspath(os.getcwd())
if module_path not in sys.path:
    sys.path.append(module_path)

import argparse
import src.constants as cst
from src.world_entities.simulator_patrolling import PatrollingSimulator
from src.config import Configuration, LearningHyperParameters, DQN_LEARNING_HYPER_PARAMETERS
import wandb
import traceback
import sys


def run_sweep(configuration: Configuration):
    """ Run sweeps to monitor models performances live. """

    # print(DQN_LEARNING_HYPER_PARAMETERS)

    sweep_id = wandb.sweep(
        project=configuration.PROJECT_NAME,
        sweep={
            'command':    ["${env}", "python3", "${program}", "${args}"],
            'program':    "src/main_WandbTrain.py",
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
            sim.run_training_loop()
            print("DONE")

    except Exception as e:
        # exit gracefully, so wandb logs the problem
        print(traceback.print_exc(), file=sys.stderr)
        exit(1)


def parser_cl_arguments(configuration: Configuration):
    """ Parses the arguments for the command line. """

    configuration.DRONE_PATROLLING_POLICY = cst.PatrollingProtocol.RL_DECISION_TRAIN

    configuration.N_EPOCHS = 200
    configuration.N_EPISODES_TRAIN = 30
    configuration.N_EPISODES_VAL = 20
    configuration.N_EPISODES_TEST = 0

    configuration.DRONES_NUMBER = 1
    configuration.TARGETS_NUMBER = 5

    args_li_learning = [
        ('-bas', 'batch_size', int),
        ('-dis', 'discount_factor', float),
        ('-ede', 'epsilon_decay', float),
        ('-lr', 'learning_rate', float),
        ('-nh1', 'n_hidden_neurons_lv1', int),
        ('-nh2', 'n_hidden_neurons_lv2', int),
        ('-nh3', 'n_hidden_neurons_lv3', int),
        ('-nh4', 'n_hidden_neurons_lv4', int),
        ('-nh5', 'n_hidden_neurons_lv5', int),
        ('-rmd', 'replay_memory_depth', int),
        ('-swa', 'swap_models_every_decision', int),
    ]

    parser = argparse.ArgumentParser(description='Patrolling Simulator arguments:')

    for nick, name, typ in args_li_learning:
        parser.add_argument(nick, "--" + name, type=typ)

    args = vars(parser.parse_args())
    # parsing arguments from cli

    print("Setting parameters...")

    for nick, name, typ in args_li_learning:
        configuration.DQN_PARAMETERS[name] = args[name]


if __name__ == "__main__":
    conf = Configuration()
    parser_cl_arguments(conf)
    run_sweep(conf)
