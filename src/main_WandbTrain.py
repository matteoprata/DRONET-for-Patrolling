
import os
import sys
module_path = os.path.abspath(os.getcwd())
if module_path not in sys.path:
    sys.path.append(module_path)

import argparse
import src.constants as cst
from src.world_entities.simulator_patrolling import PatrollingSimulator
from src.config import Configuration, LearningHyperParameters
import wandb
import traceback
import sys


# SWEEP
DQN_LEARNING_HYPER_PARAMETERS = {
    # "set" is the chosen value
    LearningHyperParameters.REPLAY_MEMORY_DEPTH.value: {'values': [100000]},
    LearningHyperParameters.EPSILON_DECAY.value: {"values": [0.15, 0.01, 0.03, 0.05, 0.08]},  # best for 200 epochs, 0.01 and 0.08
    LearningHyperParameters.LEARNING_RATE.value:  {'min': 0.00001, 'max': 0.001},
    LearningHyperParameters.DISCOUNT_FACTOR.value: {'values': [1, 0.95, 0.8]},
    LearningHyperParameters.BATCH_SIZE.value: {'values': [32, 64]},

    LearningHyperParameters.SWAP_MODELS_EVERY_DECISION.value: {'min': 500, 'max': 8000},
    LearningHyperParameters.PERCENTAGE_SWAP.value: {'min': 0.005, 'max': 1.0},

    LearningHyperParameters.N_HIDDEN_1.value: {'values': [8]},
    LearningHyperParameters.N_HIDDEN_2.value: {'values': [0]},
    LearningHyperParameters.N_HIDDEN_3.value: {'values': [0]},
    LearningHyperParameters.N_HIDDEN_4.value: {'values': [0]},
    LearningHyperParameters.N_HIDDEN_5.value: {'values': [0]},

    # LearningHyperParameters.OPTIMIZER.value: {'values': ["adam"]},
    # LearningHyperParameters.LOSS.value: {'values': ["mse"]},
}


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

            cf.run_parameters_sanity_check()
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
