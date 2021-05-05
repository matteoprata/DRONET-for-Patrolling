
from src.simulation.simulator_patrolling import PatrollingSimulator
import src.utilities.config as config
import sys

import argparse

parser = argparse.ArgumentParser(description='Run experiments of patrolling.')
parser.add_argument('-ip', '--is_pretrained', type=bool)
parser.add_argument('-bs', '--batch_size', type=int)
parser.add_argument('-lr', '--learning_rate', type=float)
parser.add_argument('-df', '--discount_factor', type=float)
parser.add_argument('-be', '--beta', type=float)
parser.add_argument('-rm', '--replay_memory_depth', type=int)
parser.add_argument('-sw', '--swap_models_every_decision', type=int)

parser.add_argument('-de', '--description', type=str, default="")
parser.add_argument('-du', '--duration', type=int, default=24000*24*15)

parser.add_argument('-po', '--positive', type=bool)
parser.add_argument('-re', '--relative', type=bool)

args = parser.parse_args()

duration = args.duration
description = args.description
learning = config.LEARNING_PARAMETERS

for arg in vars(args):
    val = getattr(args, arg)
    if val is not None and arg in learning:
        learning[arg] = val

config.POSITIVE = args.positive
config.RELATIVE = args.relative


def main():
    """ the place where to run simulations and experiments. """
    sim = PatrollingSimulator(sim_description=description,
                              sim_duration_ts=duration,
                              learning=learning)
    sim.run()


if __name__ == "__main__":
    main()