
from src.simulation.simulator_patrolling import PatrollingSimulator
import src.utilities.config as config
import argparse
import wandb
import traceback
from multiprocessing import Pool
import sys

from src.utilities.constants import IndependentVariable as indv
from src.utilities.constants import DependentVariable as depv
from src.utilities.constants import Mobility as pol
import src.utilities.constants as co
from src.utilities.utilities import initializer
from src.wandb_configs.wandb_config1 import sweep_configuration
from src.simulation_setup import setup01
from src.simulation_setup import setup02

"""
WARNING: When running sweeps, params are in the YAML otherwise be careful and look in config and main.
         Parameters are taken from the config file if they are not overwritten from the command line.
"""

parser = argparse.ArgumentParser(description='Run experiments of patrolling.')

parser.add_argument('-sweep', '--is_sweep', type=bool, default=config.IS_SWEEP)
parser.add_argument('-de', '--description', type=str, default=config.SIM_DESCRIPTION)

# -- learning params, n_hidden_naurons
parser.add_argument('-ip', '--is_pretrained', type=int, default=config.LEARNING_PARAMETERS['is_pretrained'])
parser.add_argument('-bs', '--batch_size', type=int, default=config.LEARNING_PARAMETERS['batch_size'])
parser.add_argument('-lr', '--learning_rate', type=float, default=config.LEARNING_PARAMETERS['learning_rate'])
parser.add_argument('-df', '--discount_factor', type=float, default=config.LEARNING_PARAMETERS['discount_factor'])
parser.add_argument('-rm', '--replay_memory_depth', type=int, default=config.LEARNING_PARAMETERS['replay_memory_depth'])
parser.add_argument('-sw', '--swap_models_every_decision', type=int, default=config.LEARNING_PARAMETERS['swap_models_every_decision'])
parser.add_argument('-sl', '--is_allow_self_loop', type=int, default=config.IS_ALLOW_SELF_LOOP)

# -- logging
parser.add_argument('-pl', '--plotting', type=int, default=0)
parser.add_argument('-lo', '--log_state', type=float, default=-1)

# -- battery, speed, number of targets
parser.add_argument('-sp', '--drone_speed', type=int, default=config.DRONE_SPEED)
parser.add_argument('-bat', '--battery', type=int, default=config.DRONE_MAX_ENERGY)
parser.add_argument('-tar', '--n_targets', type=int, default=config.N_TARGETS)
parser.add_argument('-pen', '--penalty', type=int, default=config.PENALTY_ON_BS_EXPIRATION)
parser.add_argument('-seed', '--seed', type=int, default=config.SIM_SEED)

# epochs episodes
parser.add_argument('-epo', '--n_epochs', type=int, default=config.N_EPOCHS)
parser.add_argument('-epi', '--n_episodes', type=int, default=config.N_EPISODES)
parser.add_argument('-edu', '--episode_duration', type=int, default=config.EPISODE_DURATION)

# network
parser.add_argument('-hn1', '--n_hidden_neurons_lv1', type=int, default=config.LEARNING_PARAMETERS['n_hidden_neurons_lv1'])
parser.add_argument('-hn2', '--n_hidden_neurons_lv2', type=int, default=config.LEARNING_PARAMETERS['n_hidden_neurons_lv2'])
parser.add_argument('-hn3', '--n_hidden_neurons_lv3', type=int, default=config.LEARNING_PARAMETERS['n_hidden_neurons_lv3'])


# END PARAMETERS DEFINITION

args = parser.parse_args()

learning = config.LEARNING_PARAMETERS
for arg in vars(args):
    if arg in learning:
        learning[arg] = getattr(args, arg)


def __run_sweep():
    """ the place where to run simulations and experiments. """

    try:
        with wandb.init() as wandb_instance:
            wandb_config = wandb_instance.config

            learning[co.HyperParameters.LR.value] = wandb_config[co.HyperParameters.LR.value]
            learning[co.HyperParameters.DISCOUNT_FACTOR.value] = wandb_config[co.HyperParameters.DISCOUNT_FACTOR.value]
            learning[co.HyperParameters.SWAP_MODELS.value] = wandb_config[co.HyperParameters.SWAP_MODELS.value]

            learning[co.HyperParameters.MLP_HID1.value] = wandb_config[co.HyperParameters.MLP_HID1.value]
            learning[co.HyperParameters.MLP_HID2.value] = wandb_config[co.HyperParameters.MLP_HID2.value]
            learning[co.HyperParameters.MLP_HID3.value] = wandb_config[co.HyperParameters.MLP_HID3.value]

            config.IS_ALLOW_SELF_LOOP = wandb_config[co.HyperParameters.IS_SELF_LOOP.value]

            sim = PatrollingSimulator(learning=learning,
                                      sim_description=args.description,
                                      n_targets=args.n_targets,
                                      drone_speed=args.drone_speed,
                                      drone_max_battery=wandb_config[co.HyperParameters.BATTERY.value],
                                      log_state=args.log_state,
                                      is_plot=bool(args.plotting),
                                      n_epochs=wandb_config[co.HyperParameters.N_EPOCHS.value],
                                      n_episodes=wandb_config[co.HyperParameters.N_EPISODES.value],
                                      episode_duration=wandb_config[co.HyperParameters.DURATION_EPISODE.value],
                                      penalty_on_bs_expiration=args.penalty,
                                      sim_seed=args.seed,
                                      drone_mobility=co.Mobility.RL_DECISION,
                                      wandb=wandb_instance)
            sim.run()

    except Exception as e:
        # exit gracefully, so wandb logs the problem
        print(traceback.print_exc(), file=sys.stderr)
        exit(1)

def main_sweep():
    sweep_id = wandb.sweep(sweep=sweep_configuration, project=co.PROJECT_NAME)
    wandb.agent(sweep_id, function=__run_sweep)


def main_normal():
    sim = PatrollingSimulator(learning=learning,
                              sim_description=args.description,
                              n_targets=args.n_targets,
                              drone_speed=args.drone_speed,
                              drone_max_battery=args.battery,
                              log_state=args.log_state,
                              is_plot=bool(args.plotting),
                              n_epochs=args.n_epochs,
                              n_episodes=args.n_episodes,
                              episode_duration=args.episode_duration,
                              penalty_on_bs_expiration=args.penalty,
                              sim_seed=args.seed)
    sim.run()


def main_batch_experiments(setup):
    # 1. Declare independent variables and their domain
    # 2. Declare what independent variable varies at this execution and what stays fixed

    stp = setup

    processes = []
    indv_fixed_original = {k: stp.indv_fixed[k] for k in stp.indv_fixed}
    for a in stp.comp_dims[indv.ALGORITHM]:
        for s in stp.comp_dims[indv.SEED]:
            for x_var_k in stp.indv_vary:
                X_var = stp.indv_vary[x_var_k]
                for x in X_var:
                    stp.indv_fixed[x_var_k] = x

                    # declare processes
                    process = [a, s] + list(stp.indv_fixed.values())
                    processes.append(process)
                    stp.indv_fixed = {k: indv_fixed_original[k] for k in indv_fixed_original}  # reset the change

    if config.IS_PARALLEL:
        with Pool(initializer=initializer, processes=co.N_CORES) as pool:
            try:
                pool.starmap(__execute_parallel_simulations, processes)
            except KeyboardInterrupt:
                pool.terminate()
                pool.join()
        print("COMPLETED SUCCESSFULLY")
    else:
        for p in processes:
            __execute_parallel_simulations(*p)


def __execute_parallel_simulations(algorithm, seed, d_speed, d_number, t_number, t_factor):
    try:
        print("Executing:\n", locals())
        sim = PatrollingSimulator(tolerance_factor=t_factor,
                                  n_targets=t_number,
                                  drone_speed=d_speed,
                                  n_drones=d_number,
                                  drone_mobility=algorithm,
                                  sim_seed=seed,
                                  is_plot=bool(args.plotting))
        sim.run()
    except:
        print(">> Could not solve problem!", locals())
        trace = traceback.format_exc()
        print("/n", trace)


if __name__ == "__main__":
    if args.is_sweep:
        main_sweep()
    else:
        main_normal()

    # python -m src.main -pl 1
    # simulate_greedy_policies(setup01)
    # main_normal()
