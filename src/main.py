
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


def __run_sweep():
    """ the place where to run simulations and experiments. """

    try:
        with wandb.init() as wandb_instance:
            wandb_config = wandb_instance.config

            learning = config.LEARNING_PARAMETERS

            learning[co.HyperParameters.LR.value] = wandb_config[co.HyperParameters.LR.value]
            learning[co.HyperParameters.DISCOUNT_FACTOR.value] = wandb_config[co.HyperParameters.DISCOUNT_FACTOR.value]
            learning[co.HyperParameters.SWAP_MODELS.value] = wandb_config[co.HyperParameters.SWAP_MODELS.value]

            learning[co.HyperParameters.MLP_HID1.value] = wandb_config[co.HyperParameters.MLP_HID1.value]
            learning[co.HyperParameters.MLP_HID2.value] = wandb_config[co.HyperParameters.MLP_HID2.value]
            learning[co.HyperParameters.MLP_HID3.value] = wandb_config[co.HyperParameters.MLP_HID3.value]

            config.IS_ALLOW_SELF_LOOP = wandb_config[co.HyperParameters.IS_SELF_LOOP.value]

            sim = PatrollingSimulator(learning=learning,
                                      drone_max_battery=wandb_config[co.HyperParameters.BATTERY.value],
                                      n_epochs=wandb_config[co.HyperParameters.N_EPOCHS.value],
                                      n_episodes=wandb_config[co.HyperParameters.N_EPISODES.value],
                                      episode_duration=wandb_config[co.HyperParameters.DURATION_EPISODE.value],
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
    sim = PatrollingSimulator()
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
                                  is_plot=False)
        sim.run()
    except:
        print(">> Could not solve problem!", locals())
        trace = traceback.format_exc()
        print("/n", trace)


if __name__ == "__main__":
    main_normal()

    # python -m src.main -pl 1
    # simulate_greedy_policies(setup01)
    # main_normal()
