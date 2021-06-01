
from src.simulation.simulator_patrolling import PatrollingSimulator
import src.utilities.config as config
import argparse
import wandb

parser = argparse.ArgumentParser(description='Run experiments of patrolling.')

parser.add_argument('-wandb', '--is_wandb', type=int, default=1)
parser.add_argument('-de', '--description', type=str, default=config.SIM_DESCRIPTION)

# -- learning params
parser.add_argument('-ip', '--is_pretrained', type=int, default=config.LEARNING_PARAMETERS['is_pretrained'])
parser.add_argument('-bs', '--batch_size', type=int, default=config.LEARNING_PARAMETERS['batch_size'])
parser.add_argument('-lr', '--learning_rate', type=float, default=config.LEARNING_PARAMETERS['learning_rate'])
parser.add_argument('-df', '--discount_factor', type=float, default=config.LEARNING_PARAMETERS['discount_factor'])
parser.add_argument('-rm', '--replay_memory_depth', type=int, default=config.LEARNING_PARAMETERS['replay_memory_depth'])
parser.add_argument('-sw', '--swap_models_every_decision', type=int, default=config.LEARNING_PARAMETERS['swap_models_every_decision'])

# -- logging
parser.add_argument('-pl', '--plotting', type=int, default=0)
parser.add_argument('-lo', '--log_state', type=float, default=-1)

# -- battery, speed, number of targets
parser.add_argument('-sp', '--drone_speed', type=int, default=config.DRONE_SPEED)
parser.add_argument('-bat', '--drone_max_energy', type=int, default=config.DRONE_MAX_ENERGY)
parser.add_argument('-tar', '--n_targets', type=int, default=config.N_TARGETS)
parser.add_argument('-pen', '--penalty', type=int, default=config.PENALTY_ON_BS_EXPIRATION)
parser.add_argument('-seed', '--seed', type=int, default=config.SIM_SEED)

# epochs episodes
parser.add_argument('-epo', '--n_epochs', type=int, default=config.N_EPOCHS)
parser.add_argument('-epi', '--n_episodes', type=int, default=config.N_EPISODES)
parser.add_argument('-edu', '--episode_duration', type=int, default=config.EPISODE_DURATION)

# END PARAMETERS DEFINITION

args = parser.parse_args()

learning = config.LEARNING_PARAMETERS
for arg in vars(args):
    if arg in learning:
        learning[arg] = getattr(args, arg)


def main():
    """ the place where to run simulations and experiments. """
    if bool(args.is_wandb):
        with wandb.init() as wandb_instance:
            wandb_config = wandb_instance.config

            learning["learning_rate"] = wandb_config["learning_rate"]
            learning["discount_factor"] = wandb_config["discount_factor"]
            learning["swap_models_every_decision"] = wandb_config["swap_models_every_decision"]

            sim = PatrollingSimulator(learning=learning,
                                      sim_description=args.description,
                                      n_targets=args.n_targets,
                                      drone_speed=args.drone_speed,
                                      drone_max_battery=args.drone_max_energy,
                                      log_state=args.log_state,
                                      is_plot=bool(args.plotting),
                                      n_epochs=wandb_config['n_epochs'],
                                      n_episodes=wandb_config['n_episodes'],
                                      episode_duration=wandb_config['episode_duration'],
                                      penalty_on_bs_expiration=args.penalty,
                                      sim_seed=args.seed,

                                      wandb=wandb_instance)
            sim.run()
    else:
        sim = PatrollingSimulator(learning=learning,
                                  sim_description=args.description,
                                  n_targets=args.n_targets,
                                  drone_speed=args.drone_speed,
                                  drone_max_battery=args.drone_max_energy,
                                  log_state=args.log_state,
                                  is_plot=bool(args.plotting),
                                  n_epochs=args.n_epochs,
                                  n_episodes=args.n_episodes,
                                  episode_duration=args.episode_duration,
                                  penalty_on_bs_expiration=args.penalty,
                                  sim_seed=args.seed)
        sim.run()



if __name__ == "__main__":
    main()


