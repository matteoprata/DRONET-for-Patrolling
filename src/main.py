
from src.simulation.simulator_patrolling import PatrollingSimulator
import src.utilities.config as config
import argparse
import wandb

parser = argparse.ArgumentParser(description='Run experiments of patrolling.')

parser.add_argument('-mode', '--mode', type=int, default=0)  # 0: train, 1:test

# -- learning params, n_hidden_naurons
parser.add_argument('-ip', '--is_pretrained', type=int, default=config.LEARNING_PARAMETERS['is_pretrained'])
parser.add_argument('-bs', '--batch_size', type=int, default=config.LEARNING_PARAMETERS['batch_size'])
parser.add_argument('-lr', '--learning_rate', type=float, default=config.LEARNING_PARAMETERS['learning_rate'])
parser.add_argument('-df', '--discount_factor', type=float, default=config.LEARNING_PARAMETERS['discount_factor'])
parser.add_argument('-rm', '--replay_memory_depth', type=int, default=config.LEARNING_PARAMETERS['replay_memory_depth'])
parser.add_argument('-sw', '--swap_models_every_decision', type=int, default=config.LEARNING_PARAMETERS['swap_models_every_decision'])
parser.add_argument('-sl', '--is_allow_self_loop', type=int, default=config.IS_ALLOW_SELF_LOOP)

# -- logging
parser.add_argument('-pl', '--is_plot', type=int, default=0)
parser.add_argument('-lo', '--log_state', type=float, default=-1)

# -- battery, speed, number of targets
parser.add_argument('-sp', '--drone_speed', type=int, default=config.DRONE_SPEED)
parser.add_argument('-bat', '--battery', type=int, default=config.DRONE_MAX_ENERGY)
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

# reward and stuff
parser.add_argument('-exp', '--is_expired_target_condition', type=int, default=config.IS_EXPIRED_TARGET_CONDITION)
parser.add_argument('-nta', '--n_targets', type=int, default=config.N_TARGETS)
parser.add_argument('-ndr', '--n_drones', type=int, default=config.N_DRONES)

# END PARAMETERS DEFINITION

args = parser.parse_args()

learning = config.LEARNING_PARAMETERS
for arg in vars(args):
    if arg in learning:
        learning[arg] = getattr(args, arg)


def main():
    """ the place where to run simulations and experiments. """
    if args.mode == 0:
        with wandb.init(project="uavsimulator_patrolling") as wandb_instance:
            wandb_config = wandb_instance.config

            learning["learning_rate"] = wandb_config["learning_rate"]
            learning["discount_factor"] = wandb_config["discount_factor"]
            learning["swap_models_every_decision"] = wandb_config["swap_models_every_decision"]

            learning["n_hidden_neurons_lv1"] = wandb_config["n_hidden_neurons_lv1"]
            learning["n_hidden_neurons_lv2"] = wandb_config["n_hidden_neurons_lv2"]
            learning["n_hidden_neurons_lv3"] = wandb_config["n_hidden_neurons_lv3"]

            sim = PatrollingSimulator(learning=learning,
                                      is_plot=bool(wandb_config["is_plot"]),
                                      drone_speed=args.drone_speed,
                                      n_targets=wandb_config['n_targets'],
                                      n_drones=wandb_config['n_drones'],
                                      drone_max_battery=wandb_config['battery'],
                                      log_state=args.log_state,
                                      n_epochs=wandb_config['n_epochs'],
                                      n_episodes=wandb_config['n_episodes'],
                                      episode_duration=wandb_config['episode_duration'],
                                      penalty_on_bs_expiration=args.penalty,
                                      sim_seed=args.seed,
                                      is_expired_target_condition=wandb_config["is_expired_target_condition"],
                                      wandb=wandb_instance)
            sim.run()

    elif args.mode == 1:
        seeds_range = 1  # for testing
        modes = (2, 2)
        name = "seeds{}-ndr{}-nta{}-modes{}".format(seeds_range, args.n_drones, args.n_targets, modes)

        for mod in range(modes[0], modes[1]+1):
            mode = config.Mobility(mod)
            sim = PatrollingSimulator(drone_mobility=mode,
                                      name=name,
                                      drone_speed=args.drone_speed,
                                      n_targets=args.n_targets,
                                      n_drones=args.n_drones,
                                      drone_max_battery=args.battery,
                                      log_state=args.log_state,
                                      is_plot=bool(args.is_plot),
                                      penalty_on_bs_expiration=args.penalty,
                                      sim_seed=1,
                                      n_epochs=args.n_epochs,
                                      n_episodes=args.n_episodes,
                                      episode_duration=args.episode_duration,
                                      is_expired_target_condition=args.is_expired_target_condition,
                                      wandb=None)
            sim.run()
        sim.metrics.join_metrics(sim.directory_simulation())


if __name__ == "__main__":
    main()

