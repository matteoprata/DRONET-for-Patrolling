import src.utilities.constants
from src.world_entities.environment import Environment
from src.world_entities.base_station import BaseStation
from src.world_entities.drone import Drone
from src.patrolling.metrics import Metrics
from data.archive.plotting import Plotting

from src.evaluation.MetricsLog import MetricsLog

from src.utilities.utilities import current_date, euclidean_distance, make_path
import src.utilities.config as config
from src.drawing import pp_draw

from tqdm import tqdm

import numpy as np
import time
import os


class PatrollingSimulator:

    def __init__(self,
                 sim_description=config.SIM_DESCRIPTION,
                 sim_seed=config.SIM_SEED,
                 ts_duration_sec=config.SIM_TS_DURATION,
                 sim_duration_ts=config.SIM_DURATION,
                 n_drones=config.N_DRONES,
                 n_base_stations=config.N_BASE_STATIONS,
                 env_width_meters=config.ENV_WIDTH,
                 env_height_meters=config.ENV_HEIGHT,
                 drone_speed=config.DRONE_SPEED,
                 drone_max_battery=config.DRONE_MAX_ENERGY,
                 drone_max_buffer=config.DRONE_MAX_BUFFER_SIZE,
                 drone_com_range_meters=config.DRONE_COM_RANGE,
                 drone_sen_range_meters=config.DRONE_SENSING_RANGE,
                 drone_radar_range_meters=config.DRONE_RADAR_RADIUS,
                 drone_coo=config.DRONE_COORDS,
                 bs_com_range_meters=config.BASE_STATION_COM_RANGE,
                 bs_coords=config.BASE_STATION_COORDS,
                 n_obstacles=config.N_OBSTACLES,
                 n_grid_cells=config.N_GRID_CELLS,
                 n_targets=config.N_TARGETS,
                 tolerance_factor=config.TOLERANCE_FACTOR,
                 drone_mobility=config.DRONE_MOBILITY,
                 learning=config.LEARNING_PARAMETERS,

                 log_state=config.LOG_STATE,
                 penalty_on_bs_expiration=config.PENALTY_ON_BS_EXPIRATION,
                 n_epochs=config.N_EPOCHS,
                 n_episodes=config.N_EPISODES,
                 n_episodes_validation=config.N_EPISODES_VAL,
                 episode_duration=config.EPISODE_DURATION,
                 is_plot=config.PLOT_SIM,
                 wandb=None
                 ):

        self.tolerance_factor = tolerance_factor
        self.log_state=log_state
        self.penalty_on_bs_expiration=penalty_on_bs_expiration
        self.n_epochs=n_epochs
        self.n_episodes=n_episodes
        self.n_episodes_validation = n_episodes_validation
        self.episode_duration=episode_duration
        self.is_plot = is_plot

        # HYPER TUNING
        self.wandb = wandb

        self.learning = learning
        self.sim_peculiarity = sim_description

        self.cur_step = 0
        self.cur_step_total = 0

        self.sim_seed = sim_seed
        self.ts_duration_sec = ts_duration_sec
        self.sim_duration_ts = sim_duration_ts
        self.env_width_meters, self.env_height_meters = env_width_meters, env_height_meters
        self.n_drones = n_drones
        self.n_targets = n_targets
        self.n_obstacles = n_obstacles
        self.grid_cell_size = 0 if n_grid_cells <= 0 else int(self.env_width_meters / n_grid_cells)

        # if this coo is not none, then the drones are self driven
        self.drone_coo = drone_coo
        self.selected_drone = None

        self.drone_mobility = drone_mobility
        self.drone_speed_meters_sec = drone_speed
        self.drone_max_battery = drone_max_battery
        self.drone_max_buffer = drone_max_buffer
        self.drone_com_range_meters = drone_com_range_meters
        self.drone_sen_range_meters = drone_sen_range_meters
        self.drone_radar_range_meters = drone_radar_range_meters
        self.n_base_stations = n_base_stations
        self.bs_com_range_meters = bs_com_range_meters
        self.bs_coords = bs_coords
        self.current_date = current_date()

        self.metrics = None  # init later
        # create the world entites
        self.__set_randomness()
        self.__create_world_entities()
        self.__setup_plotting()

        # create directory of the simulation
        make_path(self.directory_simulation() + "-")

        self.reset_episode_val = False
        self.is_validation = False
        # self.plotting = Plotting(self.name())

    # ---- # BOUNDS and CONSTANTS # ---- #

    def duration_seconds(self):
        """ Last second of the Simulation. """
        return self.sim_duration_ts * self.ts_duration_sec

    def current_second(self, next=0, cur_second_tot=False):
        """ The current second of simulation, since the beginning. """
        return (self.cur_step_total if cur_second_tot else self.cur_step) * self.ts_duration_sec + next * self.ts_duration_sec

    def max_distance(self):
        """ Maximum distance in the area. """
        return (self.environment.width**2 + self.environment.height**2)**0.5

    def max_travel_time(self):
        """ Time required to travel to maximum distance. """
        return self.max_distance() / self.drone_speed_meters_sec

    def name(self):
        return "{}-seed{}-ndrones{}-mode{}".format(self.sim_peculiarity, self.sim_seed, self.n_drones, self.drone_mobility.value)

    def directory_simulation(self):
        return config.RL_DATA + self.name() + "/"

    # ---- # KEYS and MOUSE CLICKS # ---- #

    def detect_key_pressed(self, key_pressed):
        """ Moves the drones freely. """

        if key_pressed in ['a', 'A']:  # decrease angle
            self.selected_drone.angle -= config.DRONE_ANGLE_INCREMENT
            self.selected_drone.angle = self.selected_drone.angle % 360

        elif key_pressed in ['d', 'D']:  # increase angle
            self.selected_drone.angle += config.DRONE_ANGLE_INCREMENT
            self.selected_drone.angle = self.selected_drone.angle % 360

        elif key_pressed in ['w', 'W']:  # increase speed
            self.selected_drone.speed += config.DRONE_SPEED_INCREMENT

        elif key_pressed in ['s', 'S']:  # decrease speed
            self.selected_drone.speed -= config.DRONE_SPEED_INCREMENT

    def detect_drone_click(self, position):
        """ Handles drones selection in the simulation. """
        click_coords_to_map = (self.environment.width/config.DRAW_SIZE*position[0], self.environment.height/config.DRAW_SIZE*(config.DRAW_SIZE-position[1]))
        entities_distance = [euclidean_distance(drone.coords, click_coords_to_map) for drone in self.environment.drones]
        clicked_drone = self.environment.drones[np.argmin(entities_distance)] # potentially clicked drone

        TOLERATED_CLICK_DISTANCE = 40

        closest_drone_coords = clicked_drone.coords
        dron_coords_to_screen = (closest_drone_coords[0]*config.DRAW_SIZE/self.environment.width, config.DRAW_SIZE - (closest_drone_coords[1]*config.DRAW_SIZE/self.environment.width))

        if euclidean_distance(dron_coords_to_screen, position) < TOLERATED_CLICK_DISTANCE:
            # DRONE WAS CLICKED HANDLE NOW
            self.on_drone_click(clicked_drone)

    def on_drone_click(self, clicked_drone):
        """ Defines the behaviour following a click on a drone. """
        self.selected_drone = clicked_drone

    # ---- # OTHER # ---- #

    def __setup_plotting(self):
        if self.is_plot or config.SAVE_PLOT:
            self.draw_manager = pp_draw.PathPlanningDrawer(self.environment, self, borders=True)

    def __set_randomness(self):
        """ Set the random generators. """
        self.rnd_tolerance = np.random.RandomState(self.sim_seed)
        self.rnd_env = np.random.RandomState(self.sim_seed)
        self.rnd_event = np.random.RandomState(self.sim_seed)
        self.rnd_explore = np.random.RandomState(self.sim_seed)
        self.rstate_sample_batch_training = np.random.RandomState(self.sim_seed)

    def __create_world_entities(self):
        """ Creates the world entities. """

        self.environment = Environment(self.env_width_meters, self.env_height_meters, self)

        base_stations = []
        for i in range(self.n_base_stations):
            base_stations.append(BaseStation(i, self.bs_coords, self.bs_com_range_meters, self))
        self.environment.add_base_station(base_stations)

        self.environment.spawn_obstacles()
        self.environment.spawn_targets()

        drones = []
        for i in range(self.n_drones):
            drone_path = [self.drone_coo]
            drone_speed = 0 if self.drone_mobility == src.utilities.constants.Mobility.FREE else self.drone_speed_meters_sec
            drone = Drone(identifier=i,
                          path=drone_path,
                          bs=base_stations[0],
                          angle=0,
                          speed=drone_speed,
                          com_range=self.drone_com_range_meters,
                          sensing_range=self.drone_sen_range_meters,
                          radar_range=self.drone_radar_range_meters,
                          max_buffer=self.drone_max_buffer,
                          max_battery=self.drone_max_battery,
                          simulator=self,
                          mobility=self.drone_mobility)
            drones.append(drone)

        self.selected_drone = drones[0]
        self.environment.add_drones(drones)

        self.metrics = Metrics(self)
        self.metrics.N_ACTIONS = drones[0].rl_module.N_ACTIONS
        self.metrics.N_FEATURES = drones[0].rl_module.N_FEATURES

        self.metricsV2 = MetricsLog(self)
        self.previous_metricsV2 = self.metricsV2  # the metrics at the previous epoch

    def __plot(self, cur_step, max_steps):
        """ Plot the simulation """

        if cur_step % config.SKIP_SIM_STEP != 0:
            return

        if config.WAIT_SIM_STEP > 0:
            time.sleep(config.WAIT_SIM_STEP)

        self.draw_manager.grid_plot()
        self.draw_manager.borders_plot()

        for drone in self.environment.drones:
            self.draw_manager.draw_drone(drone, cur_step)

        for base_station in self.environment.base_stations:
            self.draw_manager.draw_depot(base_station)

        for event in self.environment.get_valid_events(cur_step):
            self.draw_manager.draw_event(event)

        self.draw_manager.draw_simulation_info(cur_step=cur_step, max_steps=max_steps)
        self.draw_manager.draw_obstacles()
        self.draw_manager.draw_targets()
        self.draw_manager.update(save=config.SAVE_PLOT, filename=self.name() + str(cur_step) + ".png")

    def print_sim_info(self):
        print("simulation starting", self.name())
        print(self.learning)
        print()

    def reset_episode(self):
        self.reset_episode_val = not self.reset_episode_val

    def episode_core(self, just_setup, IS_HIDE_PRO_BARS):
        for cur_step in tqdm(range(self.episode_duration), desc='step', leave=False, disable=IS_HIDE_PRO_BARS):

            if self.reset_episode_val:
                self.reset_episode()
                break

            self.cur_step = cur_step

            if just_setup:
                return

            for drone in self.environment.drones:
                # self.environment.detect_collision(drone)
                drone.move()

            if config.SAVE_PLOT or self.is_plot:
                self.__plot(self.cur_step, self.episode_duration)

            self.cur_step_total += 1

    def run(self, just_setup=False):
        """ The method starts the simulation. """
        self.print_sim_info()

        IS_HIDE_PRO_BARS = False
        for epoch in tqdm(range(self.n_epochs), desc='epoch', disable=IS_HIDE_PRO_BARS):
            self.is_validation = False
            self.i_epoch = epoch
            episodes_perm = self.rstate_sample_batch_training.permutation(self.n_episodes)  # at each epoch you see the same episodes but shuffled
            for episode in tqdm(range(len(episodes_perm)), desc='episodes_train', leave=False, disable=IS_HIDE_PRO_BARS):

                self.i_episode = episode
                ie = episodes_perm[episode]
                self.environment.reset_simulation()

                targets = self.environment.targets_dataset[ie]  # [Target1, Target2]
                self.environment.spawn_targets(targets)

                self.cur_step = 0
                self.episode_core(just_setup, IS_HIDE_PRO_BARS)

            for episode in tqdm(range(self.n_episodes_validation), desc='episodes_val', leave=False, disable=IS_HIDE_PRO_BARS):
                self.is_validation = True
                self.i_episode = episode
                self.environment.reset_simulation()
                targets = self.environment.targets_dataset[-self.n_episodes_validation:][episode]  # [Target1, Target2]
                self.environment.spawn_targets(targets)

                self.cur_step = 0
                self.episode_core(just_setup, IS_HIDE_PRO_BARS)

    def checkout(self, epoch, is_last_epoch, do=False):
        """ Print metrics save stuff at the end of an episode. """
        # self.metricsV2.metrics_report_single_point()
        # exit()
        # self.metricsV2.print_all_metrics()
        # self.metricsV2.save_metrics()
        # exit()

        if do:
            try:
                if self.learning["is_pretrained"]:
                    self.metrics.save_dataframe()
                else:
                    self.metrics.save_dataframe_light()
            except:
                print("Couldn't save data from step", self.cur_step_total)

            try:
                if self.learning["is_pretrained"]:
                    self.plotting.plot_patrolling_performance()
                else:
                    self.plotting.plot_learning_performance()
            except:
                print("Couldn't plot from step", self.cur_step_total)

        SAVE_EPOCH_EVERY = 25

        if epoch % SAVE_EPOCH_EVERY == 0 or is_last_epoch:
            model_file_name = "model-epoch{}.h5".format(epoch)
            path = config.RL_DATA + self.name() + "/" + model_file_name if self.wandb is None else os.path.join(self.wandb.dir, model_file_name)
            self.environment.drones[0].rl_module.DQN.save_model(path)

