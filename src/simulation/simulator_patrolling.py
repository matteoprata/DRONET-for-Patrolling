import src.utilities.constants
from src.world_entities.environment import Environment
from src.world_entities.base_station import BaseStation
from src.world_entities.drone import Drone
from src.patrolling.metrics import Metrics
from data.archive.plotting import Plotting

from src.evaluation.MetricsLog import MetricsLog

from src.utilities.utilities import current_date, euclidean_distance, make_path
from src.drawing import pp_draw
from src.utilities.config import Configuration
from tqdm import tqdm

import numpy as np
import time
import os


class PatrollingSimulator:

    def __init__(self, config: Configuration):

        self.config = config
        self.tolerance_factor = config.TARGETS_TOLERANCE
        self.log_state = config.LOG_STATE
        self.penalty_on_bs_expiration = config.PENALTY_ON_BS_EXPIRATION
        self.n_epochs = config.N_EPOCHS
        self.n_episodes = config.N_EPISODES
        self.n_episodes_validation = config.N_EPISODES_VAL
        self.episode_duration = config.EPISODE_DURATION
        self.is_plot = config.PLOT_SIM

        self.sim_seed = config.SEED
        self.ts_duration_sec = config.SIM_TS_DURATION
        self.sim_duration_ts = config.EPISODE_DURATION
        self.env_width_meters, self.env_height_meters = config.ENV_WIDTH, config.ENV_HEIGHT
        self.n_drones = config.DRONES_NUMBER
        self.n_targets = config.TARGETS_NUMBER
        self.n_obstacles = config.N_OBSTACLES

        self.drone_mobility = config.DRONE_PATROLLING_POLICY
        self.drone_speed_meters_sec = config.DRONE_SPEED
        self.drone_max_battery = config.DRONE_MAX_ENERGY
        self.drone_max_buffer = config.DRONE_MAX_BUFFER_SIZE
        self.drone_com_range_meters = config.DRONE_COM_RANGE
        self.drone_sen_range_meters = config.DRONE_SENSING_RANGE
        self.drone_radar_range_meters = config.DRONE_RADAR_RADIUS
        self.n_base_stations = config.N_BASE_STATIONS
        self.bs_com_range_meters = config.DRONE_COM_RANGE
        self.bs_coords = config.BASE_STATION_COORDS

        self.grid_cell_size = 0 if config.N_GRID_CELLS <= 0 else int(config.ENV_WIDTH / config.N_GRID_CELLS)

        self.wandb = config.IS_WANDB
        self.learning = config.LEARNING_PARAMETERS

        self.cur_step = 0
        self.cur_step_total = 0

        self.selected_drone = None
        self.current_date = current_date()

        self.metrics = None  # init later
        # create the world entites
        self.__set_randomness()
        self.__create_world_entities()
        self.__setup_plotting()

        # create directory of the simulation
        # make_path(self.directory_simulation() + "-")

        self.reset_episode_val = False
        self.is_validation = False

    # ---- # BOUNDS and CONSTANTS # ---- #

    def duration_seconds(self):
        """ Last second of the Simulation. """
        return self.config.EPISODE_DURATION * self.config.SIM_TS_DURATION

    def current_second(self, next=0, cur_second_tot=False):
        """ The current second of simulation, since the beginning. """
        return (self.cur_step_total if cur_second_tot else self.cur_step) * self.config.SIM_TS_DURATION + next * self.ts_duration_sec

    def max_distance(self):
        """ Maximum distance in the area. """
        return (self.environment.width**2 + self.environment.height**2)**0.5

    def max_travel_time(self):
        """ Time required to travel to maximum distance. """
        return self.max_distance() / self.drone_speed_meters_sec

    def name(self):
        return self.config.conf_description()

    def directory_simulation(self):
        pass

    # ---- # KEYS and MOUSE CLICKS # ---- #

    def detect_key_pressed(self, key_pressed):
        """ Moves the drones freely. """

        if key_pressed in ['a', 'A']:  # decrease angle
            self.selected_drone.angle -= self.config.DRONE_ANGLE_INCREMENT
            self.selected_drone.angle = self.selected_drone.angle % 360

        elif key_pressed in ['d', 'D']:  # increase angle
            self.selected_drone.angle += self.config.DRONE_ANGLE_INCREMENT
            self.selected_drone.angle = self.selected_drone.angle % 360

        elif key_pressed in ['w', 'W']:  # increase speed
            self.selected_drone.speed += self.config.DRONE_SPEED_INCREMENT

        elif key_pressed in ['s', 'S']:  # decrease speed
            self.selected_drone.speed -= self.config.DRONE_SPEED_INCREMENT

    def detect_drone_click(self, position):
        """ Handles drones selection in the simulation. """
        click_coords_to_map = (self.environment.width/self.config.DRAW_SIZE*position[0], self.environment.height/self.config.DRAW_SIZE*(self.config.DRAW_SIZE-position[1]))
        entities_distance = [euclidean_distance(drone.coords, click_coords_to_map) for drone in self.environment.drones]
        clicked_drone = self.environment.drones[np.argmin(entities_distance)] # potentially clicked drone

        TOLERATED_CLICK_DISTANCE = 40

        closest_drone_coords = clicked_drone.coords
        dron_coords_to_screen = (closest_drone_coords[0]*self.config.DRAW_SIZE/self.environment.width, self.config.DRAW_SIZE - (closest_drone_coords[1]*self.config.DRAW_SIZE/self.environment.width))

        if euclidean_distance(dron_coords_to_screen, position) < TOLERATED_CLICK_DISTANCE:
            # DRONE WAS CLICKED HANDLE NOW
            self.on_drone_click(clicked_drone)

    def on_drone_click(self, clicked_drone):
        """ Defines the behaviour following a click on a drone. """
        self.selected_drone = clicked_drone

    # ---- # OTHER # ---- #

    def __setup_plotting(self):
        if self.is_plot or self.config.SAVE_PLOT:
            self.draw_manager = pp_draw.PathPlanningDrawer(self.environment, self, borders=True, config=self.config)

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
            drone_path = [self.config.DRONE_COORDS]
            drone_speed = 0 if self.drone_mobility == src.utilities.constants.PatrollingProtocol.FREE else self.drone_speed_meters_sec
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
        # self.metrics.N_ACTIONS = drones[0].rl_module.N_ACTIONS
        # self.metrics.N_FEATURES = drones[0].rl_module.N_FEATURES

        self.metricsV2 = MetricsLog(self)
        self.previous_metricsV2 = self.metricsV2  # the metrics at the previous epoch

    def __plot(self, cur_step, max_steps):
        """ Plot the simulation """

        if cur_step % self.config.SKIP_SIM_STEP != 0:
            return

        if self.config.WAIT_SIM_STEP > 0:
            time.sleep(self.config.WAIT_SIM_STEP)

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
        self.draw_manager.update(save=self.config.SAVE_PLOT, filename=self.name() + str(cur_step) + ".png")

    def print_sim_info(self):
        print("simulation starting", self.name())
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

            if self.config.SAVE_PLOT or self.is_plot:
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

        is_one_shot = self.n_episodes == 1 and self.n_epochs == 1 and self.n_episodes_validation == 0

        if is_one_shot:
            print("One shot execution, saving stats file...")
            self.metricsV2.save_metrics()

