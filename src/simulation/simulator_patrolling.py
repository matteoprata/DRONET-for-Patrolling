import src.constants
from src.world_entities.environment import Environment
from src.world_entities.base_station import BaseStation
from src.world_entities.drone import Drone
from src.patrolling.metrics import Metrics

from src.evaluation.MetricsLog import MetricsLog
from collections import defaultdict

from src.evaluation.MetricsEvaluationValidation import plot_validation_stats, data_matrix_multiple_episodes
from src.utilities.utilities import current_date, euclidean_distance
from src.drawing import pp_draw
from src.config import Configuration
import src.constants as cst
import copy
from tqdm import tqdm

import numpy as np
import time
import random


from src.patrolling.RLModule2 import RLModule


class PatrollingSimulator:

    def __init__(self, config: Configuration):

        self.cf = config

        # setting randomness
        random.seed(self.cf.SEED)
        np.random.seed(self.cf.SEED)

        self.tolerance_factor = config.TARGETS_TOLERANCE
        self.log_state = config.LOG_STATE
        self.penalty_on_bs_expiration = config.PENALTY_ON_BS_EXPIRATION

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

        self.cur_step = 0
        self.cur_step_total = 0

        self.selected_drone = None
        self.current_date = current_date()

        self.metrics = None  # init later
        self.rl_module = None
        self.run_state = cst.EpisodeType.TRAIN

        # create the world entites
        self.__set_randomness()
        self.__create_world_entities()
        self.__setup_plotting()

        # create directory of the simulation
        # make_path(self.directory_simulation() + "-")

    # ---- # BOUNDS and CONSTANTS # ---- #

    def episode_duration_seconds(self):
        """ Last second of the Simulation. """
        return self.cf.EPISODE_DURATION * self.cf.SIM_TS_DURATION

    def current_second(self, next=0, cur_second_tot=False):
        """ The current second of simulation, since the beginning. """
        return (self.cur_step_total if cur_second_tot else self.cur_step) * self.cf.SIM_TS_DURATION + next * self.ts_duration_sec

    def max_distance(self):
        """ Maximum distance in the area. """
        return (self.environment.width**2 + self.environment.height**2)**0.5

    def max_travel_time(self):
        """ Time required to travel to maximum distance. """
        return self.max_distance() / self.drone_speed_meters_sec

    def name(self):
        return self.cf.conf_description()

    def directory_simulation(self):
        pass

    # ---- # KEYS and MOUSE CLICKS # ---- #

    def detect_key_pressed(self, key_pressed):
        """ Moves the drones freely. """

        if key_pressed in ['a', 'A']:  # decrease angle
            self.selected_drone.angle -= self.cf.DRONE_ANGLE_INCREMENT
            self.selected_drone.angle = self.selected_drone.angle % 360

        elif key_pressed in ['d', 'D']:  # increase angle
            self.selected_drone.angle += self.cf.DRONE_ANGLE_INCREMENT
            self.selected_drone.angle = self.selected_drone.angle % 360

        elif key_pressed in ['w', 'W']:  # increase speed
            self.selected_drone.speed += self.cf.DRONE_SPEED_INCREMENT

        elif key_pressed in ['s', 'S']:  # decrease speed
            self.selected_drone.speed -= self.cf.DRONE_SPEED_INCREMENT

    def detect_drone_click(self, position):
        """ Handles drones selection in the simulation. """
        click_coords_to_map = (self.environment.width / self.cf.DRAW_SIZE * position[0], self.environment.height / self.cf.DRAW_SIZE * (self.cf.DRAW_SIZE - position[1]))
        entities_distance = [euclidean_distance(drone.coords, click_coords_to_map) for drone in self.environment.drones]
        clicked_drone = self.environment.drones[np.argmin(entities_distance)]  # potentially clicked drone

        TOLERATED_CLICK_DISTANCE = 40

        closest_drone_coords = clicked_drone.coords
        dron_coords_to_screen = (closest_drone_coords[0] * self.cf.DRAW_SIZE / self.environment.width, self.cf.DRAW_SIZE - (closest_drone_coords[1] * self.cf.DRAW_SIZE / self.environment.width))

        if euclidean_distance(dron_coords_to_screen, position) < TOLERATED_CLICK_DISTANCE:
            # DRONE WAS CLICKED HANDLE NOW
            self.on_drone_click(clicked_drone)

    def on_drone_click(self, clicked_drone):
        """ Defines the behaviour following a click on a drone. """
        self.selected_drone = clicked_drone

    # ---- # OTHER # ---- #

    def __setup_plotting(self):
        if self.is_plot or self.cf.SAVE_PLOT:
            self.draw_manager = pp_draw.PathPlanningDrawer(self.environment, self, borders=True, config=self.cf)

    def __set_randomness(self):
        """ Set the random generators. """
        self.rnd_tolerance = np.random.RandomState(self.sim_seed)
        self.rnd_env = np.random.RandomState(self.sim_seed)
        self.rnd_event = np.random.RandomState(self.sim_seed)
        self.rnd_sample_replay = np.random.RandomState(self.sim_seed)
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
            drone_path = [self.cf.DRONE_COORDS]
            drone_speed = 0 if self.drone_mobility == src.constants.PatrollingProtocol.FREE else self.drone_speed_meters_sec
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
                          patrolling_protocol=self.drone_mobility)
            drones.append(drone)

        self.selected_drone = drones[0]
        self.environment.add_drones(drones)

        self.rl_module = RLModule(self)

    def __plot(self, cur_step, max_steps):
        """ Plot the simulation """

        if cur_step % self.cf.SKIP_SIM_STEP != 0:
            return

        if self.cf.WAIT_SIM_STEP > 0:
            time.sleep(self.cf.WAIT_SIM_STEP)

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
        self.draw_manager.update(save=self.cf.SAVE_PLOT, filename=self.name() + str(cur_step) + ".png")

    def print_sim_info(self):
        print("simulation starting", self.name())
        print()

    def reset_episode(self):
        pass

    def run_episodes(self, episodes_ids, typ: cst.EpisodeType, protocol):
        print("Doing", typ)
        for epi in tqdm(episodes_ids, desc=typ.value, leave=False, disable=self.cf.IS_HIDE_PROGRESS_BAR):
            self.environment.reset_simulation()

            self.metricsV2 = MetricsLog(self)
            self.metricsV2.set_episode_details(epi, protocol.name, self.cf.DRONES_NUMBER, self.cf.TARGETS_NUMBER, self.cf.DRONE_SPEED, self.cf.TARGETS_TOLERANCE)

            self.run_state = typ
            targets = self.environment.targets_dataset[typ][epi]  # [Target1, Target2, ...]
            self.environment.spawn_targets(targets)

            self.cur_step = 0

            for cur_step in tqdm(range(self.cf.EPISODE_DURATION), desc='step', leave=False, disable=self.cf.IS_HIDE_PROGRESS_BAR):
                self.cur_step = cur_step

                for drone in self.environment.drones:
                    drone.move(protocol=protocol)

                if self.cf.SAVE_PLOT or self.cf.PLOT_SIM:
                    self.__plot(self.cur_step, self.episode_duration)

                self.cur_step_total += 1
            self.metrics_logs[protocol].append(self.metricsV2.to_json())

    # ----> RUNNING THE SIMULATION <----

    def run_training(self):
        for _ in tqdm(range(self.cf.N_EPOCHS), desc='epoch', disable=self.cf.IS_HIDE_PROGRESS_BAR):

            self.epoch_loss = []
            self.epoch_cumrew = []
            self.metrics_logs = defaultdict(list)

            # sum loss and reward during the epoch
            episodes_id = self.rstate_sample_batch_training.permutation(self.cf.N_EPISODES_TRAIN)
            self.run_episodes(episodes_id, typ=cst.EpisodeType.TRAIN, protocol=self.cf.DRONE_PATROLLING_POLICY)
            self.on_train_epoch_end(is_log=self.cf.IS_WANDB)

            # measure reward and stats to average over the episodes = seeds
            self.metrics_logs = defaultdict(list)
            for al in [self.cf.DRONE_PATROLLING_POLICY] + self.cf.VALIDATION_ALGORITHMS:
                print("Validating algorithm:", al)
                self.run_episodes(range(self.cf.N_EPISODES_VAL), typ=cst.EpisodeType.VAL, protocol=al)
            self.on_validation_epoch_end(is_log=self.cf.IS_WANDB)

        # # ON THE BEST MODEL measure reward and stats to average over the episodes = seeds
        # self.metrics_logs = defaultdict(list)
        # for al in [self.cf.DRONE_PATROLLING_POLICY] + self.cf.VALIDATION_ALGORITHMS:
        #     print("Testing algorithm:", al)
        #     self.run_episodes(range(self.cf.N_EPISODES_TEST), typ=cst.EpisodeType.TEST, protocol=al)
        # self.on_test_epoch_end(is_log=self.cf.IS_WANDB)

    def run_testing(self):
        self.run_episodes(range(self.cf.N_EPISODES_TEST), typ=cst.EpisodeType.TEST, protocol=self.cf.DRONE_PATROLLING_POLICY)

        print("Saving stats file...")
        self.metricsV2.save_metrics()

    #

    def on_train_epoch_end(self, is_log=True):
        if is_log:
            dic = {
                "loss": np.sum(self.epoch_loss),
                self.cf.FUNCTION_TO_OPTIMIZE["name"]: np.sum(self.epoch_cumrew)
            }
            self.cf.WANDB_INSTANCE.log(dic)

    def on_validation_epoch_end(self, is_log=True):
        if is_log:
            algos = [self.cf.DRONE_PATROLLING_POLICY] + self.cf.VALIDATION_ALGORITHMS
            dic_plt = {v.name: plot_validation_stats(self.cf.N_EPISODES_VAL, algos, self.metrics_logs, v) for v in self.cf.VALIDATION_VARS}
            self.cf.WANDB_INSTANCE.log(dic_plt)

    def on_test_epoch_end(self, is_log=True):
        if is_log:
            algos = [self.cf.DRONE_PATROLLING_POLICY] + self.cf.VALIDATION_ALGORITHMS
            dic_plt = {v.name: plot_validation_stats(self.cf.N_EPISODES_TEST, algos, self.metrics_logs, v) for v in self.cf.VALIDATION_VARS}
            self.cf.WANDB_INSTANCE.log(dic_plt)
