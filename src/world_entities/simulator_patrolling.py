import matplotlib.pyplot as plt
import pandas as pd
from src.patrolling.meta_patrolling import PrecomputedPolicy

import src.constants
from src.world_entities.environment import Environment
from src.world_entities.base_station import BaseStation
from src.world_entities.drone import Drone

from src.evaluation.MetricsLog import MetricsLog
from collections import defaultdict

from src.main_metrics import MetricsEvaluation
from src.evaluation.MetricsEvaluationValidation import plot_validation_stats
from src.utilities.utilities import euclidean_distance
from src.drawing import pp_draw
from src.config import Configuration
import src.constants as cst
from tqdm import tqdm
from src.utilities.utilities import current_date
import numpy as np
import time
import random
import torch
import wandb
import traceback

from src.RL.RLModule import RLModule
from src.constants import PrecomputedPatrollingProtocol as pol2

import src.patrolling.meta_patrolling as patrol_base


class PatrollingSimulator:

    def __init__(self, config: Configuration):

        self.cf = config

        # setting randomness
        random.seed(self.cf.SEED)
        np.random.seed(self.cf.SEED)

        self.tolerance_factor = config.TARGETS_TOLERANCE_FIXED
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

        self.selected_drone = None
        self.current_date = current_date()

        self.metrics = None  # init later
        self.rl_module: RLModule = None
        self.run_state = cst.EpisodeType.TRAIN

        # create the world entites
        self.__set_randomness()
        self.__create_world_entities()
        self.__setup_plotting()

        # create directory of the simulation
        # make_path(self.directory_simulation() + "-")

        # RL stuff
        self.prev_loss = np.inf
        self.prev_reward = -np.inf
        self.worst_epoch_counter = 0

        self.epoch_loss = []
        self.epoch_cumrew = []

        self.epoch = 0
        self.metrics_logs = defaultdict(list)
        self.rmean = 0
        self.rstd = 1
        self.rmin = -50
        self.rmax = 50

        self.do_break_episode = False
        self.policy = None

        self.ok_to_start = self.policy not in [pol2.CYCLE]  # in cyclic policies, the protocol should start when all the drones reached their initial point

        if self.cf.IS_AD_HOC_SCENARIO:
            print("WARNING!: ad hoc scenario running, input number of targets and UVS ignored. "
                  "To use input, edit IS_AD_HOC_SCENARIO variable in the config.")

    # ---- # BOUNDS and CONSTANTS # ---- #

    def episode_duration_seconds(self):
        """ Last second of the Simulation. """
        return self.cf.EPISODE_DURATION * self.cf.SIM_TS_DURATION

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
            drone_speed = 0 if self.drone_mobility == src.constants.OnlinePatrollingProtocol.FREE else self.drone_speed_meters_sec
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

    @staticmethod
    def point_on_line_segment(A, B, x):
        x1, y1 = A
        x2, y2 = B
        # Calculate the distance between A and B
        distance_AB = np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
        # Calculate the parameter t
        t = x / distance_AB
        # Calculate the coordinates of the point on AB at distance x from A
        point_on_AB = np.array([x1 + t * (x2 - x1), y1 + t * (y2 - y1)])
        return point_on_AB

    @staticmethod
    def generate_nodes_on_path(coordinates, N):
        if len(coordinates) < 2 or N < 2:
            raise ValueError("Invalid input: At least two coordinates and N >= 2 are required.")

        path_length = 0
        for i in range(len(coordinates) - 1):
            x1, y1 = coordinates[i]
            x2, y2 = coordinates[i + 1]
            path_length += np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
        x1, y1 = coordinates[0]
        x2, y2 = coordinates[-1]
        path_length += np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)

        distance_between_nodes = path_length / N
        current_distance = 0
        path_coordinates = [coordinates[0]]
        assigned_target = []
        for i in range(len(coordinates) - 1):
            x1, y1 = coordinates[i]
            x2, y2 = coordinates[i + 1]
            segment_length = np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)

            while current_distance + distance_between_nodes < segment_length:
                ratio = (current_distance + distance_between_nodes) / segment_length
                x = x1 + ratio * (x2 - x1)
                y = y1 + ratio * (y2 - y1)
                # print(ratio, (x, y))
                assigned_target.append((i+1) % len(coordinates))
                path_coordinates.append((x, y))
                current_distance += distance_between_nodes

            current_distance -= segment_length
        assigned_target = [1] + assigned_target

        # # Extract x and y coordinates for plotting
        # x_original, y_original = zip(*coordinates)
        # x_nodes, y_nodes = zip(*path_coordinates)
        #
        # # Plotting
        # plt.figure(figsize=(8, 6))
        # plt.plot(x_original, y_original, marker='o', linestyle='-', color='blue', label='Original Coordinates')
        # plt.scatter(x_nodes, y_nodes, color='red', label='Generated Nodes')
        #
        # # Annotate the indices of the next node in the path
        # for i, index in enumerate(assigned_target):
        #     plt.annotate(index, (x_nodes[i], y_nodes[i]), textcoords="offset points", xytext=(0, 5), ha='center')
        #
        # # Adding labels and legend
        # plt.title('Nodes along the Path with Associated Indices')
        # plt.xlabel('X-axis')
        # plt.ylabel('Y-axis')
        # plt.legend()
        #
        # # Show the plot
        # plt.show()

        return path_coordinates, assigned_target

    def assign_initial_pos(self):
        # prev_t0, prev_t1 = None, None
        # sim_prev = 0
        #
        # tsp_cost = self.policy.route_info["cost"]
        # tsp_targets = self.policy.cyclic_to_visit[0]
        #
        # intra_drone_distance = tsp_cost // len(self.environment.drones)
        # for it in range(len(tsp_targets)-1):
        #     t1, t2 = self.environment.targets[tsp_targets[it]], self.environment.targets[tsp_targets[it+1]]
        #     dist = euclidean_distance(t1.coords, t2.coords)
        #     n_drones = int(dist / intra_drone_distance) + 1
        #     for ind in range(n_drones):
        #         cocco = self.point_on_line_segment(t1.coords, t2.coords, intra_drone_distance)

        if self.cf.NO_BATTERY:
            if issubclass(type(self.policy), PrecomputedPolicy):
                if type(self.policy) is pol2.CYCLE.value:
                    # tsp_cost = self.policy.route_info["cost"]
                    # intra_drone_distance = tsp_cost // len(self.environment.drones)
                    tsp_ids =  [self.environment.targets[t].identifier for t in self.policy.cyclic_to_visit[0]]
                    coors = [self.environment.targets[t].coords for t in self.policy.cyclic_to_visit[0]]
                    print("-----")
                    if len(self.environment.drones) > 1:
                        c1, c2 = self.generate_nodes_on_path(coors, len(self.environment.drones))
                    else:
                        c1, c2 = [self.environment.targets[self.policy.cyclic_to_visit[0][0]].coords], [0]

                    di = len(self.environment.drones) - len(c1)
                    if di > 0:
                        rand_tar_id = np.random.randint(0, len(self.environment.drones), size=di)
                        for r in rand_tar_id:
                            c1.append(self.environment.targets[r].coords)
                            c2.append(r)

                    print("C2", c2, tsp_ids)
                    for dr in self.environment.drones:
                        COORS = c1[dr.identifier]
                        dr.coords = COORS
                        dr.visited_targets_coordinates[0] = COORS
                        dr.previous_coords = COORS

                        t_id = tsp_ids[c2[dr.identifier]-1]
                        dr.prev_target = self.environment.targets[t_id]

                        idx = self.policy.cyclic_to_visit[dr.identifier].index(dr.prev_target.identifier)
                        # print(dr.prev_target.identifier, idx, self.policy.cyclic_to_visit[dr.identifier])
                        self.policy.cyclic_to_visit[dr.identifier] = self.policy.cyclic_to_visit[dr.identifier][idx:] + self.policy.cyclic_to_visit[dr.identifier][:idx]
                        print(dr.identifier, self.policy.cyclic_to_visit[dr.identifier])
                else:
                    # partition e infocom
                    for dr in self.environment.drones:
                        # print("QUI ", self.policy.cyclic_to_visit[dr.identifier])
                        index_tar = 1 if type(self.policy) is pol2.INFOCOM.value else 0
                        # print(self.policy.cyclic_to_visit)
                        # exit()
                        tar_0 = self.environment.targets[self.policy.cyclic_to_visit[dr.identifier][0] + index_tar]
                        dr.coords = tar_0.coords
                        dr.visited_targets_coordinates[0] = tar_0.coords
                        dr.previous_coords = tar_0.coords
                        dr.prev_target = tar_0
            else:
                for dr in self.environment.drones:
                    # print("QUI ", self.policy.cyclic_to_visit[dr.identifier])
                    rand_tar = np.random.randint(0, len(self.environment.targets), 1)[0]
                    tar_0 = self.environment.targets[rand_tar]
                    dr.coords = tar_0.coords
                    dr.visited_targets_coordinates[0] = tar_0.coords
                    dr.previous_coords = tar_0.coords
                    dr.prev_target = tar_0

    def run_episodes(self, episodes_ids, typ: cst.EpisodeType, protocol):
        print("Doing", typ)
        self.run_state = typ
        for epi in tqdm(episodes_ids, desc=typ.value, leave=False, disable=self.cf.IS_HIDE_PROGRESS_BAR):
            self.environment.reset_simulation()

            targets = self.environment.targets_dataset[typ][epi]  # [Target1, Target2, ...]
            self.environment.spawn_targets(targets)

            self.cur_step = 0
            self.metricsV2 = MetricsLog(self)
            self.metricsV2.set_episode_details(epi, protocol.name, self.cf.DRONES_NUMBER, self.cf.TARGETS_NUMBER,
                                               self.cf.DRONE_SPEED, self.cf.TARGETS_POSITION_SCENARIO.name)

            self.prepare_drones_routes(self.cf.DRONE_PATROLLING_POLICY)  # assign a schedule to the drone
            # TODO better pisition? this allows the drone to depart form the first target
            self.assign_initial_pos()

            # a = self.policy.next_visit(0)
            # print(a)
            # exit()
            for cur_step in tqdm(range(self.cf.EPISODE_DURATION), desc='step', leave=False, disable=self.cf.IS_HIDE_PROGRESS_BAR):
                if self.do_break_episode:
                    self.do_break_episode = False
                    break

                self.cur_step = cur_step

                for drone in self.environment.drones:
                    drone.move(protocol=protocol)

                if self.cf.SAVE_PLOT or self.cf.PLOT_SIM:
                    self.__plot(self.cur_step, self.episode_duration)

            if typ in [cst.EpisodeType.VAL]:
                self.metrics_logs[protocol].append(self.metricsV2.to_json())

    # ----> RUNNING THE SIMULATION <----

    def run_training_loop(self):

        if self.cf.DRONE_PATROLLING_POLICY == cst.OnlinePatrollingProtocol.RL_DECISION_TRAIN:
            for epo in tqdm(range(self.cf.N_EPOCHS), desc='epoch', disable=self.cf.IS_HIDE_PROGRESS_BAR):
                self.epoch = epo

                if self.early_stop_check(self.epoch_loss, epo):
                    break
                self.on_epoch_start(epo)

                self.training()
                self.validation(epo)

                self.save_best_model(self.epoch_cumrew)
        else:
            print("Unhandled protocol for training.")
            exit(1)

    def prepare_drones_routes(self, policy_val):
        if type(policy_val) == cst.PrecomputedPatrollingProtocol:
            self.policy = policy_val.value(self.environment.drones, self.environment.targets)

    def run_testing_loop(self):
        if self.cf.DRONE_PATROLLING_POLICY in [e for e in list(cst.OnlinePatrollingProtocol) + list(cst.PrecomputedPatrollingProtocol) if e != cst.OnlinePatrollingProtocol.RL_DECISION_TRAIN]:
            self.run_episodes([0], typ=cst.EpisodeType.TEST, protocol=self.cf.DRONE_PATROLLING_POLICY)
            print("Saving stats file...")
            self.metricsV2.save_metrics()
            self.patrolling_report()
        else:
            print("Unhandled protocol for testing.")
            exit(1)
    #

    def on_epoch_start(self, epo):
        if epo > 0:
            self.rmean = np.mean(self.epoch_cumrew)
            self.rstd = np.std(self.epoch_cumrew)
            self.rmin = np.min(self.epoch_cumrew)
            self.rmin = np.max(self.epoch_cumrew)

        self.epoch_loss = []
        self.epoch_cumrew = []

    def training(self):
        # sum loss and reward during the epoch
        epi = self.cf.N_EPISODES_TRAIN_UPPER if self.cf.N_EPISODES_TRAIN_UPPER is not None else self.cf.N_EPISODES_TRAIN
        episodes_id = self.rstate_sample_batch_training.randint(low=0, high=epi, size=self.cf.N_EPISODES_TRAIN)
        try:
            self.run_episodes(episodes_id, typ=cst.EpisodeType.TRAIN, protocol=self.cf.DRONE_PATROLLING_POLICY)
        except:
            print("There was a problem running this episode. I will ignore it.")
            print(traceback.format_exc())
        self.on_train_epoch_end(is_log=self.cf.IS_WANDB)

    def validation(self, epo):
        """ Validation of the RL protocol"""
        if epo % self.cf.VALIDATE_EVERY == 0:
            val_algos = [self.cf.DRONE_PATROLLING_POLICY]

            if epo > 0:
                self.metrics_logs[self.cf.DRONE_PATROLLING_POLICY] = list()  # reset validation metrics
            else:
                # compute validation only once, for all the algorithms except RL
                val_algos += self.cf.VALIDATION_ALGORITHMS

            for al in val_algos:
                print("Validating algorithm:", al)
                self.run_episodes(range(self.cf.N_EPISODES_VAL), typ=cst.EpisodeType.VAL, protocol=al)
            self.on_validation_epoch_end(tot_episodes=self.cf.N_EPISODES_VAL, is_log=self.cf.IS_WANDB)

    def on_train_epoch_end(self, is_log=True):
        if is_log:
            dic = {
                "loss": np.sum(self.epoch_loss),
                "explore-prob": self.rl_module.dqn_mod.explore_prob,
                self.cf.FUNCTION_TO_OPTIMIZE["name"]: np.sum(self.epoch_cumrew),
            }
            self.cf.WANDB_INSTANCE.log(dic)

    def on_validation_epoch_end(self, tot_episodes, is_log=True):
        if is_log:
            algos = [self.cf.DRONE_PATROLLING_POLICY] + self.cf.VALIDATION_ALGORITHMS
            metrics_to_log = plot_validation_stats(tot_episodes, algos, self.metrics_logs, self.cf.VALIDATION_DEP_VARS)
            self.cf.WANDB_INSTANCE.log(metrics_to_log)

    def on_test_epoch_end(self, tot_episodes, is_log=True):
        self.on_validation_epoch_end(tot_episodes, is_log)

    def early_stop_check(self, loss_vector, epoch):
        if epoch <= self.cf.N_EPOCHS * .5:
            # otherwise at the beginning the loss is 0
            return False

        cur_loss = np.sum(loss_vector)
        if self.prev_loss <= cur_loss:
            # NO IMPROVEMENT
            print("Loss increased {} times, is {} was {}.".format(self.worst_epoch_counter, cur_loss, self.prev_loss))
            self.worst_epoch_counter += 1
        else:
            self.worst_epoch_counter = 0
            self.prev_loss = cur_loss

        halt = self.worst_epoch_counter >= self.cf.LOSS_EARLY_STOP_EPOCHS
        if halt:
            print("Early stopped")
        return halt

    def save_best_model(self, reward_vector):
        # https://docs.wandb.ai/guides/models/walkthrough#4.-use-a-model-version

        if not self.cf.IS_WANDB:
            return

        this_model_name = "model-{}.pt".format(self.cf.WANDB_INSTANCE.id)
        this_model_path_root = "data/model"
        this_model_path_full = "{}/{}".format(this_model_path_root, this_model_name)
        torch.save(self.rl_module.dqn_mod.model, this_model_path_full)

        art = wandb.Artifact(this_model_name, type="model")
        art.add_file(this_model_path_full, this_model_name)

        reward = np.sum(reward_vector)
        if reward > self.prev_reward:
            self.cf.WANDB_INSTANCE.log_artifact(art, aliases=["latest", "best"])
            self.prev_reward = reward
        else:
            self.cf.WANDB_INSTANCE.log_artifact(art)

    def is_final_state(self):
        """ Next transition will happen in the next seconds. """
        steps_to_end = self.cf.EPISODE_DURATION - self.cur_step
        steps_to_max = self.cf.max_time_distance() / self.cf.SIM_TS_DURATION
        is_not_enough = steps_to_max >= steps_to_end

        # print(steps_to_end, steps_to_max, self.cf.max_time_distance(), self.cf.SIM_TS_DURATION, is_not_enough)
        if is_not_enough:
            self.do_break_episode = True
            return True
        return False

    def patrolling_report(self):
        print()
        print("📌 REPORT:")
        print("🚁 ")

        met = MetricsEvaluation(sim_seed=self.sim_seed,
                                drone_mobility=self.drone_mobility,
                                n_drones=self.n_drones,
                                n_targets=self.n_targets,
                                drone_speed_meters_sec=self.drone_speed_meters_sec,
                                geographic_scenario=self.cf.TARGETS_POSITION_SCENARIO.name,
                                tolerance_scenario=self.cf.TARGETS_TOLERANCE_SCENARIO.name,
                                tolerance_fixed=self.cf.TARGETS_TOLERANCE_FIXED)

        plt.clf()
        plt.close()
        target_visits = []
        for t_id in met.targets_tolerance:
            if t_id != '0':
                x, times = met.AOI_func(t_id, is_absolute=True)
                target_visits.append(times)

        target_visits = np.array(target_visits).T  # TIME x TARGETS

        stats = []
        stats_names = ["c1: integral aoi", "c2: integral delay", "c3: max aoi", "c4: number violations", "c?: max delay"]
        stats_names = [p.upper() for p in stats_names]

        for t_id in range(target_visits.shape[1]):
            aois = target_visits[:, t_id]
            target_idle = self.environment.targets[t_id+1].maximum_tolerated_idleness
            stat_1 = met.AOI1_integral_func(aois)
            stat_2 = met.AOI2_max_func(aois)
            stat_3 = met.AOI3_max_delay_func(aois, target_idle)
            stat_4 = met.AOI4_n_violations_func(aois, target_idle)
            stat_5 = met.AOI5_violation_time_func(aois, target_idle)
            vec = [stat_1, stat_5, stat_2, stat_4, stat_3]
            stats.append(vec)
        stats = pd.DataFrame(stats, columns=stats_names)

        print()
        print(stats)
        print()
        print(stats.describe())
