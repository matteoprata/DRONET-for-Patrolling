
from src.utilities.utilities import log, is_segments_intersect, distance_point_segment, TraversedCells, euclidean_distance, clustering_kmeans
from src.world_entities.target import Target
from tqdm import tqdm
from src.utilities import tsp
from scipy.stats import truncnorm
import numpy as np
from collections import defaultdict
from src.evaluation.MetricsLog import MetricsLog
from src.constants import EpisodeType, TargetFamily, ToleranceScenario, PositionScenario
from src.utilities.utilities import generate_random_coordinates_in_circle


class ObstacleHandler:

    def __init__(self, simulator, width, height):
        self.height = height
        self.width = width
        self.simulator = simulator
        self.obstacles = []

    def spawn_obstacles(self, orthogonal_obs=False):
        """ Appends obstacles in the environment """
        for i in range(self.simulator.n_obstacles):
            startx, starty = self.simulator.rnd_env.randint(0, self.width), self.simulator.rnd_env.randint(0, self.height)
            length = self.simulator.rnd_env.randint(100, 300)
            angle = self.simulator.rnd_env.randint(0, 359) if not orthogonal_obs else self.simulator.rnd_env.choice([0, 90, 180, 270], 1)[0]

            endx, endy = startx + length * np.cos(np.radians(angle)), starty + length * np.sin(np.radians(angle))
            obstacle = (startx, starty, endx, endy)
            self.obstacles.append(obstacle)

    def detect_collision(self, drone):
        """ Detects a collision happened in the previous step. """
        if len(self.obstacles) > 0:

            # drone - obstacle collision
            distance_obstacles = self.distance_obstacles(drone)
            distance_travelled = drone.speed * self.simulator.ts_duration_sec
            critic_obstacles = np.array(self.obstacles)[distance_obstacles <= distance_travelled]  # obstacles that could be crossed in a time step

            for critical_segment in critic_obstacles:
                p1, p2 = (critical_segment[0], critical_segment[1]), (critical_segment[2], critical_segment[3])
                p3, p4 = drone.previous_coords, drone.coords

                if is_segments_intersect(p1, p2, p3, p4) or distance_point_segment(p1, p2, p4) < 1:
                    # COLLISION HAPPENED DO SOMETHING
                    self.__handle_collision(drone)
                    return

        # # drone - drone collision to fix
        # if drone.coords != self.sim.drone_coo:
        #     u1p1, u1p2 = drone.previous_coords, drone.coords
        #
        #     path_u1p1 = [0, u1p1[0], u1p1[1]]
        #     path_u1p2 = [self.sim.ts_duration_sec,
        #                  self.sim.ts_duration_sec * drone.speed + u1p1[0],
        #                  self.sim.ts_duration_sec * drone.speed + u1p1[1]]
        #
        #     for other_drone in self.drones:
        #         if drone.identifier > other_drone.identifier:
        #             u2p1, u2p2 = other_drone.previous_coords, other_drone.coords
        #
        #             path_u2p1 = [0, u2p1[0], u2p1[1]]
        #             path_u2p2 = [self.sim.ts_duration_sec,
        #                          self.sim.ts_duration_sec * drone.speed + u2p1[0],
        #                          self.sim.ts_duration_sec * drone.speed + u2p1[1]]
        #
        #             if is_segments_intersect(path_u1p1, path_u1p2, path_u2p1, path_u2p2) \
        #                     or euclidean_distance(u1p2, u2p2) < 1:
        #                 self.__handle_collision(drone)
        #                 self.__handle_collision(other_drone)
        #                 return

    def distance_obstacles(self, drone):
        """ Returns the distance for all the obstacles. """

        distance_obstacles = np.array([-1] * self.simulator.n_obstacles)
        p3 = np.array(drone.coords)
        for en, ob in enumerate(self.obstacles):
            p1, p2 = np.array([ob[0], ob[1]]), np.array([ob[2], ob[3]])
            distance_obstacles[en] = distance_point_segment(p1, p2, p3)
        return distance_obstacles

    @staticmethod
    def __handle_collision(drone):
        """ Takes countermeasure when drone collides. """
        drone.coords = drone.visited_targets_coordinates[0]


class Environment(ObstacleHandler):
    """ The environment is an entity that represents the area of interest."""

    def __init__(self, width, height, simulator):
        super().__init__(simulator, width, height)

        self.simulator = simulator
        self.width = width
        self.height = height
        self.drones = None
        self.base_stations = None

        # set events, set obstacles
        self.events: list = []  # even expired ones
        self.targets = []

        self.closest_target = []
        self.furthest_target = []
        self.targets_dataset = defaultdict(list)

    def add_drones(self, drones: list):
        """ add a list of drones in the env """
        # log("Added {} drones to the environment.".format(len(drones)))
        self.drones = drones

    def add_base_station(self, base_stations: list):
        """ add depots in the env """
        # log("Added {} base stations in the environment.".format(len(base_stations)))
        self.base_stations = base_stations

    def get_expired_events(self, current_ts):
        return [e for e in self.events if e.is_expired(current_ts)]

    def get_valid_events(self, current_ts):
        return [e for e in self.events if not e.is_expired(current_ts)]

    def get_current_cell(self, drone):
        drone_coods = drone.coords
        cell_index = TraversedCells.coord_to_cell(size_cell=self.simulator.grid_cell_size,
                                                  width_area=self.width,
                                                  x_pos=drone_coods[0],
                                                  y_pos=drone_coods[1])
        return cell_index

    def reset_simulation(self, is_end_epoch=True):
        """ Reset the scenario after every episode. """

        for drone in self.drones:
            drone.coords = drone.bs.coords
            # drone.rl_module.reset_MDP_episode()
            drone.reset()

        for tar in self.targets:
            tar.reset()

    def read_target_combinations(self):
        targets_json = {
            0: {"coords": (250, 250), "family": "BLUE", "idle": 300},
            1: {"coords": (300, 180), "family": "BLUE", "idle": 300},
            2: {"coords": (1250, 1150), "family": "BLUE", "idle": 300},
        }
        self.simulator.cf.TARGETS_NUMBER = len(targets_json)
        self.simulator.n_targets = len(targets_json)

        # self.simulator.cf.DRONES_NUMBER = len(targets_json)
        # self.simulator.n_drones = len(targets_json)

        epoch_targets = []
        for i in targets_json:
            t = Target(identifier=len(self.base_stations) + i,
                       coords=tuple(targets_json[i]["coords"]),
                       maximum_tolerated_idleness=targets_json[i]["idle"],
                       simulator=self.simulator,
                       family=TargetFamily[targets_json[i]["family"]])
            epoch_targets.append(t)
        self.targets_dataset[EpisodeType.TEST].append(epoch_targets)
        return self.targets_dataset

    def tolerances_function(self, target_coords, clusters_split):
        """ KEY: returns the thresholds for every target, based on the scenario"""
        scen = self.simulator.cf.TARGETS_TOLERANCE_SCENARIO

        if scen == ToleranceScenario.CONSTANT:
            return [self.simulator.cf.TARGETS_TOLERANCE_FIXED] * len(target_coords)  # +1 in case

        elif scen == ToleranceScenario.UNIFORM:
            LOW, HIGH = 0, 300
            return np.random.randint(LOW, HIGH, len(target_coords))

        elif scen == ToleranceScenario.GAUSSIAN:
            LOC, SCALE = 150, 250 * .25
            thetas = np.random.normal(LOC, SCALE, len(target_coords))
            thetas = [int(t) for t in thetas]
            return thetas

        elif scen == ToleranceScenario.CLUSTERED:
            N_CLUSTERS = len(clusters_split)
            thetas = []
            for i, clus in enumerate(clusters_split):
                LOW, HIGH = 0, 500
                THRESHOLDS = np.random.randint(LOW, HIGH, N_CLUSTERS)
                thetas += [THRESHOLDS[i] for _ in clus]

            return thetas

    def positions_function(self):
        """ KEY: Returns the coordinates of the targets. Both in a UNIFORM deployment and CLUSTERED. """

        scen = self.simulator.cf.TARGETS_POSITION_SCENARIO
        coordinates = []
        clusters_split = []  # contains the coordinates of the N clusters

        if scen == PositionScenario.UNIFORM:
            for i in range(self.simulator.n_targets):
                point_coords = [self.simulator.rnd_env.randint(0, self.width), self.simulator.rnd_env.randint(0, self.height)]
                coordinates.append(point_coords)
            print(self.simulator.n_targets)

        elif scen == PositionScenario.CLUSTERED:
            N_EPICENTERS = 5
            MAP_SIZE = self.height  # assumed squared
            POINT_PER_CLUSTER = self.simulator.n_targets // N_EPICENTERS
            SPARE = self.simulator.n_targets - (POINT_PER_CLUSTER * N_EPICENTERS)
            SPREAD = 20  # higher the tighter

            epicenters = np.random.rand(N_EPICENTERS, 2) * MAP_SIZE
            for epicenter in epicenters:
                cluster = epicenter + np.random.randn(POINT_PER_CLUSTER, 2) * MAP_SIZE / SPREAD
                cluster = np.clip(cluster, 0, MAP_SIZE)
                coordinates.extend(cluster)
                clusters_split.append(cluster)

            if SPARE > 0:
                cluster = epicenters[0] + np.random.randn(SPARE, 2) * MAP_SIZE / SPREAD
                cluster = np.clip(cluster, 0, MAP_SIZE)
                coordinates.extend(cluster)
                clusters_split[0].append(cluster)

        return coordinates, clusters_split

    def generate_target_combinations(self):
        """
        Assumption, file stores for each seed, up to 100 targets, up to 500 episodes
        :param seed:
        :return:
        """
        # Creates a dataset of targets to iterate over to
        # loading targets list
        # targets_fname = conf.TARGETS_FILE + "targets_s{}_nt{}_sp{}.json".format(seed, self.sim.n_targets, self.sim.drone_speed_meters_sec)

        targets_x_episode = defaultdict(list)
        # print("START: generating random episodes")

        # assert(self.sim.n_episodes <= MAX_N_EPISODES)
        epi = self.simulator.cf.N_EPISODES_TRAIN_UPPER if self.simulator.cf.N_EPISODES_TRAIN_UPPER is not None else self.simulator.cf.N_EPISODES_TRAIN
        epi_type = [(EpisodeType.TRAIN, epi),
                    (EpisodeType.VAL, self.simulator.cf.N_EPISODES_VAL),
                    (EpisodeType.TEST, self.simulator.cf.N_EPISODES_TEST)]

        for et, en in epi_type:
            for ep in range(en):
                coordinates, clusters_split = self.positions_function()
                thetas = self.tolerances_function(coordinates, clusters_split)

                for i in range(self.simulator.n_targets):  # set the threshold for the targets
                    idleness = thetas[i]  # self.simulator.cf.TARGETS_TOLERANCE_FIXED
                    targets_x_episode[ep].append((i, tuple(coordinates[i]), idleness))

        for et, en in epi_type:
            for ep in range(en):
                epoch_targets = []
                for t_id, t_coord, t_idleness in targets_x_episode[ep][:self.simulator.n_targets]:

                    t = Target(identifier=len(self.base_stations) + t_id,
                               coords=tuple(t_coord),
                               maximum_tolerated_idleness=t_idleness,
                               simulator=self.simulator)

                    epoch_targets.append(t)
                self.targets_dataset[et].append(epoch_targets)  # epoch: list of targets

        # print("done", self.targets_dataset)
        # print("TARGETS:", self.targets_dataset)
        return self.targets_dataset

    def spawn_targets(self, targets=None):

        self.targets = []
        if targets is None:
            if self.simulator.cf.IS_AD_HOC_SCENARIO:
                targets = self.read_target_combinations()[0]
            else:
                targets = self.generate_target_combinations()[0]

        # The base station is a target
        for i in range(self.simulator.n_base_stations):
            battery_seconds = self.simulator.drone_max_battery * self.simulator.ts_duration_sec
            self.targets.append(Target(i, self.base_stations[i].coords, battery_seconds, self.simulator))

        self.targets += targets

        # targets may be
        # if targets is not None:
        #     for j, (x, y, tol_del) in enumerate(targets):
        #         self.targets.append(Target(i+j+1, (x, y), tol_del, self.sim))

        # FOR each target set the furthest and closest target
        for tar1 in self.targets:
            distances = [euclidean_distance(tar1.coords, tar2.coords) for tar2 in self.targets]
            distances_min, distances_max = distances[:], distances[:]
            distances_min[tar1.identifier] = np.inf
            distances_max[tar1.identifier] = -np.inf

            tar1.furthest_target = self.targets[np.argmax(distances_max)]
            tar1.closest_target = self.targets[np.argmin(distances_min)]

    @staticmethod
    def get_truncated_normal(mean=0, sd=1, low=0, upp=10):
        return truncnorm((low - mean) / sd, (upp - mean) / sd, loc=mean, scale=sd)

    # def time_cost_hamiltonian(self, coords):
    #     for i in range(len(coords)-1):
    #         a = coords[i]
    #         a_prime = coords[i+1]
    #         time = euclidean_distance(a, a_prime) / self.drones[0].speed
    #

    def tsp_path_time(self, coordinates):
        coordinates_plus = [self.base_stations[0].coords] + coordinates
        tsp_ob = tsp.TSP()
        tsp_ob.read_data(coordinates_plus)
        two_opt = tsp.TwoOpt_solver(initial_tour='NN', iter_num=100)
        path, cost = tsp_ob.get_approx_solution(two_opt, star_node=0)
        return cost / self.simulator.drone_speed_meters_sec



