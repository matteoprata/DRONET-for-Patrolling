
from src.utilities.utilities import log, is_segments_intersect, distance_point_segment, TraversedCells, euclidean_distance
from src.world_entities.target import Target
from src.utilities.utilities import config
from tqdm import tqdm
from src.utilities import tsp
from scipy.stats import truncnorm
import numpy as np
from src.utilities import utilities as util
from collections import defaultdict

class Environment:
    """ The environment is an entity that represents the area of interest."""

    def __init__(self, width, height, simulator):

        self.simulator = simulator
        self.width = width
        self.height = height
        self.drones = None
        self.base_stations = None

        # set events, set obstacles
        self.events: list = []  # even expired ones
        self.obstacles = []
        self.targets = []

        self.closest_target = []
        self.furthest_target = []
        self.targets_dataset = []

    def add_drones(self, drones: list):
        """ add a list of drones in the env """
        log("Added {} drones to the environment.".format(len(drones)))
        self.drones = drones

    def add_base_station(self, base_stations: list):
        """ add depots in the env """
        log("Added {} base stations in the environment.".format(len(base_stations)))
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

    def reset_drones_targets(self):
        """ Reset the scenario. """

        for target in self.targets:
            target.last_visit_ts = 0  # + config.DELTA_DEC * config.SIM_TS_DURATION

        for drone in self.drones:
            drone.coords = drone.bs.coords
            # drone.path.append(drone.bs.coords)

    def generate_target_combinations(self, seed):
        """
        Assumption, file stores for each seed, up to 100 targets, up to 500 episodes
        :param seed:
        :return:
        """
        # Creates a dataset of targets to iterate over to

        # loading targets list
        to_json = util.read_json(config.TARGETS_FILE + "targets_s{}_nt{}.json".format(seed, config.N_TARGETS))
        MAX_N_EPISODES = 2000
        MAX_N_TARGETS = config.N_TARGETS

        if to_json is None:
            to_json = defaultdict(list)
            print("START: generating random episodes")

            for ep in tqdm(range(MAX_N_EPISODES)):
                coordinates = []
                for i in range(MAX_N_TARGETS):
                    point_coords = [self.simulator.rnd_env.randint(0, self.width), self.simulator.rnd_env.randint(0, self.height)]
                    coordinates.append(point_coords)
                tsp_path_time = self.tsp_path_time(coordinates)

                for i in range(MAX_N_TARGETS):
                    rtt = 2 * (euclidean_distance(coordinates[i], self.base_stations[0].coords) / self.simulator.drone_speed_meters_sec)
                    LOW = int(rtt)
                    UP = int(rtt)+1 if int(rtt) >= int(tsp_path_time) else int(tsp_path_time)
                    idleness = self.simulator.rnd_env.randint(LOW, UP)
                    to_json[ep].append((i, tuple(coordinates[i]), idleness))
            util.save_json(to_json, config.TARGETS_FILE + "targets_s{}_nt{}.json".format(seed, config.N_TARGETS))
            print("DONE: generating random episodes")

        print("LOADING random episodes")
        assert(config.N_EPISODES <= MAX_N_EPISODES)
        for ep in range(config.N_EPISODES):
            epoch_targets = []
            for t_id, t_coord, t_idleness in to_json[str(ep)][:config.N_TARGETS]:

                t = Target(identifier=len(self.base_stations) + t_id,
                           coords=tuple(t_coord),
                           maximum_tolerated_idleness=t_idleness,
                           simulator=self.simulator)

                epoch_targets.append(t)
            self.targets_dataset.append(epoch_targets)
        return self.targets_dataset

    def spawn_targets(self, targets=None):

        self.targets = []

        if targets is None:
            targets = self.generate_target_combinations(self.simulator.sim_seed)[0]

        # The base station is a target
        for i in range(self.simulator.n_base_stations):
            self.targets.append(Target(i, self.base_stations[i].coords, self.simulator.drone_max_battery, self.simulator))

        self.targets += targets

        # targets may be
        # if targets is not None:
        #     for j, (x, y, tol_del) in enumerate(targets):
        #         self.targets.append(Target(i+j+1, (x, y), tol_del, self.simulator))

        # FOR each target set the furthest and closest target
        for tar1 in self.targets:
            distances = [euclidean_distance(tar1.coords, tar2.coords) for tar2 in self.targets]
            distances_min, distances_max = distances[:], distances[:]
            distances_min[tar1.identifier] = np.inf
            distances_max[tar1.identifier] = -np.inf

            tar1.furthest_target = self.targets[np.argmax(distances_max)]
            tar1.closest_target = self.targets[np.argmin(distances_min)]

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
        # if drone.coords != self.simulator.drone_coo:
        #     u1p1, u1p2 = drone.previous_coords, drone.coords
        #
        #     path_u1p1 = [0, u1p1[0], u1p1[1]]
        #     path_u1p2 = [self.simulator.ts_duration_sec,
        #                  self.simulator.ts_duration_sec * drone.speed + u1p1[0],
        #                  self.simulator.ts_duration_sec * drone.speed + u1p1[1]]
        #
        #     for other_drone in self.drones:
        #         if drone.identifier > other_drone.identifier:
        #             u2p1, u2p2 = other_drone.previous_coords, other_drone.coords
        #
        #             path_u2p1 = [0, u2p1[0], u2p1[1]]
        #             path_u2p2 = [self.simulator.ts_duration_sec,
        #                          self.simulator.ts_duration_sec * drone.speed + u2p1[0],
        #                          self.simulator.ts_duration_sec * drone.speed + u2p1[1]]
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

    def __handle_collision(self, drone):
        """ Takes countermeasure when drone collides. """
        drone.coords = drone.path[0]

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
