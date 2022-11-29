
import src.utilities.constants
from src.world_entities.entity import SimulatedEntity
from src.world_entities.base_station import BaseStation
from src.world_entities.antenna import AntennaEquippedDevice

from src.utilities.utilities import euclidean_distance, log, angle_between_three_points
import numpy as np
from src.utilities import config
import src.utilities.constants as co
from src.patrolling.RLModule import RLModule
import src.patrolling.patrolling_baselines as planners


class Drone(SimulatedEntity, AntennaEquippedDevice):

    def __init__(self,
                 identifier,
                 path: list,
                 bs: BaseStation,
                 angle, speed,
                 com_range, sensing_range, radar_range,
                 max_battery, max_buffer,
                 simulator,
                 mobility):

        SimulatedEntity.__init__(self, identifier, path[0], simulator)
        AntennaEquippedDevice.__init__(self)

        self.mobility = mobility
        self.angle, self.speed = angle, speed
        self.com_range, self.sensing_range, self.radar_range = com_range, sensing_range, radar_range
        self.max_battery, self.max_buffer = max_battery, max_buffer
        self.bs = bs

        # parameters to reset
        self.visited_targets_coordinates = path
        self.previous_coords = self.visited_targets_coordinates[0]
        self.prev_target     = self.simulator.environment.targets[0]

        self.rl_module = RLModule(self)

        # TODO: add picker policy instanciate the right policy

    def reset(self):
        self.visited_targets_coordinates = [self.visited_targets_coordinates[0]]
        self.previous_coords = self.visited_targets_coordinates[0]
        self.prev_target = self.simulator.environment.targets[0]

    # ------> (begin) MOVEMENT routines

    def is_flying(self):
        return not self.will_reach_target_now()

    def is_hovering(self):
        return not self.is_flying()

    def current_target(self):
        return self.simulator.environment.targets[self.prev_target.identifier]

    def move(self):
        """ Called at every time step. """

        if self.mobility == co.Mobility.RL_DECISION_TRAIN:
            if (self.will_reach_target_now() and config.IS_DECIDED_ON_TARGET) or (self.rl_module.is_decision_step() and not config.IS_DECIDED_ON_TARGET and self.will_reach_target_now()):
                self.coords = self.next_target_coo()  # this instruction sets the position of the drone on top of the target (useful due to discrete time)
                self.__handle_metrics()
                self.__update_target_time_visit_upon_reach()

                action = self.rl_module.query_policy()
                self.__update_next_target_upon_reach(action)

            elif self.rl_module.is_decision_step() and not config.IS_DECIDED_ON_TARGET:
                action = self.rl_module.query_policy()
                self.__update_next_target_upon_reach(action)

        elif self.mobility == co.Mobility.RL_DECISION_TEST:
            pass

        elif self.mobility == co.Mobility.RANDOM_MOVEMENT:
            if self.will_reach_target_now():
                self.coords = self.next_target_coo()  # this instruction sets the position of the drone on top of the target (useful due to discrete time)
                self.__handle_metrics()
                self.__update_target_time_visit_upon_reach()
                policy = planners.RandomPolicy(self, self.simulator.environment.drones, self.simulator.environment.targets)
                target = policy.next_visit()
                self.__update_next_target_upon_reach(target)

        elif self.mobility == co.Mobility.GO_MAX_AOI:
            if self.will_reach_target_now():
                self.coords = self.next_target_coo()
                self.__handle_metrics()
                self.__update_target_time_visit_upon_reach()

                policy = planners.MaxAOIPolicy(self, self.simulator.environment.drones, self.simulator.environment.targets)
                target = policy.next_visit()
                self.__update_next_target_upon_reach(target)

        elif self.mobility == co.Mobility.GO_MIN_RESIDUAL:
            if self.will_reach_target_now():
                self.coords = self.next_target_coo()
                self.__handle_metrics()
                self.__update_target_time_visit_upon_reach()

                policy = planners.MaxAOIRatioPolicy(self, self.simulator.environment.drones, self.simulator.environment.targets)
                target = policy.next_visit()
                self.__update_next_target_upon_reach(target)

        elif self.mobility == co.Mobility.GO_MIN_SUM_RESIDUAL:
            if self.will_reach_target_now():
                self.coords = self.next_target_coo()
                self.__handle_metrics()
                self.__update_target_time_visit_upon_reach()

                policy = planners.MaxSumResidualPolicy(self, self.simulator.environment.drones, self.simulator.environment.targets)
                target = policy.next_visit()
                self.__update_next_target_upon_reach(target)

        elif self.mobility == co.Mobility.FREE:
            if self.will_reach_target_now():
                self.__update_target_time_visit_upon_reach()

        if self.is_flying():
            self.__set_next_target_angle()
            self.__movement(self.angle)

    def __set_next_target_angle(self):
        """ Set the angle of the next target """
        if self.mobility != src.utilities.constants.Mobility.FREE:
            horizontal_coo = np.array([self.coords[0] + 1, self.coords[1]])
            self.angle = angle_between_three_points(self.next_target_coo(), np.array(self.coords), horizontal_coo)

    def __movement(self, angle):
        """ updates drone coordinate based on the angle cruise """
        self.previous_coords = np.asarray(self.coords)
        distance_travelled = self.speed * self.simulator.ts_duration_sec
        coords = np.asarray(self.coords)

        # update coordinates based on angle
        x = coords[0] + distance_travelled * np.cos(np.radians(angle))
        y = coords[1] + distance_travelled * np.sin(np.radians(angle))
        coords = [x, y]

        # do not cross walls
        coords[0] = max(0, min(coords[0], self.simulator.env_width_meters))
        coords[1] = max(0, min(coords[1], self.simulator.env_height_meters))
        self.coords = coords

    def next_target_coo(self):
        return np.array(self.visited_targets_coordinates[-1])

    def will_reach_target_now(self):
        """ Returns true if the drone will reach its target or overcome it in this step. """
        return self.speed * self.simulator.ts_duration_sec + config.OK_VISIT_RADIUS >= euclidean_distance(self.coords, self.next_target_coo())

    # ------> (end) MOVEMENT routines

    def __update_next_target_upon_reach(self, next_target):
        """ When the target is chosen, sets a new target as the next. """

        # if next_target.lock is not None:
        #   print("The drone {} looped over target {}".format(self.identifier, next_target))

        next_target.lock = self
        self.prev_target.lock = None

        self.prev_target = next_target
        self.visited_targets_coordinates.append(next_target.coords)

    def __update_target_time_visit_upon_reach(self):
        """ Once reached, update target last visit """
        self.prev_target.last_visit_ts = self.simulator.cur_step
        self.prev_target.last_visit_ts_by_drone[self.identifier] = self.simulator.cur_step  # vector of times of visit

    def __handle_metrics(self):
        if not config.IS_TRAINING_MODE or self.simulator.is_validation:
            self.simulator.metricsV2.visit_done(self, self.prev_target, self.simulator.cur_step)

    def __hash__(self):
        return hash(self.identifier)
