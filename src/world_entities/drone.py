
import src.utilities.constants
from src.world_entities.entity import SimulatedEntity
from src.world_entities.base_station import BaseStation
from src.world_entities.antenna import AntennaEquippedDevice

from src.utilities.utilities import euclidean_distance, log, angle_between_three_points
import numpy as np
from src.utilities import config
from src.patrolling.patrolling_MDP import RLModule

import src.patrolling.patrolling_planner as planners


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

        self.path = path
        self.previous_coords = path[0]
        self.current_waypoint_count = 0
        self.mobility = mobility

        self.angle, self.speed = angle, speed
        self.com_range, self.sensing_range, self.radar_range = com_range, sensing_range, radar_range
        self.max_battery, self.max_buffer = max_battery, max_buffer
        self.bs = bs

        # parameters
        self.state_manager = RLModule(self)
        self.previous_ts_coordinate = None
        self.buffer = list()
        self.was_final = False
        self.was_final_epoch = False
        self.learning_tuple = None
        self.decision_time = 0
        self.prev_target = self.simulator.environment.targets[0]
        self.cum_rew = 0
        self.prev_step_at_decision = 0

    # MOVEMENT ROUTINES

    def move(self):
        """ Called at every time step. """
        if self.mobility == src.utilities.constants.Mobility.FIXED_TRAJECTORIES:
            if self.will_reach_target():
                self.coords = self.next_target()
                self.increase_waypoint_counter()

        elif self.mobility == src.utilities.constants.Mobility.RL_DECISION:
            if config.IS_DECIDED_ON_TARGET:
                self.decided_on_target(self.will_reach_target())
            else:
                self.decided_on_flight(self.is_decision_step())

        elif self.mobility == src.utilities.constants.Mobility.RANDOM_MOVEMENT:
            if self.will_reach_target():
                self.coords = self.next_target()  # this instruction sets the position of the drone on top of the target (useful due to discrete time)
                self.handle_metrics()
                self.update_target_reached()

                target_id = self.simulator.rnd_explore.randint(0, len(self.simulator.environment.targets))
                target = self.simulator.environment.targets[target_id]
                self.update_next_target_at_reach(target)

        elif self.mobility == src.utilities.constants.Mobility.GO_MAX_AOI:
            if self.will_reach_target():
                self.coords = self.next_target()
                self.handle_metrics()
                self.update_target_reached()

                target = planners.max_aoi(self.simulator.environment.targets, self)
                self.update_next_target_at_reach(target)

        elif self.mobility == src.utilities.constants.Mobility.GO_MIN_RESIDUAL:
            if self.will_reach_target():
                self.coords = self.next_target()
                self.handle_metrics()
                self.update_target_reached()

                target = planners.min_residual(self.simulator.environment.targets, self)
                self.update_next_target_at_reach(target)

        elif self.mobility == src.utilities.constants.Mobility.GO_MIN_SUM_RESIDUAL:
            if self.will_reach_target():
                self.coords = self.next_target()
                self.handle_metrics()
                self.update_target_reached()

                target = planners.min_sum_residual(self.simulator.environment.targets,
                                                   self.current_target(),
                                                   self.speed,
                                                   self.simulator.cur_step,
                                                   self.simulator.ts_duration_sec,
                                                   self)
                self.update_next_target_at_reach(target)

        elif self.mobility == src.utilities.constants.Mobility.FREE:
            if self.will_reach_target():
                self.update_target_reached()

        elif self.mobility == src.utilities.constants.Mobility.MICHELE:
            pass

        if self.is_flying():
            self.set_next_target_angle()
            self.__movement(self.angle)

    def save_metrics(self):
        if not self.simulator.learning["is_pretrained"]:
            # self.simulator.metrics.append_statistics_on_target_reached_light(self.learning_tuple)
            if self.simulator.wandb is not None:
                reward, epsilon, loss, _, _, _, _ = self.learning_tuple
                self.cum_rew += reward

                metrics = {"cumulative_reward": self.cum_rew,
                           "experience": epsilon,
                           "loss": 0 if loss is None else loss}
                self.simulator.wandb.log(metrics)  # , commit=self.is_new_episode())
        else:
            self.simulator.metrics.append_statistics_on_target_reached(self.prev_target.identifier)

    def reset_environment_info(self):
        self.simulator.environment.reset_drones_targets()
        self.state_manager.reset_MDP()
        self.prev_target = self.simulator.environment.targets[0]
        self.path.append(self.prev_target.coords)
        self.increase_waypoint_counter()

    def is_decision_step(self):
        """ Whether is time to make a decision step or not """
        return (self.simulator.cur_step * self.simulator.ts_duration_sec) % config.DELTA_DEC == 0

    def is_flying(self):
        return not self.will_reach_target()

    def current_target(self):
        return self.simulator.environment.targets[self.prev_target.identifier]

    def update_next_target_at_reach(self, next_target):
        next_target.lock = self
        self.prev_target.lock = None

        self.prev_target = next_target
        self.path.append(next_target.coords)
        self.increase_waypoint_counter()

    def update_target_reached(self):
        """ Once reached, update target last visit """
        self.prev_target.last_visit_ts = self.simulator.cur_step
        self.prev_target.last_visit_ts_by_drone[self.identifier] = self.simulator.cur_step  # vector of times of visit

    def set_next_target_angle(self):
        """ Set the angle of the next target """
        if self.mobility != src.utilities.constants.Mobility.FREE:
            horizontal_coo = np.array([self.coords[0] + 1, self.coords[1]])
            self.angle = angle_between_three_points(self.next_target(), np.array(self.coords), horizontal_coo)

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

    def next_target(self):
        """ In case of planned movement, returns the drone target. """
        return np.array(self.path[self.current_waypoint_count])

    def will_reach_target(self):
        """ Returns true if the drone will reach its target or overcome it in this step. """
        return self.speed * self.simulator.ts_duration_sec + config.OK_VISIT_RADIUS >= euclidean_distance(self.coords, self.next_target())

    def increase_waypoint_counter(self):
        """ Cyclic visit in the waypoints list. """
        self.current_waypoint_count = self.current_waypoint_count + 1 if self.current_waypoint_count + 1 < len(self.path) else 0

    def is_new_episode(self):
        return self.prev_step_at_decision >= self.simulator.cur_step

    # DECISION STEP TYPE

    def decided_on_target(self, eval_trigger):
        # if self.will_reach_target() or self.was_final:
        if eval_trigger or self.was_final:
            # print(self.was_final, self.is_decision_step())
            self.was_final = False

            if self.will_reach_target():
                self.simulator.environment.targets[self.prev_target.identifier].lock = None
                self.coords = self.next_target()
            # t
            reward, epsilon, loss, is_end, s, s_prime = self.state_manager.invoke_train()
            action, q = (0, None) if is_end else self.state_manager.invoke_predict(s_prime)

            # it takes one step to realize it was an end None state
            if not self.is_flying():
                self.prev_target.last_visit_ts = self.simulator.cur_step + (1 if is_end else 0)
                self.prev_target = self.simulator.environment.targets[action]
                self.path.append(self.prev_target.coords)
                self.increase_waypoint_counter()

            self.was_final = is_end

            self.learning_tuple = reward, epsilon, loss, is_end, s, q, self.was_final_epoch
            self.save_metrics()
            self.was_final_epoch = False

            self.prev_step_at_decision = self.simulator.cur_step

    def decided_on_flight(self, eval_trigger):
        self.decided_on_target(eval_trigger)

    def handle_metrics(self):
        self.simulator.metrics.append_statistics_on_target_reached(self.prev_target.identifier)
        self.simulator.metricsV2.visit_done(self, self.prev_target, self.simulator.cur_step)

    def __hash__(self):
        return hash(self.identifier)
