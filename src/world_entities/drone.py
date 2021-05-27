import time

from src.world_entities.entity import SimulatedEntity
from src.world_entities.base_station import BaseStation
from src.world_entities.antenna import AntennaEquippedDevice
from src.world_entities.target import Target

from src.utilities.utilities import euclidean_distance, log, angle_between_three_points
import numpy as np
from src.utilities import config
from src.patrolling.patrolling_MDP import RLModule


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

    # MOVEMENT ROUTINES

    def move(self):
        """ Called at every time step. """
        if self.mobility == config.Mobility.PLANNED:
            if self.will_reach_target():
                self.coords = self.next_target()
                self.increase_waypoint_counter()

        elif self.mobility == config.Mobility.DECIDED:

            if self.will_reach_target() or self.was_final:
                self.was_final = False

                self.coords = self.next_target()
                reward, epsilon, loss, is_end, s, s_prime = self.state_manager.invoke_train()
                action, q = (0, None) if is_end else self.state_manager.invoke_predict(s_prime)

                # self.prev_target.last_visit_ts = self.simulator.cur_step
                self.prev_target.last_visit_ts = self.simulator.cur_step + (1 if is_end else 0)
                self.prev_target = self.simulator.environment.targets[action]
                self.path.append(self.prev_target.coords)
                self.increase_waypoint_counter()
                self.was_final = is_end

                self.learning_tuple = reward, epsilon, loss, is_end, s, q, self.was_final_epoch
                self.save_metrics()
                self.was_final_epoch = False

        elif self.mobility == config.Mobility.RANDOM_MOVEMENT:
            if self.will_reach_target():
                self.coords = self.next_target()
                self.simulator.metrics.append_statistics_on_target_reached(self.prev_target.identifier)
                self.update_target_reached()

                action = self.simulator.rnd_explore.randint(0, len(self.simulator.environment.targets))
                target = self.simulator.environment.targets[action]
                self.update_next_target_at_reach(target)

        elif self.mobility == config.Mobility.GO_MAX_AOI:
            if self.will_reach_target():
                self.coords = self.next_target()
                self.simulator.metrics.append_statistics_on_target_reached(self.prev_target.identifier)
                self.update_target_reached()

                target = Target.max_aoi(self.simulator.environment.targets, self.current_target())
                self.update_next_target_at_reach(target)

        elif self.mobility == config.Mobility.GO_MIN_RESIDUAL:
            if self.will_reach_target():
                self.coords = self.next_target()
                self.simulator.metrics.append_statistics_on_target_reached(self.prev_target.identifier)
                self.update_target_reached()

                target = Target.min_residual(self.simulator.environment.targets, self.current_target())
                self.update_next_target_at_reach(target)

        elif self.mobility == config.Mobility.GO_MIN_SUM_RESIDUAL:
            if self.will_reach_target():
                self.coords = self.next_target()
                self.simulator.metrics.append_statistics_on_target_reached(self.prev_target.identifier)
                self.update_target_reached()

                target = Target.min_sum_residual(self.simulator.environment.targets,
                                                 self.current_target(),
                                                 self.speed,
                                                 self.simulator.cur_step,
                                                 self.simulator.ts_duration_sec)
                self.update_next_target_at_reach(target)

        elif self.mobility == config.Mobility.FREE:
            if self.will_reach_target():
                self.update_target_reached()

        if self.is_flying():
            self.set_next_target_angle()
            self.__movement(self.angle)

    def save_metrics(self):
        if not self.simulator.learning["is_pretrained"]:
            self.simulator.metrics.append_statistics_on_target_reached_light(self.learning_tuple)
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
        return self.simulator.cur_step % (config.DELTA_DEC / self.simulator.ts_duration_sec) == 0

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

    def set_next_target_angle(self):
        """ Set the angle of the next target """
        if self.mobility != config.Mobility.FREE:
            horizontal_coo = np.array([self.coords[0] + 1, self.coords[1]])
            self.angle = angle_between_three_points(self.next_target(), np.array(self.coords), horizontal_coo)

    def __movement(self, angle):
        """ Moves update drone coordinate based on the angle cruise """
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
        return self.speed * self.simulator.ts_duration_sec >= euclidean_distance(self.coords, self.next_target())

    def increase_waypoint_counter(self):
        """ Cyclic visit in the waypoints list. """
        self.current_waypoint_count = self.current_waypoint_count + 1 if self.current_waypoint_count + 1 < len(self.path) else 0

    # DRONE BUFFER

    def is_full(self):
        return self.buffer_length() == self.max_buffer

    def is_known_packet(self, packet):
        return packet in self.buffer

    def buffer_length(self):
        return len(self.buffer)

    # DROPPING PACKETS

    def drop_expired_packets(self, ts):
        """ Drops expired packets form the buffer. """
        for packet in self.buffer:
            if packet.is_expired(ts):
                self.buffer.remove(packet)
                log("drone: {} - removed a packet id: {}".format(str(self.identifier), str(packet.identifier)))

    def drop_packets(self, packets):
        """ Drops the packets from the buffer. """
        for packet in packets:
            if packet in self.buffer:
                self.buffer.remove(packet)
                log("drone: {} - removed a packet id: {}".format(str(self.identifier), str(packet.identifier)))

    def routing(self, simulator):
        pass

    def feel_event(self, cur_step):
        pass

    def __hash__(self):
        return hash(self.identifier)
