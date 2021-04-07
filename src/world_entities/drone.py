
from src.world_entities.entity import SimulatedEntity
from src.world_entities.base_station import BaseStation
from src.world_entities.antenna import AntennaEquippedDevice

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

    # MOVEMENT ROUTINES

    def move(self):
        """ Called at every time step. """
        if self.mobility == config.Mobility.PLANNED:
            if self.will_reach_target():
                self.coords = self.next_target()
                self.increase_waypoint_counter()

        elif self.mobility == config.Mobility.DECIDED:
            if self.will_reach_target():
                self.coords = self.next_target()
                self.update_target_reached()
                self.invoke_patrolling_MDP()  # train nn and get next action
                self.increase_waypoint_counter()

        self.set_next_target_angle()
        self.__movement(self.angle)

        # final training over all the dataset
        # if self.simulator.cur_step == self.simulator.sim_duration_ts-1:
        #     self.state_manager.DQN.train_whole_memory(5)
        #     self.state_manager.DQN.save_model()

    def invoke_patrolling_MDP(self):
        self.state_manager.DQN.n_decision_step += 1

        self.state_manager.invoke_train()
        action = self.state_manager.invoke_predict()

        target = self.simulator.environment.targets[action]
        self.path.append(target.coords)

        self.state_manager.plot_statistics()

    def update_target_reached(self):
        """ Once reached, update target last visit """
        if self.state_manager.previous_action is not None:
            target = self.simulator.environment.targets[self.state_manager.previous_action]
            target.last_visit_ts = self.simulator.cur_step

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
        self.current_waypoint_count = self.current_waypoint_count + 1 if self.current_waypoint_count < len(self.path) - 1 else 0

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
