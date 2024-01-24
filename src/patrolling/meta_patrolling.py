
from src.utilities.utilities import euclidean_distance
class PlottingStyle:
    line_tick = "-"
    marker = "o"


class PatrollingPolicy(PlottingStyle):  # LIVE
    name = None
    identifier = None

    def __init__(self, patrol_drone, set_drones, set_targets):
        self.patrol_drone = patrol_drone
        self.set_drones = set_drones
        self.set_targets = set_targets

    def next_visit(self, visited_before=None):
        pass


class PrecomputedPolicy(PatrollingPolicy):  # PRECOMPUTED
    name = "PrecomputedPolicy"
    identifier = 4
    line_tick = "-"
    marker = "+"

    def __init__(self, set_drones, set_targets, depot=None):
        super().__init__(patrol_drone=None, set_drones=set_drones, set_targets=set_targets)
        self.depot = None
        self.cyclic_to_visit = self.set_tour()  # map a drone to a list of target ids
        # self.drone_visit_last = {d.identifier: 0 for d in set_drones}
        self.traveled_so_far = {d.identifier: 0 for d in set_drones}
        self.last_visit_tid  = {d.identifier: 0 for d in set_drones}  # id tsp
        self.battery_limit = self.set_drones[0].cf.DRONE_MAX_ENERGY

    def add_cyclic_to_visit(self, cyclic_to_visit):
        self.cyclic_to_visit = cyclic_to_visit

    def next_visit(self, drone_id):
        """ Called every time the drone reaches a new target"""
        tid =  self.last_visit_tid[drone_id]
        news_target_id = self.cyclic_to_visit[drone_id][tid]
        tid = (tid + 1) % len(self.cyclic_to_visit[drone_id])
        self.last_visit_tid[drone_id] = tid
        return self.set_targets[news_target_id]

    def next_visit_battery(self, drone_id):
        """ Called every time the drone reaches a new target"""



        tid = self.last_visit_tid[drone_id]
        old_target_id = self.cyclic_to_visit[drone_id][tid]
        old_target = self.set_targets[old_target_id]
        # print(tid, old_target_id, self.battery_limit, self.cyclic_to_visit[drone_id])

        tid = (tid + 1) % len(self.cyclic_to_visit[drone_id])
        new_target_id = self.cyclic_to_visit[drone_id][tid]
        new_target = self.set_targets[new_target_id]
        # print(tid, new_target_id, self.battery_limit, old_target.coords, new_target.coords)

        dist_to_depot = euclidean_distance(old_target.coords, self.depot.coords)
        dist_to_next = euclidean_distance(old_target.coords, new_target.coords)

        if drone_id == 2:
            print(drone_id, dist_to_next, self.cyclic_to_visit[drone_id])

        # print(travelled, dist_to_depot, self.battery_limit)
        # print()

        energy_tu = 2140 / 2860  # distance traveled / time tu
        if dist_to_next == 0:
            self.traveled_so_far[drone_id] += energy_tu

        travelled = self.traveled_so_far[drone_id]
        if travelled + dist_to_depot >= self.battery_limit:  # limit reached
            self.traveled_so_far[drone_id] = 0  # reset the battery
            return self.depot
        else:
            self.traveled_so_far[drone_id] += dist_to_next
            self.last_visit_tid[drone_id] = tid
            return self.set_targets[self.cyclic_to_visit[drone_id][tid]]

    def set_tour(self) -> dict:  # must override everywhere
        return dict()

    def set_route_info(self, info):
        self.route_info = info


class RLPolicy(PatrollingPolicy):
    """ Reinforcement Learning trained policy. """
    name = "Go-RL"
    identifier = 4
    line_tick = "-"
    marker = "o"

    def __init__(self, patrol_drone, set_drones, set_targets):
        super().__init__(patrol_drone=patrol_drone, set_drones=set_drones, set_targets=set_targets)

    def next_visit(self):
        """ Returns a random target. """
        pass
