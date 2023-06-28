
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

    def __init__(self, set_drones, set_targets):
        super().__init__(patrol_drone=None, set_drones=set_drones, set_targets=set_targets)
        self.cyclic_to_visit = self.set_tour()  # map a drone to a list of target ids
        self.drone_visit_last = {d.identifier: 0 for d in set_drones}

    def add_cyclic_to_visit(self, cyclic_to_visit):
        self.cyclic_to_visit = cyclic_to_visit

    def next_visit(self, drone):
        """ Called every time the drone reaches a new target"""
        just_visited_id = self.drone_visit_last[drone.identifier]
        news_visited_id = just_visited_id + 1 if just_visited_id < len(self.cyclic_to_visit[drone.identifier]) - 1 else 0
        self.drone_visit_last[drone.identifier] = news_visited_id
        news_target_id = self.cyclic_to_visit[drone.identifier][news_visited_id]
        return self.set_targets[news_target_id]

    def set_tour(self) -> dict:  # must override everywhere
        return dict()


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
