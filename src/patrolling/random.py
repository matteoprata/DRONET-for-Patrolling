
from src.patrolling.meta_patrolling import PatrollingPolicy


class RandomPolicy(PatrollingPolicy):
    name = "Random"
    identifier = 0
    line_tick = 0
    marker = 0

    def __init__(self, patrol_drone, set_drones, set_targets):
        super().__init__(patrol_drone=patrol_drone, set_drones=set_drones, set_targets=set_targets)

    def next_visit(self):
        """ Returns a random target. """
        target_id = self.patrol_drone.sim.rnd_explore.randint(1, len(self.patrol_drone.sim.environment.targets))
        target = self.patrol_drone.sim.environment.targets[target_id]
        return target
