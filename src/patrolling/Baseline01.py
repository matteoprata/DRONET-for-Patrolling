
from src.patrolling.patrolling_baselines import PatrollingPolicy


class Baseline01(PatrollingPolicy):

    def __init__(self, patrol_drone, set_drones, set_targets):
        super().__init__(patrol_drone=patrol_drone, set_drones=set_drones, set_targets=set_targets)
        self.path_planning_dict = self.path_planning_logic()

    def next_visit(self):
        return self.path_planning_dict[self.patrol_drone]

    @staticmethod
    def path_planning_logic():
        return
