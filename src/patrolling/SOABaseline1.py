
from src.patrolling.Baselines import PatrollingPolicy


class Baseline01(PatrollingPolicy):

    def __init__(self, patrol_drone, set_drones, set_targets):
        super().__init__(patrol_drone=patrol_drone, set_drones=set_drones, set_targets=set_targets)

    def next_visit(self):
        pass

    @staticmethod
    def path_planning_logic():
        return
