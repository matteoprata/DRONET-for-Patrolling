
from src.world_entities.entity import SimulatedEntity


class Target(SimulatedEntity):

    def __init__(self, identifier, coords, maximum_tolerated_idleness, simulator):
        SimulatedEntity.__init__(self, identifier, coords, simulator)
        self.maximum_tolerated_idleness = maximum_tolerated_idleness
        self.last_visit_ts = -maximum_tolerated_idleness / self.simulator.ts_duration_sec

    def age_of_information(self):
        return (self.simulator.cur_step - self.last_visit_ts)*self.simulator.ts_duration_sec

    def relative_aoi(self):
        return self.maximum_tolerated_idleness - self.age_of_information()
