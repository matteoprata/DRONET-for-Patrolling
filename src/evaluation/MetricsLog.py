from collections import defaultdict
from src.utilities import utilities as util
from src.simulation import simulator_patrolling as sim_pat
from src.utilities.constants import PATH_STATS
from src.utilities.constants import JSONFields


class MetricsLog:
    """ This class is used to log the stats of the simulation. """
    def __init__(self, simulator):
        self.simulator = simulator
        self.times_visit = defaultdict(lambda: defaultdict(list))

        self.to_store_dictionary = dict()
        for tidx in range(self.simulator.n_targets+1):
            for didx in range(self.simulator.n_drones):
                self.times_visit[tidx][didx] = list()

        self.to_store_dictionary[JSONFields.SIMULATION_INFO.value] = dict()
        self.to_store_dictionary[JSONFields.SIMULATION_INFO.value][JSONFields.EPISODE_DURATION.value] = self.simulator.episode_duration
        self.to_store_dictionary[JSONFields.SIMULATION_INFO.value][JSONFields.TS_DURATION.value] = self.simulator.ts_duration_sec

        tols = {str(t.identifier): t.maximum_tolerated_idleness for t in self.simulator.environment.targets}
        self.to_store_dictionary[JSONFields.SIMULATION_INFO.value][JSONFields.TOLERANCE.value] = tols

    def fname_generator(self):
        # independent variables
        fname = self.simulator.config.conf_description() + ".json"
        print("saving " + fname)
        return fname

    def save_metrics(self):
        util.write_json(self.to_store_dictionary, PATH_STATS + self.fname_generator())

    def visit_done(self, drone, target, time_visit):
        """ Saves in the matrix the visit time of the drone to the target. """
        self.times_visit[str(target.identifier)][str(drone.identifier)].append(time_visit)
        self.to_store_dictionary[JSONFields.VISIT_TIMES.value] = self.times_visit
