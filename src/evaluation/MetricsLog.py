from collections import defaultdict
from src.utilities import utilities as util
from src.constants import PATH_STATS
from src.constants import JSONFields


class MetricsLog:
    """ This class is used to log the stats of the simulation. """
    def __init__(self, simulator):
        self.simulator = simulator
        # times_visit = defaultdict(lambda: defaultdict(list))

        times_visit = dict()
        for tidx in range(self.simulator.n_targets+1):
            times_visit[tidx] = dict()

        for tidx in range(self.simulator.n_targets + 1):
            for didx in range(self.simulator.n_drones):
                times_visit[tidx][didx] = list()   # TARGET x DRONE : LISTA

        self.id_episode = None
        self.id_algo = None

        self.to_store_dictionary = dict()
        self.to_store_dictionary[JSONFields.SIMULATION_INFO.value] = dict()
        self.to_store_dictionary[JSONFields.SIMULATION_INFO.value][JSONFields.EPISODE_DURATION.value] = self.simulator.episode_duration
        self.to_store_dictionary[JSONFields.SIMULATION_INFO.value][JSONFields.TS_DURATION.value] = self.simulator.ts_duration_sec

        tols = {t.identifier: t.maximum_tolerated_idleness for t in self.simulator.environment.targets}
        self.to_store_dictionary[JSONFields.SIMULATION_INFO.value][JSONFields.TOLERANCE.value] = tols
        self.to_store_dictionary[JSONFields.VISIT_TIMES.value] = times_visit

    def to_json(self):
        return self.to_store_dictionary

    def set_episode_details(self, idd, algo, drone_number, target_number, drone_speed, tol_factor):
        self.to_store_dictionary[JSONFields.SIMULATION_INFO.value][JSONFields.VAL_EPISODE_ID.value] = idd
        self.to_store_dictionary[JSONFields.SIMULATION_INFO.value][JSONFields.VAL_EPISODE_ALGO.value] = algo

        self.to_store_dictionary[JSONFields.SIMULATION_INFO.value][JSONFields.DRONE_NUMBER.value] = drone_number
        self.to_store_dictionary[JSONFields.SIMULATION_INFO.value][JSONFields.TARGET_NUMBER.value] = target_number
        self.to_store_dictionary[JSONFields.SIMULATION_INFO.value][JSONFields.DRONE_SPEED.value] = drone_speed
        self.to_store_dictionary[JSONFields.SIMULATION_INFO.value][JSONFields.TOLERANCE_FACTOR.value] = tol_factor

    def fname_generator(self):
        # independent variables
        fname = self.simulator.cf.conf_description() + ".json"
        print("saving " + fname)
        return fname

    def save_metrics(self):
        util.write_json(self.to_store_dictionary, PATH_STATS + self.fname_generator())

    def visit_done(self, drone, target, time_visit):
        """ Saves in the matrix the visit time of the drone to the target. """
        self.to_store_dictionary[JSONFields.VISIT_TIMES.value][target.identifier][drone.identifier].append(time_visit)
