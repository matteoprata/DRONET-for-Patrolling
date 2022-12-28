

import numpy as np
import matplotlib.pyplot as plt

from src.utilities import utilities as util

from src.constants import PATH_STATS
from src.constants import JSONFields


class MetricsEvaluation:
    """ This class is used to evaluate the stats of the simulation, logged on files. """

    def __init__(self, sim_seed, n_drones, n_targets, drone_mobility, drone_speed_meters_sec, tolerance_factor):

        self.sim_seed               = sim_seed
        self.n_drones               = n_drones
        self.n_targets              = n_targets
        self.drone_mobility         = drone_mobility
        self.drone_speed_meters_sec = drone_speed_meters_sec
        self.tolerance_factor       = tolerance_factor

        simulation_visits_info = self.load_metrics()
        self.times_visit = simulation_visits_info[JSONFields.VISIT_TIMES.value]

        self.targets_tolerance = simulation_visits_info[JSONFields.SIMULATION_INFO.value][JSONFields.TOLERANCE.value]
        self.episode_duration  = simulation_visits_info[JSONFields.SIMULATION_INFO.value][JSONFields.EPISODE_DURATION.value]
        self.ts_duration_sec   = simulation_visits_info[JSONFields.SIMULATION_INFO.value][JSONFields.TS_DURATION.value]

    def fname_generator(self):
        # independent variables
        fname = "seed={}_nd={}_nt={}_pol={}_sp={}_tolf={}.json".format(
            self.sim_seed,
            self.n_drones,
            self.n_targets,
            self.drone_mobility.name,
            self.drone_speed_meters_sec,
            self.tolerance_factor
            )
        print("reading", fname)
        return fname

    def load_metrics(self):
        # print("Looking for file", self.fname_generator())
        json_dict = util.read_json(PATH_STATS + self.fname_generator())
        return json_dict

    def plot_aoi(self, target_id, drone_id=None):
        X, Y = self.AOI_func(target_id, drone_id)
        plt.plot(X, Y)
        plt.hlines(1, 0, self.ts_duration_sec * self.episode_duration, colors="red", label="Tolerance")
        plt.xlabel("Time")
        plt.ylabel("AOI")
        plt.title("AOI for target {}".format(target_id))
        plt.legend()
        plt.show()

    def plot_avg_aoi(self, drone_id=None):
        Ys = []
        for t in range(1, self.n_targets):
            X, Y = self.AOI_func(t, drone_id)
            Ys.append(Y)

        Y_avg = np.average(Ys, axis=0)
        plt.plot(X, Y_avg)
        plt.hlines(1, 0, self.ts_duration_sec * self.episode_duration, colors="red", label="Tolerance")
        plt.xlabel("Time")
        plt.ylabel("AOI")
        plt.title("AOI average over the targets")
        plt.legend()
        plt.show()
        return X, Y_avg

    def __AOI_func_PH1(self, target_id, drone_id=None, is_absolute=False):
        # careful adding [self.simulator.episode_duration] adds the last visit even if it did not happen

        def times_visit_map(target_id, drone_id=None):
            """ Times of visit of input target from particular drone (if not none). """
            if drone_id is not None:
                return self.times_visit[str(target_id)][str(drone_id)]
            else:
                times = []
                for didx in range(self.n_drones):
                    times += self.times_visit[str(target_id)][str(didx)]
                return sorted(times)

        def visit_AOI_map(time_visit, target_tolerance):
            # age of information on the Y, X is the time of the visit
            difs = np.roll(time_visit, -1) - np.array(time_visit)
            difs = np.roll(difs, 1)

            Y = difs / (target_tolerance if not is_absolute else 1)
            Y[0] = 0
            X = time_visit
            return X, Y

        def AOI_progressions(X, Y):
            """ returns a dictionary of line objects one for each segment"""
            funcs_dic = dict()
            for i in range(len(X) - 1):
                xs = [X[i], X[i + 1]]
                ys = [0, Y[i + 1]]
                dx = xs[1] - xs[0]
                if dx > 0:
                    coefficients = [(ys[1] - ys[0]) / (xs[1] - xs[0]), (xs[1]*ys[0] - xs[0]*ys[1]) / (xs[1] - xs[0])]  # C
                    funcs_dic[i + 1] = np.poly1d(coefficients)
                else:
                    # line is parallel to Y axis as dx = 0
                    funcs_dic[i + 1] = None
            return funcs_dic

        time_visit = [0] + times_visit_map(target_id, drone_id) + [self.episode_duration]
        time_visit = np.asarray(time_visit) * self.ts_duration_sec  # seconds
        target_tolerance = self.targets_tolerance[str(target_id)]

        X, Y = visit_AOI_map(time_visit, target_tolerance)
        lines = AOI_progressions(X, Y)
        return X, Y, lines

    @staticmethod
    def __AOI_func_PH2(x, X, lines):
        """ Given x it returns the associated normalized (by tolerance) AOI """
        assert 0 <= x <= X[-1]
        if x in X:
            return 0
        else:
            id_function = np.digitize(np.array([x]), X)[0]
            func = lines[id_function]
            return func(x) if func is not None else 0

    # ------------ FUNCTIONS ------------

    # AOI
    def AOI_func(self, target_id, drone_id=None, density=1000, is_absolute=False):
        MAX_TIME = self.episode_duration * self.ts_duration_sec
        x_axis = np.linspace(0, MAX_TIME, density)

        X, _, lines = self.__AOI_func_PH1(target_id, drone_id, is_absolute)
        y_axis = np.array([self.__AOI_func_PH2(i, X, lines) for i in x_axis])
        return x_axis, y_axis

    # ------------ FUNCTIONS ------------

    @staticmethod
    def AOI1_integral_func(y_axis):
        sums = np.sum(y_axis, axis=0)
        return sums

    @staticmethod
    def AOI2_max_func(y_axis):
        sums = np.max(y_axis, axis=0)
        return sums

    @staticmethod
    def AOI3_max_delay_func(y_axis):
        y_axis[y_axis < 1] = 0
        sums = np.max(y_axis, axis=0)
        return sums

    @staticmethod
    def AOI4_n_violations_func(y_axis):
        n_violations = np.sum((y_axis >= 1) * 1 & (np.roll(y_axis, 1, axis=0) < 1) * 1, axis=0)
        return n_violations

    @staticmethod
    def AOI5_violation_time_func(y_axis):
        violation_time = np.sum((y_axis >= 1) * 1, axis=0)
        return violation_time

    # PLOTTING
    @staticmethod
    def plot_AOI(x_axis, y_axis):
        plt.plot(x_axis, y_axis)
        plt.hlines(y=1, xmin=0, xmax=max(x_axis), color="red")
        plt.show()