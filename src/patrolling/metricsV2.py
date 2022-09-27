
import numpy as np
from collections import defaultdict
from src.utilities import utilities as util
import matplotlib.pyplot as plt
from src.utilities import constants as co


class MetricsV2:

    def __init__(self, simulator, offline=False):
        self.simulator = simulator
        # {Target: {Drone: Time}}
        self.times_visit = defaultdict(lambda: defaultdict(list))  # list of np.zeros((self.simulator.n_drones, self.simulator.n_targets))

        # initialize the dict
        for tidx in range(self.simulator.n_targets):
            for didx in range(self.simulator.n_drones):
                self.times_visit[tidx][didx] = list()

    def fname_generator(self):
        # independent variables
        return "exp_se-{}_nd-{}_nt-{}_du-{}_pol-{}_sp-{}.json".format(self.simulator.sim_seed,
                                                                      self.simulator.n_drones,
                                                                      self.simulator.n_targets,
                                                                      self.simulator.episode_duration,
                                                                      self.simulator.drone_mobility.value,
                                                                      self.simulator.drone_speed_meters_sec
                                                                      )

    def save_metrics(self):
        util.write_json(self.times_visit, co.PATH_STATS + self.fname_generator())

    def load_metrics(self, sim_seed, n_drones, n_targets, episode_duration, drone_mobility, drone_speed_meters_sec):
        self.times_visit = util.read_json(co.PATH_STATS + self.fname_generator())

    def visit_done(self, drone, target, time_visit):
        """ Saves in the matrix the visit time of the drone to the target. """
        self.times_visit[target.identifier][drone.identifier].append(time_visit)

    def times_visit_map(self, target, drone=None):
        """ Times of visit of input target from particular drone (if not none). """
        if drone is not None:
            return self.times_visit[target.identifier][drone.identifier]
        else:
            times = []
            for didx in range(self.simulator.n_drones):
                times += self.times_visit[target.identifier][didx]
            return sorted(times)

    def __AOI_func_PH1(self, target, drone=None):
        # careful adding [self.simulator.episode_duration] adds the last visit even if it did not happen

        def visit_AOI_map(time_visit, target_tolerance):
            # age of information on the Y, X is the time of the visit
            difs = np.roll(time_visit, -1) - np.array(time_visit)
            difs = np.roll(difs, 1)

            Y = difs / target_tolerance
            Y[0] = 0
            X = time_visit
            return X, Y

        def AOI_progressions(X, Y):
            """ returns a dictionary of line objects one for each segment"""
            funcs_dic = dict()
            for i in range(len(X) - 1):
                xs = [X[i], X[i + 1]]
                ys = [0, Y[i + 1]]
                # coefficients = np.polyfit(xs, ys, 1)
                coefficients = [(ys[1] - ys[0]) / (xs[1] - xs[0]), (xs[1]*ys[0] - xs[0]*ys[1])/(xs[1] - xs[0])]  # m & c
                funcs_dic[i + 1] = np.poly1d(coefficients)
            return funcs_dic

        time_visit = [0] + self.times_visit_map(target, drone) + [self.simulator.episode_duration]
        time_visit = np.asarray(time_visit) * self.simulator.ts_duration_sec  # seconds
        target_tolerance = target.maximum_tolerated_idleness

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
            return lines[id_function](x)

    # ------------ FUNCTIONS ------------

    # AOI
    def AOI_func(self, target, drone=None, density=1000):
        MAX_TIME = self.simulator.episode_duration * self.simulator.ts_duration_sec
        x_axis = np.linspace(0, MAX_TIME, density)

        X, Y, lines = self.__AOI_func_PH1(target, drone)
        y_axis = np.array([self.__AOI_func_PH2(i, X, lines) for i in x_axis])
        return x_axis, y_axis

    # AOI INTEGRAL
    def AOI_integral_func(self, target, drone=None, density=1000):
        _, y_axis = self.AOI_func(target, drone, density)
        sums = sum(y_axis)
        return sums

    # N_VIOLATIONS
    def AOI_n_violations_func(self, target, drone=None, density=1000):
        _, y_axis = self.AOI_func(target, drone, density)
        n_violations = sum((y_axis >= 1) & (np.roll(y_axis, 1) < 1) * 1)
        return n_violations

    # TIME VIOLATION
    def AOI_violation_time_func(self, target, drone=None, density=1000):
        _, y_axis = self.AOI_func(target, drone, density)
        violation_time = sum((y_axis >= 1) * 1)
        return violation_time

    # ----- mean over targets
    def per_target_metrics(self, metric, drone=None, density=1000):
        vals = [metric(t, drone, density) for t in self.simulator.environment.targets if t.identifier != 0]
        return np.average(vals), np.std(vals)

    def print_all_metrics(self):
        t1 = self.simulator.environment.targets[7]

        # AOI target 1 plot
        xa, ya = self.AOI_func(t1)
        self.plot_AOI(xa, ya)

        print("AOI INTEGRAL t1", self.AOI_integral_func(t1))
        print("N_VIOLATIONS t1", self.AOI_n_violations_func(t1))
        print("TIME VIOLATION t1", self.AOI_violation_time_func(t1))

        print("AOI INTEGRAL mean", self.per_target_metrics(self.AOI_integral_func))
        print("N_VIOLATIONS mean", self.per_target_metrics(self.AOI_n_violations_func))
        print("TIME VIOLATION mean", self.per_target_metrics(self.AOI_violation_time_func))

    # PLOTTING

    def plot_AOI(self, x_axis, y_axis):
        plt.plot(x_axis, y_axis)
        plt.hlines(y=1, xmin=0, xmax=max(x_axis), color="red")
        plt.show()

    # def measure_mean_AOI(self):
    #     YYS = []
    #     DENSITY = 1000
    #     for t in self.simulator.environment.targets:
    #         if t.identifier != 0:
    #             X, Y, lines = self.__AOI_funcs(t, None)
    #             x_axis = np.linspace(0, max(X), DENSITY)
    #             y_axis = np.array([self.AOI_func(i, X, lines) for i in x_axis])
    #             YYS.append(y_axis)
    #
    #     YYS = np.asarray(YYS)
    #     avg_YYS = np.average(YYS, axis=0)
    #     quality = np.sum(avg_YYS)
    #     print("QUALITY", quality)
    #
    #     plt.plot(x_axis, avg_YYS)
    #     plt.hlines(y=1, xmin=0, xmax=max(X), color="red")
    #     plt.show()

