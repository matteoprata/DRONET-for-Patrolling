import os

import numpy as np
import matplotlib.pyplot as plt

from src.utilities import utilities as util

from src.constants import PATH_STATS
from src.constants import JSONFields

from src.evaluation.MetricsLog import MetricsLog
import logging


class MetricsEvaluation:
    """ This class is used to evaluate the stats of the simulation, logged on files. """

    def __init__(self, sim_seed=None, n_drones=None, n_targets=None, drone_mobility=None, drone_speed_meters_sec=None,
                 geographic_scenario=None, tolerance_fixed=None, tolerance_scenario=None, metrics_log: dict = None):

        if metrics_log is None:
            self.sim_seed = sim_seed
            self.n_drones = n_drones
            self.n_targets = n_targets
            self.drone_mobility = drone_mobility
            self.drone_speed_meters_sec = drone_speed_meters_sec
            self.geographic_scenario = geographic_scenario
            self.tolerance_fixed = tolerance_fixed
            self.tolerance_scenario = tolerance_scenario

            simulation_visits_info = self.load_metrics()
            self.times_visit = simulation_visits_info[JSONFields.VISIT_TIMES.value]

            self.targets_tolerance = simulation_visits_info[JSONFields.SIMULATION_INFO.value][JSONFields.TOLERANCE.value]
            self.episode_duration = simulation_visits_info[JSONFields.SIMULATION_INFO.value][JSONFields.EPISODE_DURATION.value]
            self.ts_duration_sec = simulation_visits_info[JSONFields.SIMULATION_INFO.value][JSONFields.TS_DURATION.value]

        else:
            # for the validation
            self.sim_seed = metrics_log[JSONFields.SIMULATION_INFO.value][JSONFields.VAL_EPISODE_ID.value]
            self.drone_mobility = metrics_log[JSONFields.SIMULATION_INFO.value][JSONFields.VAL_EPISODE_ALGO.value]

            self.n_drones = metrics_log[JSONFields.SIMULATION_INFO.value][JSONFields.DRONE_NUMBER.value]
            self.n_targets = metrics_log[JSONFields.SIMULATION_INFO.value][JSONFields.TARGET_NUMBER.value]
            self.drone_speed_meters_sec = metrics_log[JSONFields.SIMULATION_INFO.value][JSONFields.DRONE_SPEED.value]
            self.geographic_scenario = metrics_log[JSONFields.SIMULATION_INFO.value][JSONFields.GEOGRAPHIC_SCENARIO.value]
            self.tolerance_scenario = metrics_log[JSONFields.SIMULATION_INFO.value][JSONFields.TOLERANCE_SCENARIO.value]
            self.tolerance_fixed = metrics_log[JSONFields.SIMULATION_INFO.value][JSONFields.TOLERANCE_FIXED.value]

            self.times_visit = metrics_log[JSONFields.VISIT_TIMES.value]
            self.targets_tolerance = metrics_log[JSONFields.SIMULATION_INFO.value][JSONFields.TOLERANCE.value]
            self.episode_duration = metrics_log[JSONFields.SIMULATION_INFO.value][JSONFields.EPISODE_DURATION.value]
            self.ts_duration_sec = metrics_log[JSONFields.SIMULATION_INFO.value][JSONFields.TS_DURATION.value]

    def fname_generator(self):
        # independent variables
        fname = "seed={}_nd={}_nt={}_pol={}_sp={}_tolscen={}_geoscen={}_deadco={}.json".format(
            self.sim_seed,
            self.n_drones,
            self.n_targets,
            self.drone_mobility.name,
            self.drone_speed_meters_sec,
            self.tolerance_scenario,
            self.geographic_scenario,
            self.tolerance_fixed,
        )
        print("reading", fname)
        return fname

    def load_metrics(self):
        # print("Looking for file", self.fname_generator())

        path = PATH_STATS + self.fname_generator()
        json_dict = util.read_json(path)

        if not os.path.exists(path):
            raise Exception("File {} you are trying to load does not exist. Run a simulation first.".format(path))
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


    def plot_aoi_illustrative(self, target_id, drone_id=None):
        X, Y = self.AOI_func(target_id, drone_id)
        figure_size = (6, 6)

        plt.figure(figsize=figure_size)

        ima = np.argmax(Y)
        plt.scatter(X[ima], np.max(Y), color="red", label="Max AOI", zorder=1000, edgecolors='black')
        plt.fill_between(X, np.zeros(shape=len(X)), Y, color='#72B0A6', hatch='..', edgecolor='black', label='Cumulative AOI', alpha=1)
        plt.fill_between(X, np.ones (shape=len(X)), np.maximum(Y, np.ones(shape=len(X))), color='red', hatch='x', edgecolor="black", label='Cumulative AOI delay', alpha=.4)

        # aa = X[Y >= 1]
        # Find start and end points of line segments

        # for t in zip(X, Y, Y>1):
        #     print(t)
        # for i in zip(Y, Y>=1):
        #     print(i)

        plt.axhline(1, color="gray", label="Threshold")
        plt.scatter(X[Y>=1], np.ones(len(X[Y>=1])), color="red", label="Total delay", marker="s", s=[10 for _ in range(len(X[Y>=1]))], zorder=100)

        plt.plot(X, Y, label='$a_p(t)$', color="black")
        plt.xlim(-1, 2251)

        plt.xticks([0, 500, 1000, 1500, 2000, 2250], [0, 500, 1000, 1500, 2000, 2250])
        plt.yticks([0.5, 1, 1.5, 2, 2.5, 3, 3.5, 4], [30, "$\\theta(p)$", 90, 120, 150, 180, 210, 240])
        plt.xlabel("Time ($s$)")
        plt.ylabel("AOI ($s$)")



        # plt.title("AOI for target {}".format(target_id))
        plt.legend()
        import datetime
        plt.savefig(f"data/imgs/AOI-{datetime.datetime.now()}.pdf")

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

    # AOI
    def AOI_func(self, target_id, is_absolute=False):
        """ if is_absolute it is the absolute, else / the threshold! """
        # self.episode_duration seconds
        target_tolerance = self.targets_tolerance[str(target_id)]
        MAX_TIME = int(self.episode_duration * self.ts_duration_sec)  # seconds

        x_axis = np.linspace(0, MAX_TIME, MAX_TIME)

        times = []
        for didx in range(self.n_drones):  # steps
            target_id, didx = str(target_id), str(didx)
            times += self.times_visit[target_id][didx]

        times = np.array(times) * self.ts_duration_sec
        times = [int(t) for t in times]
        times = sorted(times)  # time of the visits [199, 2334, 55566]

        if len(times) > 0:
            assert max(times) <= MAX_TIME

        i = 0
        y_axis = []
        for visit_time in times + [MAX_TIME]:
            y_axis += list(np.arange(visit_time-i))
            i = visit_time
        y_axis = np.array(y_axis)

        y_axis = y_axis / (target_tolerance if not is_absolute else 1)

        # UNCOMMENT TO PLOT AOI
        # THETA = target_tolerance if is_absolute else 1
        # self.plot_x_y(x_axis, y_axis, MAX_TIME, THETA=THETA, is_show_threshold=False, is_absolute=True)

        assert(len(y_axis) == len(x_axis))
        return x_axis, y_axis

    # ------------ FUNCTIONS ------------
    def plot_x_y(self, x_axis, y_axis, MAX_TIME, THETA, is_show_threshold=True, is_absolute=False):

        # print(self.AOI1_integral_func(y_axis))
        # print(self.AOI2_max_func(y_axis))
        # print(self.AOI3_max_delay_func(y_axis, THETA))
        # print(self.AOI4_n_violations_func(y_axis, THETA))
        # print(self.AOI5_violation_time_func(y_axis, THETA))

        plt.figure(figsize=(6.7, 6))
        plt.rcParams.update({'font.size': 15})

        if not is_show_threshold:
            plt.fill_between(x_axis, y_axis, 0, color='yellow', alpha=.2)
            if is_absolute:
                AX = [int(i) for i in range(MAX_TIME) if i % 1000 == 0] + [MAX_TIME]
                plt.yticks(AX,
                           [int(i) for i in range(MAX_TIME) if i % 1000 == 0] + ["M"])  # "$\\theta(p_1)$"

                plt.xticks(AX,
                           [int(i) for i in range(MAX_TIME) if i % 1000 == 0] + ["M"])  # "$\\theta(p_1)$"
                plt.ylim((0, max(AX)))
        else:
            plt.fill_between(x_axis, y_axis, THETA, where=y_axis>THETA, color='red', alpha=.2)
            plt.axhline(THETA, color='red')
            if is_absolute:
                plt.yticks([int(i) for i in range(MAX_TIME) if i % 10000 == 0]+[MAX_TIME, THETA], [int(i) for i in range(MAX_TIME) if i % 10000 == 0]+["M", "$\\theta(p_1)$"])  # "$\\theta(p_1)$"

        plt.plot(x_axis, y_axis, color='blue', label="$a_{p1}(t)$")
        plt.xlabel('Time $(s)$')
        plt.xticks([int(i) for i in range(MAX_TIME) if i % 10000 == 0]+[MAX_TIME], [int(i) for i in range(MAX_TIME) if i % 10000 == 0]+["M"])
        plt.axvline(MAX_TIME, color='red')

        plt.ylabel('AoI $(s)$')
        plt.title('AoI of IP $p_1$')
        plt.legend()
        plt.tight_layout()
        # plt.savefig("data/aoi_t.pdf")
        plt.show()

    @staticmethod
    def AOI1_integral_func(y_axis, theta=None):
        sums = np.sum(y_axis, axis=0)
        return sums

    @staticmethod
    def AOI2_max_func(y_axis, theta=None):
        sums = np.max(y_axis, axis=0)
        return sums

    @staticmethod
    def AOI3_max_delay_func(y_axis, theta=1):
        y_axis_var = np.array(y_axis)
        y_axis_var[y_axis_var < theta] = 0
        sums = np.max(y_axis_var, axis=0)
        return sums

    @staticmethod
    def AOI4_n_violations_func(y_axis, theta=1):
        n_violations = np.sum((y_axis >= theta) * 1 & (np.roll(y_axis, 1, axis=0) < theta) * 1, axis=0)
        return n_violations

    @staticmethod
    def AOI6_cumulative_delay_AOI_func(y_axis, theta=1):
        """ somma triangoli rossi"""
        # for i in y_axis[:, 6,0,2,0]:
        #     print(i)
        # exit()  # 1, 2, 3,4
        viol = y_axis >= theta
        triangles = (y_axis-theta) * (viol * 1)
        # print(triangles.shape)
        violation_time = np.sum(triangles, axis=0)
        # integral_trapezoidal = np.trapz(triangles, axis=0)
        # print(integral_trapezoidal[0,0,0,0], violation_time[0,0,0,0])
        # print(integral_trapezoidal[0,0,2,0], violation_time[0,0,2,0])
        # print(integral_trapezoidal[1,0,2,0], violation_time[1,0,2,0])
        # exit()
        return violation_time


    @staticmethod
    def AOI5_violation_time_func(y_axis, theta=1):
        violation_time = np.sum((y_axis >= theta) * 1, axis=0)
        return violation_time

    # PLOTTING
    @staticmethod
    def plot_AOI(x_axis, y_axis):
        plt.plot(x_axis, y_axis)
        plt.hlines(y=1, xmin=0, xmax=max(x_axis), color="red")
        plt.show()
