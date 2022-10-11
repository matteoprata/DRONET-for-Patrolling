import time

import numpy as np
import matplotlib.pyplot as plt

from collections import defaultdict
from src.utilities import utilities as util

from enum import Enum
from src.simulation_setup import setup01
from src.simulation_setup import setup02

from src.simulation import simulator_patrolling as sim_pat

from src.utilities.constants import PATH_STATS
from src.utilities.constants import IndependentVariable as indv
from src.utilities.constants import DependentVariable as depv
from src.utilities.constants import Mobility as mo


class ErrorType(Enum):
    STD = "std"
    STD_ERROR = "stde"


class JSONFields(Enum):
    # L0
    VISIT_TIMES = "visit_times"
    SIMULATION_INFO = "info"

    # L1
    TOLERANCE = "targets_tolerance"
    EPISODE_DURATION = "episode_duration"
    TS_DURATION = "ts_duration_sec"


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
        return "exp_se-{}_nd-{}_nt-{}_pol-{}_sp-{}_tolf-{}.json".format(self.sim_seed,
                                                                        self.n_drones,
                                                                        self.n_targets,
                                                                        self.drone_mobility.value,
                                                                        self.drone_speed_meters_sec,
                                                                        self.tolerance_factor
                                                                        )

    def load_metrics(self):
        print("Looking for file", self.fname_generator())
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
        n_violations = np.sum((y_axis >= 1) * 1 & (np.roll(y_axis, 1) < 1) * 1, axis=0)
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

        tols = {t.identifier: t.maximum_tolerated_idleness for t in self.simulator.environment.targets}
        self.to_store_dictionary[JSONFields.SIMULATION_INFO.value][JSONFields.TOLERANCE.value] = tols

    def fname_generator(self):
        # independent variables
        return "exp_se-{}_nd-{}_nt-{}_pol-{}_sp-{}_tolf-{}.json".format(self.simulator.sim_seed,
                                                                        self.simulator.n_drones,
                                                                        self.simulator.n_targets,
                                                                        self.simulator.drone_mobility.value,
                                                                        self.simulator.drone_speed_meters_sec,
                                                                        self.simulator.tolerance_factor
                                                                        )

    def save_metrics(self):
        util.write_json(self.to_store_dictionary, PATH_STATS + self.fname_generator())

    def visit_done(self, drone, target, time_visit):
        """ Saves in the matrix the visit time of the drone to the target. """
        self.times_visit[target.identifier][drone.identifier].append(time_visit)
        self.to_store_dictionary[JSONFields.VISIT_TIMES.value] = self.times_visit


def __setup_simulation(args):
    algorithm, seed, d_speed, d_number, t_number, t_factor = args
    sim = sim_pat.PatrollingSimulator(tolerance_factor=t_factor,
                                      n_targets=t_number,
                                      drone_speed=d_speed,
                                      n_drones=d_number,
                                      drone_mobility=algorithm,
                                      sim_seed=seed)
    # sim.run(just_setup=True)
    return sim


def __data_matrix_multiple_exps(setup_file, independent_variable):
    """ Assuming that all the files are present according to setup_file, it generates the matrix
    TIME x SEEDS x ALGORITHMS x TARGETS x INDEPENDENT containing the AOI of the targets. """

    stp = setup_file
    indv_fixed_original = {k: stp.indv_fixed[k] for k in stp.indv_fixed}
    TOT_MAT = None
    is_first = True

    for ai, a in enumerate(stp.comp_dims[indv.ALGORITHM]):
        for si, s in enumerate(stp.comp_dims[indv.SEED]):
            for x_var_k in stp.indv_vary:
                if x_var_k != independent_variable:
                    continue
                X_var = stp.indv_vary[x_var_k]
                for xi, x in enumerate(X_var):
                    stp.indv_fixed[x_var_k] = x

                    # process = (a, s) + tuple(stp.indv_fixed.values())
                    met = MetricsEvaluation(sim_seed               = s,
                                            drone_mobility         = a,
                                            n_drones               = stp.indv_fixed[indv.DRONES_NUMBER],
                                            n_targets              = stp.indv_fixed[indv.TARGETS_NUMBER],
                                            drone_speed_meters_sec = stp.indv_fixed[indv.DRONES_SPEED],
                                            tolerance_factor       = stp.indv_fixed[indv.TARGETS_TOLERANCE])

                    met.load_metrics()

                    N_TARGETS = max(stp.indv_vary[indv.TARGETS_NUMBER]) if independent_variable == indv.TARGETS_NUMBER else stp.indv_fixed[indv.TARGETS_NUMBER]

                    times = []  # for each target
                    for t_id in met.targets_tolerance:
                        if t_id != str(0):
                            _, timee = met.AOI_func(t_id, is_absolute=False)
                            times.append(timee)

                    times_array = np.asarray(times).T  # rows is time, column are targets
                    times_array = np.pad(times_array, ((0, 0), (0, N_TARGETS - times_array.shape[1])), 'constant', constant_values=((0, 0), (0, 0)))  # adds zero vectors

                    if is_first:
                        # SIGNATURE: TIME x SEEDS x ALGORITHMS x TARGETS x INDEPENDENT
                        TOT_MAT = np.zeros((len(times[0]),
                                            len(stp.comp_dims[indv.SEED]),
                                            len(stp.comp_dims[indv.ALGORITHM]),
                                            N_TARGETS,
                                            len(X_var)))
                        is_first = False

                    stp.indv_fixed = {k: indv_fixed_original[k] for k in indv_fixed_original}  # reset the change
                    TOT_MAT[:, si, ai, :, xi] = times_array
    return TOT_MAT


dep_var_map = {depv.CUMULATIVE_AR: MetricsEvaluation.AOI1_integral_func,
               depv.CUMULATIVE_DELAY_AR: MetricsEvaluation.AOI5_violation_time_func,
               depv.WORST_DELAY: MetricsEvaluation.AOI3_max_delay_func,
               depv.WORST_AGE: MetricsEvaluation.AOI2_max_func,
               depv.VIOLATION_NUMBER: MetricsEvaluation.AOI4_n_violations_func
               }


def plot_stats_dep_ind_var(setup, indep_var, dep_var, error_type=ErrorType.STD_ERROR, is_boxplot=True):
    """ Given a matrix of data, plots an XY chart """
    data = __data_matrix_multiple_exps(setup, indep_var)

    # removes temporal dimensions, becomes: SEEDS x ALGORITHMS x TARGETS x INDEPENDENT
    metrics_aoi = dep_var_map[dep_var](data)

    plt.close('all')
    _, ax = plt.subplots()

    # BOXPLOT
    if is_boxplot:
        X = setup01.indv_vary[indep_var]
        AL = setup01.comp_dims[indv.ALGORITHM]
        boxes = []

        for al in range(len(AL)):
            for xi in range(len(X)):
                # this is done because when the independent variable is the number of targets, the average must be done on
                # a limited set of columns, for each tick
                N_TARGETS = xi if indep_var == indv.TARGETS_NUMBER else setup.indv_fixed[indv.TARGETS_NUMBER]
                data = metrics_aoi[:, al, :N_TARGETS, xi].ravel()
                bp = util.box_plot(data, pos=[al + xi * (len(AL)+1)], edge_color=util.sample_color(al), fill_color=util.sample_color(al))
                if xi == 0:
                    boxes.append(bp["boxes"][0])
        plt.xticks(np.arange(0, len(X) * (len(AL)+1), len(AL)+1), X)
        plt.legend(boxes, [al.name for al in AL])

    # LINE PLOT
    else:
        n_dims = len(setup.indv_vary[indep_var])
        for al in range(len(setup01.comp_dims[indv.ALGORITHM])):
            X, Y, Y_std, Y_ste = setup.indv_vary[indep_var], np.zeros(n_dims), np.zeros(n_dims), np.zeros(n_dims)
            for x_ind, xi in enumerate(setup01.indv_vary[indep_var]):
                print(x_ind, xi, setup01.indv_vary[indep_var])
                data = metrics_aoi[:, al, :xi, x_ind] if indep_var == indv.TARGETS_NUMBER else metrics_aoi[:, al, :, x_ind]
                Y[x_ind] = np.average(data, axis=(0, 1))
                Y_std[x_ind] = np.std(data, axis=(0, 1))
                Y_ste[x_ind] = np.std(data, axis=(0, 1)) / np.sqrt(len(data.ravel()))

            error = Y_std
            if error_type == ErrorType.STD_ERROR:
                error = Y_ste   # standard error vs standard deviation
            elif error_type == ErrorType.STD:
                error = Y_std

            ax.plot(X, Y, label=setup01.comp_dims[indv.ALGORITHM][al].name)
            ax.fill_between(X, Y+error, Y-error, alpha=.2)
        plt.xticks(setup.indv_vary[indep_var])
        plt.legend()

    plt.xlabel(indep_var.value["NAME"])
    plt.ylabel(dep_var.value["NAME"])
    plt.tight_layout()
    plt.show()


def plot_stats_single_seed(setup, seed, algorithm):
    """ Given a matrix of data, plots an XY chart """

    met = MetricsEvaluation(sim_seed=seed,
                            drone_mobility=algorithm,
                            n_drones=setup.indv_fixed[indv.DRONES_NUMBER],
                            n_targets=setup.indv_fixed[indv.TARGETS_NUMBER],
                            drone_speed_meters_sec=setup.indv_fixed[indv.DRONES_SPEED],
                            tolerance_factor=setup.indv_fixed[indv.TARGETS_TOLERANCE])
    # N 1
    X, Yavg = met.plot_avg_aoi()

    # N 2
    for t in range(1, setup.indv_fixed[indv.TARGETS_NUMBER]+1):
        met.plot_aoi(t)

    print()
    print("STATS for seed", seed, "with mobility", algorithm)
    print(depv.CUMULATIVE_AR.name, "on avg vector:", dep_var_map[depv.CUMULATIVE_AR](Yavg))
    print(depv.CUMULATIVE_DELAY_AR.name, "on avg vector:", dep_var_map[depv.CUMULATIVE_DELAY_AR](Yavg))
    print(depv.WORST_DELAY.name, "on avg vector:", dep_var_map[depv.WORST_DELAY](Yavg))
    print(depv.WORST_AGE.name, "on avg vector:", dep_var_map[depv.WORST_AGE](Yavg))
    print(depv.VIOLATION_NUMBER.name, "on avg vector:", dep_var_map[depv.VIOLATION_NUMBER](Yavg))
    print()

    met.load_metrics()



if __name__ == '__main__':
    # 1. Declare independent variables and their domain
    # 2. Declare what independent variable varies at this execution and what stays fixed

    # plot_stats_single_seed(setup01, seed=0, algorithm=mo.RANDOM_MOVEMENT)

    # plot_stats_dep_ind_var(setup01, indv.TARGETS_NUMBER, depv.CUMULATIVE_AR, is_boxplot=False)
    # plot_stats_dep_ind_var(setup01, indv.TARGETS_NUMBER, depv.VIOLATION_NUMBER, is_boxplot=False)
    # plot_stats_dep_ind_var(setup01, indv.TARGETS_NUMBER, depv.WORST_DELAY, is_boxplot=False)
    # plot_stats_dep_ind_var(setup01, indv.TARGETS_NUMBER, depv.WORST_AGE, is_boxplot=False)
    # plot_stats_dep_ind_var(setup01, indv.TARGETS_NUMBER, depv.CUMULATIVE_DELAY_AR, is_boxplot=False)
    #
    # plot_stats_dep_ind_var(setup01, indv.DRONES_NUMBER, depv.CUMULATIVE_AR, is_boxplot=False)
    # plot_stats_dep_ind_var(setup01, indv.DRONES_NUMBER, depv.VIOLATION_NUMBER, is_boxplot=False)
    # plot_stats_dep_ind_var(setup01, indv.DRONES_NUMBER, depv.WORST_DELAY, is_boxplot=False)
    # plot_stats_dep_ind_var(setup01, indv.DRONES_NUMBER, depv.WORST_AGE, is_boxplot=False)
    # plot_stats_dep_ind_var(setup01, indv.DRONES_NUMBER, depv.CUMULATIVE_DELAY_AR, is_boxplot=False)
    #
    # plot_stats_dep_ind_var(setup01, indv.DRONES_SPEED, depv.CUMULATIVE_AR, is_boxplot=False)
    # plot_stats_dep_ind_var(setup01, indv.DRONES_SPEED, depv.VIOLATION_NUMBER, is_boxplot=False)
    # plot_stats_dep_ind_var(setup01, indv.DRONES_SPEED, depv.WORST_DELAY, is_boxplot=False)
    # plot_stats_dep_ind_var(setup01, indv.DRONES_SPEED, depv.WORST_AGE, is_boxplot=False)
    # plot_stats_dep_ind_var(setup01, indv.DRONES_SPEED, depv.CUMULATIVE_DELAY_AR, is_boxplot=False)
    #
    plot_stats_dep_ind_var(setup01, indv.TARGETS_TOLERANCE, depv.CUMULATIVE_AR, is_boxplot=False)
    # plot_stats_dep_ind_var(setup01, indv.TARGETS_TOLERANCE, depv.VIOLATION_NUMBER, is_boxplot=False)
    # plot_stats_dep_ind_var(setup01, indv.TARGETS_TOLERANCE, depv.WORST_DELAY, is_boxplot=False)
    # plot_stats_dep_ind_var(setup01, indv.TARGETS_TOLERANCE, depv.WORST_AGE, is_boxplot=False)
    # plot_stats_dep_ind_var(setup01, indv.TARGETS_TOLERANCE, depv.CUMULATIVE_DELAY_AR, is_boxplot=False)

