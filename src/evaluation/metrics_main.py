import time

import numpy as np
import matplotlib.pyplot as plt

from src.utilities import utilities as util

from enum import Enum
from src.simulation_setup import setup01
from src.utilities.constants import IndependentVariable as indv
from src.utilities.constants import PatrollingProtocol as mo
from src.utilities.constants import DependentVariable as depv
from src.evaluation.MetricsEvaluation import MetricsEvaluation


class ErrorType(Enum):
    STD = "std"
    STD_ERROR = "stde"


dep_var_map = {depv.CUMULATIVE_AR: MetricsEvaluation.AOI1_integral,
               depv.CUMULATIVE_DELAY_AR: MetricsEvaluation.AOI5_violation_time,
               depv.WORST_DELAY: MetricsEvaluation.AOI3_worst_delay,
               depv.WORST_AGE: MetricsEvaluation.AOI2_worst_age,
               depv.VIOLATION_NUMBER: MetricsEvaluation.AOI4_n_violations
               }


def __data_matrix_multiple_exps(setup_file, independent_variable):
    """ Assuming that all the files are present according to setup_file, it generates the matrix
    TIME x SEEDS x ALGORITHMS x TARGETS x INDEPENDENT containing the AOI of the targets. """

    stp = setup_file
    indv_fixed_original = {k: stp.indv_fixed[k] for k in stp.indv_fixed}
    TOT_MAT = None
    is_first = True

    for ai, a in enumerate(stp.comp_dims[indv.DRONE_PATROLLING_POLICY]):
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
                                            drone_speed_meters_sec = stp.indv_fixed[indv.DRONE_SPEED],
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
                                            len(stp.comp_dims[indv.DRONE_PATROLLING_POLICY]),
                                            N_TARGETS,
                                            len(X_var)))
                        is_first = False

                    stp.indv_fixed = {k: indv_fixed_original[k] for k in indv_fixed_original}  # reset the change
                    TOT_MAT[:, si, ai, :, xi] = times_array
    return TOT_MAT


def plot_stats_dep_ind_var(setup, indep_var, dep_var, error_type=ErrorType.STD_ERROR, targets_aggregator=np.average, is_boxplot=True):
    """ Given a matrix of data, plots an XY chart """
    print("Plotting the stats...")
    data = __data_matrix_multiple_exps(setup, indep_var)

    # removes temporal dimensions, becomes: [(TIME) X SEEDS x ALGORITHMS x TARGETS x INDEPENDENT]
    metrics_aoi = dep_var_map[dep_var](data)

    plt.close('all')
    _, ax = plt.subplots()

    # BOXPLOT
    if is_boxplot:
        X = setup01.indv_vary[indep_var]
        AL = setup01.comp_dims[indv.DRONE_PATROLLING_POLICY]
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
        for al in range(len(setup01.comp_dims[indv.DRONE_PATROLLING_POLICY])):
            X, Y, Y_std, Y_ste = setup.indv_vary[indep_var], np.zeros(n_dims), np.zeros(n_dims), np.zeros(n_dims)
            for x_ind, xi in enumerate(setup01.indv_vary[indep_var]):
                # print(x_ind, xi, setup01.indv_vary[indep_var])
                data = metrics_aoi[:, al, :xi, x_ind] if indep_var == indv.TARGETS_NUMBER else metrics_aoi[:, al, :, x_ind]
                Y[x_ind] = np.average(targets_aggregator(data, axis=1), axis=0)
                Y_std[x_ind] = np.std(targets_aggregator(data, axis=1), axis=0)
                Y_ste[x_ind] = np.std(targets_aggregator(data, axis=1), axis=0) / np.sqrt(len(data.ravel()))

            error = Y_std
            if error_type == ErrorType.STD_ERROR:
                error = Y_ste   # standard error vs standard deviation
            elif error_type == ErrorType.STD:
                error = Y_std

            ax.plot(X, Y, label=setup01.comp_dims[indv.DRONE_PATROLLING_POLICY][al].name)
            ax.fill_between(X, Y+error, Y-error, alpha=.2)
        plt.xticks(setup.indv_vary[indep_var])
        plt.legend()

    plt.xlabel(indep_var.value["NAME"])
    plt.ylabel(dep_var.value["NAME"] + "tar-agg {}".format(targets_aggregator.__name__))
    plt.tight_layout()
    plt.show()


def plot_stats_single_seed(setup, seed, algorithm):
    """ Given a matrix of data, plots an XY chart """

    met = MetricsEvaluation(sim_seed=seed,
                            drone_mobility=algorithm,
                            n_drones=setup.indv_fixed[indv.DRONES_NUMBER],
                            n_targets=setup.indv_fixed[indv.TARGETS_NUMBER],
                            drone_speed_meters_sec=setup.indv_fixed[indv.DRONE_SPEED],
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

    # python -m src.evaluation.metrics_main

    # X, Y
    plot_stats_dep_ind_var(setup01, indv.DRONES_NUMBER, depv.CUMULATIVE_AR, is_boxplot=False)
    plot_stats_dep_ind_var(setup01, indv.DRONE_SPEED, depv.CUMULATIVE_AR, is_boxplot=False)
    plot_stats_dep_ind_var(setup01, indv.TARGETS_TOLERANCE, depv.CUMULATIVE_AR, is_boxplot=False)
    plot_stats_dep_ind_var(setup01, indv.TARGETS_NUMBER, depv.CUMULATIVE_AR, is_boxplot=False)

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
    # #
    # plot_stats_dep_ind_var(setup01, indv.DRONE_SPEED, depv.CUMULATIVE_AR, is_boxplot=False)
    # plot_stats_dep_ind_var(setup01, indv.DRONE_SPEED, depv.VIOLATION_NUMBER, is_boxplot=False)
    # plot_stats_dep_ind_var(setup01, indv.DRONE_SPEED, depv.WORST_DELAY, is_boxplot=False)
    # plot_stats_dep_ind_var(setup01, indv.DRONE_SPEED, depv.WORST_AGE, is_boxplot=False)
    # plot_stats_dep_ind_var(setup01, indv.DRONE_SPEED, depv.CUMULATIVE_DELAY_AR, is_boxplot=False)
    # #
    # plot_stats_dep_ind_var(setup01, indv.TARGETS_TOLERANCE, depv.CUMULATIVE_AR, is_boxplot=False)
    # # plot_stats_dep_ind_var(setup01, indv.TARGETS_TOLERANCE, depv.VIOLATION_NUMBER, is_boxplot=False)
    # plot_stats_dep_ind_var(setup01, indv.TARGETS_TOLERANCE, depv.WORST_DELAY, is_boxplot=False)
    # plot_stats_dep_ind_var(setup01, indv.TARGETS_TOLERANCE, depv.WORST_AGE, is_boxplot=False)
    # plot_stats_dep_ind_var(setup01, indv.TARGETS_TOLERANCE, depv.CUMULATIVE_DELAY_AR, is_boxplot=False)

