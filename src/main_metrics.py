import numpy as np
import matplotlib.pyplot as plt

from src.utilities import utilities as util

from src.constants import Setups
from src.constants import IndependentVariable as indv
from src.constants import DependentVariable as depv
from src.constants import ErrorType
from src.evaluation.MetricsEvaluation import MetricsEvaluation


dep_var_map = {depv.CUMULATIVE_AR: MetricsEvaluation.AOI1_integral_func,
               depv.CUMULATIVE_DELAY_AR: MetricsEvaluation.AOI6_cumulative_delay_AOI_func,
               depv.WORST_DELAY: MetricsEvaluation.AOI3_max_delay_func,
               depv.WORST_AGE: MetricsEvaluation.AOI2_max_func,
               depv.VIOLATION_NUMBER: MetricsEvaluation.AOI4_n_violations_func,
               depv.DELAY_SUM: MetricsEvaluation.AOI5_violation_time_func
               }

grid_alpha = .2
figure_size = (6.7, 6)
marker_size = 8
line_width = 2
legend_font = 15
fontsize = 12


def __data_matrix_multiple_exps(setup_file, independent_variable):
    """ Assuming that all the files are present according to setup_file, it generates the matrix
    TIME x SEEDS x ALGORITHMS x TARGETS x INDEPENDENT containing the AOI of the targets. """

    stp = setup_file
    indv_fixed_original = {k: stp.indv_fixed[k] for k in stp.indv_fixed}
    TOT_MAT = None
    is_first = True

    for ai, a in enumerate(stp.comp_dims[indv.DRONE_PATROLLING_POLICY]):
        print("Algorithm", a)
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
                                            geographic_scenario= stp.indv_fixed[indv.TARGETS_POSITION_SCENARIO].name,
                                            tolerance_scenario     = stp.indv_fixed[indv.TARGETS_TOLERANCE_SCENARIO].name,
                                            tolerance_fixed=stp.indv_fixed[indv.TARGETS_TOLERANCE_FIXED])

                    met.load_metrics()
                    N_TARGETS = max(stp.indv_vary[indv.TARGETS_NUMBER]) if independent_variable == indv.TARGETS_NUMBER else stp.indv_fixed[indv.TARGETS_NUMBER]

                    times = []  # for each target
                    for t_id in met.targets_tolerance:
                        if t_id != str(0):
                            _, timee = met.AOI_func(t_id)
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

    # print(np.sum(TOT_MAT[:, 0, 0, 0, 0]))
    # print(np.sum(TOT_MAT[:, 0, 0, 0, 1]))
    # exit()
    return TOT_MAT


def plot_stats_dep_ind_var(setup, indep_var, dep_var, error_type=ErrorType.STD, targets_aggregator=np.average, is_boxplot=True):
    """ Given a matrix of data, plots an XY chart """

    print("Plotting the stats...")
    data = __data_matrix_multiple_exps(setup, indep_var)
    print("Done filling up the matrix.")

    # removes temporal dimensions, becomes: [(TIME) X SEEDS x ALGORITHMS x TARGETS x INDEPENDENT]
    metrics_aoi = dep_var_map[dep_var](data)  # data shape (5400, 1, 2, 30, 4)

    plt.close('all')
    _, ax = plt.subplots(figsize=figure_size)
    plt.grid(True, alpha=grid_alpha)

    # BOXPLOT
    if is_boxplot:
        X = setup.indv_vary[indep_var]
        AL = setup.comp_dims[indv.DRONE_PATROLLING_POLICY]
        boxes = []

        for al in range(len(AL)):
            al_id = setup.comp_dims[indv.DRONE_PATROLLING_POLICY][al].value
            for xi in range(len(X)):
                # this is done because when the independent variable is the number of targets, the average must be done on
                # a limited set of columns, for each tick
                N_TARGETS = xi if indep_var == indv.TARGETS_NUMBER else setup.indv_fixed[indv.TARGETS_NUMBER]
                data = metrics_aoi[:, al, :N_TARGETS, xi].ravel()
                bp = util.box_plot(data, pos=[al + xi * (len(AL) + 1)])  # edge_color=util.sample_color(map_color[al_id]), fill_color=util.sample_color(map_color[al_id]))

                if xi == 0:
                    boxes.append(bp["boxes"][0])

        plt.xticks(np.arange(0, len(X) * (len(AL) + 1), len(AL) + 1), X, fontsize=fontsize)
        plt.legend(boxes, [al.name for al in AL], fontsize=legend_font)

    # LINE PLOT
    else:
        n_dims = len(setup.indv_vary[indep_var])
        for al in range(len(setup.comp_dims[indv.DRONE_PATROLLING_POLICY])):
            al_id = setup.comp_dims[indv.DRONE_PATROLLING_POLICY][al].value
            X, Y, Y_std, Y_ste = setup.indv_vary[indep_var], np.zeros(n_dims), np.zeros(n_dims), np.zeros(n_dims)
            for x_ind, xi in enumerate(setup.indv_vary[indep_var]):
                data = metrics_aoi[:, al, :xi, x_ind] if indep_var == indv.TARGETS_NUMBER else metrics_aoi[:, al, :, x_ind]
                # SEEDS x TARGETS

                Y[x_ind] = np.average(targets_aggregator(data, axis=1), axis=0)
                # print(al, Y[x_ind])     # -------------------------------------------------------------------------------
                Y_std[x_ind] = np.std(targets_aggregator(data, axis=1), axis=0)
                Y_ste[x_ind] = np.std(targets_aggregator(data, axis=1), axis=0) / np.sqrt(len(data.ravel()))

            error = Y_std
            if error_type == ErrorType.STD_ERROR:
                error = Y_ste  # standard error vs standard deviation
            elif error_type == ErrorType.STD:
                error = Y_std

            plt.errorbar(X, Y, yerr=error, capsize=3,
                         label=setup.comp_dims[indv.DRONE_PATROLLING_POLICY][al].value.name,  # marker=algo_marker[al_id],
                         marker=util.sample_marker(setup.comp_dims[indv.DRONE_PATROLLING_POLICY][al].value.marker),
                         color=util.sample_color(setup.comp_dims[indv.DRONE_PATROLLING_POLICY][al].value.identifier)
                         )  # color=util.sample_color(map_color[al_id]))

            # plt.title("Scenario geo {}, temp {}".format(setup.indv_fixed[indv.TARGETS_POSITION_SCENARIO].name,
            #                                                   setup.indv_fixed[indv.TARGETS_TOLERANCE_SCENARIO].name,))

            # ax.plot(X, Y, label=setup.comp_dims[indv.ALGORITHM][al].name, color=util.sample_color(map_color[al_id]))
            # ax.fill_between(X, Y+error, Y-error, alpha=.2)
        plt.xticks(setup.indv_vary[indep_var], fontsize=fontsize)
        plt.legend(fontsize=legend_font)

    plt.xlabel(indep_var.value["NAME"], fontsize=legend_font)
    plt.ylabel(dep_var.value["NAME"], fontsize=legend_font)  # + " (tar-agg {})".format(target_aggregator.__name__))
    plt.tight_layout()

    fname = "{}_{}_{}.pdf".format(str(dep_var.name), str([v for k, v in setup.indv_fixed.items()]), str([v for k, v in setup.indv_vary.items()]))  #, str(setup.comp_dims))
    plt.savefig("data/imgs/{}".format(fname))
    plt.show()


def plot_stats_single_seed(setup, seed, algorithm):
    """ Given a matrix of data, plots an XY chart """

    met = MetricsEvaluation(sim_seed=seed,
                            drone_mobility=algorithm,
                            n_drones=setup.indv_fixed[indv.DRONES_NUMBER],
                            n_targets=setup.indv_fixed[indv.TARGETS_NUMBER],
                            drone_speed_meters_sec=setup.indv_fixed[indv.DRONE_SPEED],
                            geographic_scenario=setup.indv_fixed[indv.TARGETS_POSITION_SCENARIO],
                            tolerance_scenario=setup.indv_fixed[indv.TARGETS_TOLERANCE_SCENARIO].name,
                            tolerance_fixed=setup.indv_fixed[indv.TARGETS_TOLERANCE_FIXED])
    # N 1
    X, Yavg = met.plot_avg_aoi()

    # N 2
    for t in range(1, setup.indv_fixed[indv.TARGETS_NUMBER]+1):
        met.plot_aoi(t)

    print()
    print("STATS for seed", seed, "with patrolling_protocol", algorithm)
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

    # python -m src.main_metrics
    setu = Setups.SETUP0.value
    for k, _ in setu.indv_vary.items():
        plot_stats_dep_ind_var(setu, k, depv.CUMULATIVE_DELAY_AR, is_boxplot=False, error_type=ErrorType.STD_ERROR, targets_aggregator=np.max)
        plot_stats_dep_ind_var(setu, k, depv.CUMULATIVE_AR, is_boxplot=False, error_type=ErrorType.STD_ERROR, targets_aggregator=np.max)
        plot_stats_dep_ind_var(setu, k, depv.WORST_AGE, is_boxplot=False, error_type=ErrorType.STD_ERROR, targets_aggregator=np.max)
        plot_stats_dep_ind_var(setu, k, depv.DELAY_SUM, is_boxplot=False, error_type=ErrorType.STD_ERROR, targets_aggregator=np.max)

    # plot_stats_dep_ind_var(setu, indv.DRONES_NUMBER, depv.WORST_DELAY, is_boxplot=False, error_type=ErrorType.STD_ERROR, targets_aggregator=np.max)
    # plot_stats_dep_ind_var(setu, indv.DRONES_NUMBER, depv.VIOLATION_NUMBER, is_boxplot=False, error_type=ErrorType.STD_ERROR, targets_aggregator=np.max)

    # plot_stats_dep_ind_var(setup0, indv.DRONES_NUMBER, depv.WORST_AGE, is_boxplot=False, error_type=ErrorType.STD, targets_aggregator=np.average)
    # plot_stats_dep_ind_var(setup0, indv.DRONES_NUMBER, depv.WORST_DELAY, is_boxplot=False, error_type=ErrorType.STD, targets_aggregator=np.average)
