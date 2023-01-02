
import numpy as np
from src.evaluation.MetricsEvaluation import MetricsEvaluation

import matplotlib.pyplot as plt
from src.constants import ErrorType
import pandas as pd
import plotly.express as px

from src.utilities import utilities as util
from src.constants import DependentVariable as depv
from src.constants import IndependentVariable as indv
import wandb
import copy
import matplotlib

from matplotlib.pyplot import figure
figure(figsize=(8, 6), dpi=400)

matplotlib.use('agg')

dep_var_map = {depv.CUMULATIVE_AR: MetricsEvaluation.AOI1_integral_func,
               depv.CUMULATIVE_DELAY_AR: MetricsEvaluation.AOI5_violation_time_func,
               depv.WORST_DELAY: MetricsEvaluation.AOI3_max_delay_func,
               depv.WORST_AGE: MetricsEvaluation.AOI2_max_func,
               depv.VIOLATION_NUMBER: MetricsEvaluation.AOI4_n_violations_func
               }


def data_matrix_multiple_episodes(n_episodes, val_algos, metrics_logs):
    """ Assuming that all the files are present according to setup_file, it generates the matrix
    TIME x SEEDS x ALGORITHMS x TARGETS x INDEPENDENT containing the AOI of the targets. """

    TOT_MAT = None
    is_first = True

    for ai, a in enumerate(val_algos):
        for ne in range(n_episodes):
            met = MetricsEvaluation(metrics_log=metrics_logs[a][ne])

            times = []  # for each target
            for t_id in met.targets_tolerance:
                if t_id != 0:
                    _, timee = met.AOI_func(t_id, is_absolute=False)
                    times.append(timee)

            times_array = np.asarray(times).T  # rows is time, column are targets
            if is_first:
                # SIGNATURE: TIME x EPISODES x ALGORITHMS x TARGETS
                TOT_MAT = np.zeros((len(times[0]),
                                    n_episodes,
                                    len(val_algos),
                                    met.n_targets))
                is_first = False

            TOT_MAT[:, ne, ai, :] = times_array
    return TOT_MAT


def plot_validation_stats(n_episodes, val_algos, metrics_logs, dep_var, error_type=ErrorType.STD_ERROR, targets_aggregator=np.average, is_boxplot=True):
    """ Given a matrix of data, plots an XY chart """

    print("Plotting the stats...")
    data = data_matrix_multiple_episodes(n_episodes, val_algos, metrics_logs)
    print("Done filling up the matrix.")

    # removes temporal dimensions, becomes: [(TIME) X EPISODE x ALGORITHM x TARGETS]
    metrics_aoi = dep_var_map[dep_var](data)

    fig, ax = plt.subplots()
    # BOXPLOT

    if is_boxplot:
        boxes = [metrics_aoi[:, i, :].ravel() for i, _ in enumerate(val_algos)]
        ax.boxplot(boxes, labels=[v.name for v in val_algos], showmeans=True)

    plt.xlabel("Algorithms")
    plt.ylabel(dep_var.value["NAME"])
    plt.tight_layout()
    ofig = wandb.Image(copy.deepcopy(fig))
    plt.clf()
    return ofig

