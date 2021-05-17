import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import src.utilities.config as config

pd.set_option('display.max_columns', None)


class Plotting:
    def __init__(self, simulation_name):
        self.fig = plt.figure()
        self.simulation_name = simulation_name

    def plot_patrolling_performance(self):
        self.d1 = pd.read_csv(config.RL_DATA + self.simulation_name + "/log_simulation_constants.csv")
        self.tar_stats = pd.read_csv(config.RL_DATA + self.simulation_name + "/log_simulation.csv",
                                     index_col="date",
                                     infer_datetime_format=True,
                                     parse_dates=True)


        self.reindex_interpolate()

        self.compute_residuals()
        self.plot_average_residual()
        self.plot_n_violations()
        self.plot_average_expiration()

    def plot_learning_performance(self):
        self.dqn_stats = pd.read_csv(config.RL_DATA + self.simulation_name + "/dqn_training_data.csv")
        self.plot_dqn_stats()

    def plot_average_residual(self, is_min_std=True, window=120):
        """ Plot min residual of the targets population. """
        rolled_min = self.tar_stats["min_residual"].rolling(window).mean()
        rolled_max = self.tar_stats["max_residual"].rolling(window).mean()
        rolled_avg = self.tar_stats["avg_residual"].rolling(window).mean()
        rolled_std = self.tar_stats["std_residual"].rolling(window).mean()

        if is_min_std:
            plt.fill_between(self.tar_stats["row_index"],
                             rolled_min,
                             rolled_max, alpha=0.10, color='red', label="min-max")
        else:
            plt.fill_between(self.tar_stats["row_index"],
                             rolled_avg - rolled_std,
                             rolled_avg + rolled_std, alpha=0.10, color='green', label="std")

        plt.plot(self.tar_stats["row_index"], rolled_avg, label="moving avg")

        le = len(self.tar_stats["avg_residual"])
        plt.plot(self.tar_stats["row_index"], [self.tar_stats["avg_residual"].mean()] * le, label="avg")

        self.__plot_now(self.simulation_name, "seconds", "moving avg ({}) avg res".format(window))

    def plot_n_violations(self, window=120):
        # print(self.d2["n_violations"])
        n_violations_rolled = self.tar_stats["n_violations"].rolling(window).mean()
        plt.plot(self.tar_stats["row_index"], n_violations_rolled, label="moving avg")

        le = len(self.tar_stats["n_violations"])
        plt.plot(self.tar_stats["row_index"], [self.tar_stats["n_violations"].mean()] * le, label="avg")

        self.__plot_now(self.simulation_name, "seconds", "moving avg ({}) number violations".format(window))

    def plot_average_expiration(self, window=120):
        """ Plot min residual of the targets population. """
        rolled_min = self.tar_stats["min_res_violations"].rolling(window).mean()
        rolled_max = self.tar_stats["max_res_violations"].rolling(window).mean()
        rolled_avg = self.tar_stats["mean_res_violations"].rolling(window).mean()

        plt.fill_between(self.tar_stats["row_index"],
                         rolled_min,
                         rolled_max, alpha=0.10, color='red', label="min-max")

        plt.plot(self.tar_stats["row_index"], rolled_avg, label="moving avg")

        le = len(self.tar_stats["mean_res_violations"])
        plt.plot(self.tar_stats["row_index"], [self.tar_stats["mean_res_violations"].mean()] * le, label="avg")

        self.__plot_now(self.simulation_name, "seconds", "moving avg ({}) avg expiration".format(window))

    def plot_dqn_stats(self, window=50):
        """ Plot min residual of the targets population. """
        rolled_avg_loss = self.dqn_stats["loss"].rolling(window).mean()
        rolled_epsilon = self.dqn_stats["epsilon"].rolling(window).mean()
        cumulative_rew = self.dqn_stats["reward"].cumsum()

        is_end = [i for i, j in enumerate(self.dqn_stats["is_end"]) if j == 1]

        plt.plot(list(self.dqn_stats.index), rolled_avg_loss, label="moving avg")
        for i in is_end:
            plt.axvline(i, color="red", alpha=0.10)
        self.__plot_now(self.simulation_name, "steps", "moving avg ({}) avg loss".format(window))

        plt.plot(list(self.dqn_stats.index), cumulative_rew, label="rew")
        self.__plot_now(self.simulation_name, "steps", "cumulative reward")

        plt.plot(list(self.dqn_stats.index), rolled_epsilon, label="experience")
        self.__plot_now(self.simulation_name, "steps", "experience probability")

    def compute_residuals(self):
        column_names_aois = [name for name in list(self.tar_stats.columns.values) if "aoi" in name]
        column_names_ress = [col.replace("aoi", "res") for col in column_names_aois]

        df_den = np.array(([list(self.d1.transpose().iloc[1, :])] * self.tar_stats.shape[0]))
        df_num = np.array(self.tar_stats.loc[:, column_names_aois])
        df_ones = np.ones([self.tar_stats.shape[0], len(column_names_aois)])

        df_residuals = df_ones - df_num / df_den
        df_residuals = pd.DataFrame(df_residuals).set_index(self.tar_stats.index)

        self.tar_stats = pd.concat([self.tar_stats, df_residuals], axis=1)
        self.tar_stats = self.tar_stats.rename(columns={i: n for i, n in enumerate(column_names_ress)})

        # average residual & std residual
        self.tar_stats["min_residual"] = self.tar_stats[column_names_ress].min(axis=1)
        self.tar_stats["max_residual"] = self.tar_stats[column_names_ress].max(axis=1)

        self.tar_stats["avg_residual"] = self.tar_stats[column_names_ress].mean(axis=1)
        self.tar_stats["std_residual"] = self.tar_stats[column_names_ress].std(axis=1)

        # number of violations
        self.tar_stats["n_violations"] = (self.tar_stats[column_names_ress] <= 0).sum(axis=1)
        df_res_vilations = (self.tar_stats[column_names_ress] <= 0) * (df_num / df_den)

        epsilon = 10e-10
        n_violations_row = (np.sum(df_res_vilations != 0, axis=1) == 0) * epsilon + np.sum(df_res_vilations != 0, axis=1)
        self.tar_stats["mean_res_violations"] = np.sum(df_res_vilations, axis=1) / n_violations_row
        self.tar_stats["min_res_violations"] = df_res_vilations.replace(0, np.inf).min(axis=1)
        self.tar_stats["max_res_violations"] = df_res_vilations.max(axis=1)

    def reindex_interpolate(self):
        idx = pd.date_range(self.tar_stats.index[0], self.tar_stats.index[-1], freq='1S')

        self.d3 = pd.DataFrame(index=idx, columns=self.tar_stats.columns)
        self.d3 = self.d3.append(self.tar_stats)
        self.tar_stats = self.d3

        self.tar_stats.sort_index(inplace=True)
        self.tar_stats = self.tar_stats.interpolate()

        self.tar_stats["row_index"] = list(range(self.tar_stats.shape[0]))

    @staticmethod
    def __plot_now(title, xlabel, ylabel):
        plt.legend(loc="upper left")
        plt.title(title)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.savefig(config.RL_DATA + "/" + title + "/" + ylabel)
        plt.clf()


if __name__ == '__main__':
    for i in range(6, 6+1):
        Plotting("-seed24-ndrones3-mode{}".format(i))

