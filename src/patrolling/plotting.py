import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import src.utilities.config as config

pd.set_option('display.max_columns', None)


class Plotting:
    def __init__(self, simulation_name):
        self.fig = plt.figure(1, figsize=(5, 5))

        self.simulation_name = simulation_name

        self.d1 = pd.read_csv(config.RL_DATA + simulation_name + "/log_simulation_constants.csv")
        self.d2 = pd.read_csv(config.RL_DATA + simulation_name + "/log_simulation.csv", index_col="date",
                              infer_datetime_format=True, parse_dates=True)

        self.reindex_interpolate()

        self.compute_residuals()
        self.compute_average_residual()
        self.plot_average_residual()

    def plot_average_residual(self, window=60):
        rolled_avg = self.d2["average_residual"].rolling(window).mean()
        rolled_std = self.d2["std_residual"].rolling(window).mean()

        plt.plot(self.d2["row_index"], rolled_avg)
        plt.fill_between(self.d2["row_index"],
                         rolled_avg - rolled_std,
                         rolled_avg + rolled_std, alpha=0.25)
        le = len(self.d2["average_residual"])
        plt.plot(self.d2["row_index"], [self.d2["average_residual"].mean()]*le)
        plt.title(self.simulation_name)
        plt.show()

    def compute_average_residual(self):
        column_names_res = [name for name in list(self.d2.columns.values) if "res" in name]
        self.d2["average_residual"] = self.d2[column_names_res].mean(axis=1)
        self.d2["std_residual"] = self.d2[column_names_res].std(axis=1)

    def compute_residuals(self):
        column_names_aois = [name for name in list(self.d2.columns.values) if "aoi" in name]
        column_names_ress = [col.replace("aoi", "res") for col in column_names_aois]

        df_den = np.array(([list(self.d1.transpose().iloc[1, :])] * self.d2.shape[0]))
        df_num = np.array(self.d2.loc[:, column_names_aois])
        df_ones = np.ones([self.d2.shape[0], len(column_names_aois)])

        df_residuals = df_ones - df_num / df_den
        df_residuals = pd.DataFrame(df_residuals).set_index(self.d2.index)

        self.d2 = pd.concat([self.d2, df_residuals], axis=1)
        self.d2 = self.d2.rename(columns={i: n for i, n in enumerate(column_names_ress)})

    def reindex_interpolate(self):
        idx = pd.date_range(self.d2.index[0], self.d2.index[-1], freq='1S')

        self.d3 = pd.DataFrame(index=idx, columns=self.d2.columns)
        self.d3 = self.d3.append(self.d2)
        self.d2 = self.d3

        self.d2.sort_index(inplace=True)
        self.d2 = self.d2.interpolate()
        self.d2["row_index"] = list(range(self.d2.shape[0]))

    @staticmethod
    def moving_average(x, w):
        return np.asarray(pd.DataFrame(x).rolling(w).mean()).T[0]

    @staticmethod
    def moving_std(x, w):
        return np.asarray(pd.DataFrame(x).rolling(w).std()).T[0]

    # def plot(self, time, sim_name="sim", ROLLING_WINDOW=50):
    #     json_to_save = {"AOIS": self.aois, "LOSSES": self.losses, "REWS": self.rewards}
    #     utilities.save_json(json_to_save, config.RL_DATA + "data-{}.json".format(sim_name))
    #
    #     if config.DRONE_MOBILITY == config.Mobility.DECIDED:
    #         loss_avg, loss_std = self.moving_average(self.losses, ROLLING_WINDOW), self.moving_std(
    #             self.losses, ROLLING_WINDOW)
    #         plt.plot(range(len(loss_avg)), loss_avg)
    #         plt.fill_between(range(len(loss_avg)), loss_avg - loss_std, loss_avg + loss_std, alpha=0.25)
    #         self.fig.savefig(config.RL_DATA + "loss-training-{}.png".format(sim_name))
    #         plt.clf()
    #
    #         rew_avg, rew_std = self.moving_average(self.rewards, ROLLING_WINDOW), self.moving_std(
    #             self.rewards, ROLLING_WINDOW)
    #         plt.plot(range(len(rew_avg)), rew_avg)
    #         plt.fill_between(range(len(rew_avg)), rew_avg - rew_std, rew_avg + rew_std, alpha=0.25)
    #         self.fig.savefig(config.RL_DATA + "rew-training-{}.png".format(sim_name))
    #         plt.clf()
    #
    #         # plt.plot(range(len(self.epsilon)), self.epsilon)
    #         # self.fig.savefig(config.RL_DATA + "eps-training-{}.png".format(sim_name))
    #         # plt.clf()
    #
    #     aois_avg, aois_std = self.moving_average(self.aois, ROLLING_WINDOW), self.moving_std(self.aois,
    #                                                                                          ROLLING_WINDOW)
    #     plt.plot(range(len(aois_avg)), aois_avg)
    #     plt.fill_between(range(len(aois_avg)), aois_avg - aois_std, aois_avg + aois_std, alpha=0.25)
    #     self.fig.savefig(config.RL_DATA + "aoi-training-{}.png".format(sim_name))
    #     plt.clf()


if __name__ == '__main__':
    a = Plotting("-seed24-ndrones1-mode6")

