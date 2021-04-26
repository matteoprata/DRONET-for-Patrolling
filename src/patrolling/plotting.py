import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


class Plotting:
    def __init__(self):
        self.fig = plt.figure()

    @staticmethod
    def moving_average(x, w):
        return np.asarray(pd.DataFrame(x).rolling(w).mean()).T[0]

    @staticmethod
    def moving_std(x, w):
        return np.asarray(pd.DataFrame(x).rolling(w).std()).T[0]

    def plot(self, time, sim_name="sim", ROLLING_WINDOW=50):
        json_to_save = {"AOIS": self.aois, "LOSSES": self.losses, "REWS": self.rewards}
        utilities.save_json(json_to_save, config.RL_DATA + "data-{}.json".format(sim_name))

        if config.DRONE_MOBILITY == config.Mobility.DECIDED:
            loss_avg, loss_std = self.moving_average(self.losses, ROLLING_WINDOW), self.moving_std(
                self.losses, ROLLING_WINDOW)
            plt.plot(range(len(loss_avg)), loss_avg)
            plt.fill_between(range(len(loss_avg)), loss_avg - loss_std, loss_avg + loss_std, alpha=0.25)
            self.fig.savefig(config.RL_DATA + "loss-training-{}.png".format(sim_name))
            plt.clf()

            rew_avg, rew_std = self.moving_average(self.rewards, ROLLING_WINDOW), self.moving_std(
                self.rewards, ROLLING_WINDOW)
            plt.plot(range(len(rew_avg)), rew_avg)
            plt.fill_between(range(len(rew_avg)), rew_avg - rew_std, rew_avg + rew_std, alpha=0.25)
            self.fig.savefig(config.RL_DATA + "rew-training-{}.png".format(sim_name))
            plt.clf()

            # plt.plot(range(len(self.epsilon)), self.epsilon)
            # self.fig.savefig(config.RL_DATA + "eps-training-{}.png".format(sim_name))
            # plt.clf()

        aois_avg, aois_std = self.moving_average(self.aois, ROLLING_WINDOW), self.moving_std(self.aois,
                                                                                             ROLLING_WINDOW)
        plt.plot(range(len(aois_avg)), aois_avg)
        plt.fill_between(range(len(aois_avg)), aois_avg - aois_std, aois_avg + aois_std, alpha=0.25)
        self.fig.savefig(config.RL_DATA + "aoi-training-{}.png".format(sim_name))
        plt.clf()