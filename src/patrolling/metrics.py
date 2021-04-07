
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from src.utilities import config, utilities


class Metrics:
    def __init__(self, simulator):
        self.aois = []
        self.losses = []
        self.rewards = []

        self.cum_loss = 0
        self.cum_reward = 0
        self.cum_aois = 0

        self.epsilon = []

        self.simulator = simulator

    def reset_counters(self):
        self.cum_loss = 0
        self.cum_reward = 0
        self.cum_aois = 0

    @staticmethod
    def moving_average(x, w):
        return np.asarray(pd.DataFrame(x).rolling(w).mean()).T[0]

    @staticmethod
    def moving_std(x, w):
        return np.asarray(pd.DataFrame(x).rolling(w).std()).T[0]

    def plot(self, time):
        json_to_save = {"AOIS": self.aois, "LOSSES": self.losses, "REWS": self.rewards}
        utilities.save_json(json_to_save, config.RL_DATA + "data-plotted-{}.json".format(self.simulator.simulation_name()))

        ROLLING_WINDOW = 50

        fig = plt.figure()
        loss_avg, loss_std = self.moving_average(self.losses, ROLLING_WINDOW), self.moving_std(self.losses, ROLLING_WINDOW)
        plt.plot(range(len(loss_avg)), loss_avg)
        plt.fill_between(range(len(loss_avg)), loss_avg - loss_std, loss_avg + loss_std, alpha=0.25)
        fig.savefig(config.RL_DATA + "loss-training-{}.png".format(self.simulator.simulation_name()))
        plt.clf()

        aois_avg, aois_std = self.moving_average(self.aois, ROLLING_WINDOW), self.moving_std(self.aois, ROLLING_WINDOW)
        plt.plot(range(len(aois_avg)), aois_avg)
        plt.fill_between(range(len(aois_avg)), aois_avg - aois_std, aois_avg + aois_std, alpha=0.25)
        fig.savefig(config.RL_DATA + "aoi-training-{}.png".format(self.simulator.simulation_name()))
        plt.clf()

        rew_avg, rew_std = self.moving_average(self.rewards, ROLLING_WINDOW), self.moving_std(self.rewards, ROLLING_WINDOW)
        plt.plot(range(len(rew_avg)), rew_avg)
        plt.fill_between(range(len(rew_avg)), rew_avg - rew_std, rew_avg + rew_std, alpha=0.25)
        fig.savefig(config.RL_DATA + "rew-training-{}.png".format(self.simulator.simulation_name()))
        plt.clf()

        plt.plot(range(len(self.epsilon)), self.epsilon)
        fig.savefig(config.RL_DATA + "eps-training-{}.png".format(self.simulator.simulation_name()))
        plt.clf()
