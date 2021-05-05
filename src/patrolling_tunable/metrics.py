import matplotlib.pyplot as plt
import numpy as np
from src.utilities import config, utilities
import os

from collections import defaultdict
import pandas as pd


class Metrics:

    def __init__(self, simulator=None):
        self.simulator = simulator
        self.targets_threshold = {}

        self.visited_target = []
        self.target_aoi = defaultdict(list)
        self.cur_second = []

        self.reward = []
        self.epsilon = []
        self.loss = []
        self.end_epoch = []

    def append_statistics_on_target_reached(self, action, reward=None, epsilon=None, loss=None, end_epoch=None):
        """ Saves the current second, aoi, reward, epsilon, loss. For computing final plots. """

        for rep in [-1, 0]:
            self.cur_second.append(self.simulator.current_second(rep))
            for t in self.simulator.environment.targets:
                if rep == -1:
                    if t.identifier == action:
                        self.target_aoi[t.identifier].append(t.age_of_information(rep))
                    else:
                        self.target_aoi[t.identifier].append(None)
                else:
                    if t.identifier == action:
                        self.target_aoi[t.identifier].append(0)
                    else:
                        self.target_aoi[t.identifier].append(t.age_of_information())

        # if reward is not None and epsilon is not None and loss is not None and end_epoch is not None:
        self.reward.append(reward)
        self.epsilon.append(epsilon)
        self.loss.append(loss)
        self.end_epoch.append(1 if end_epoch else 0)

        if len(self.targets_threshold.keys()) == 0:
            self.targets_threshold = {t.identifier: t.maximum_tolerated_idleness for t in self.simulator.environment.targets}

    def save_dataframe(self):
        """ Saves data to disk. """

        N_ROWS = len(self.cur_second)
        target_aoi_columns = ["target_{}_aoi".format(t.identifier) for t in self.simulator.environment.targets]

        df_dqn = pd.DataFrame()
        df_tar = pd.DataFrame()

        # df["visited_target"] = pd.Series(self.visited_target)
        df_tar["cur_second"] = pd.Series(self.cur_second)
        df_tar["date"] = pd.Series([pd.to_datetime('now').normalize().strftime("%d-%m-%Y %H:%M:%S")]*N_ROWS)
        df_tar["date"] = pd.to_datetime(df_tar["date"]) + pd.Series([pd.to_timedelta(delta, unit='s') for delta in self.cur_second])

        # df_dqn["date"] = df_tar["cur_second"].iloc[::2]
        df_dqn["loss"] = pd.Series(self.loss)
        df_dqn["reward"] = pd.Series(self.reward)
        df_dqn["epsilon"] = pd.Series(self.epsilon)
        df_dqn["is_end"] = self.end_epoch

        for i, t in enumerate(self.simulator.environment.targets):
            df_tar[target_aoi_columns[i]] = pd.Series(self.target_aoi[t.identifier])

        #
        # Dataset with simulation target data
        df_tar = df_tar.set_index("date")
        df_tar.to_csv(self.simulator.directory_simulation() + "log_simulation.csv")

        # Dataset with constants data
        df_const = pd.DataFrame(columns=["threshold"])
        df_const["threshold"] = pd.Series(self.targets_threshold)
        df_const.to_csv(self.simulator.directory_simulation() + "log_simulation_constants.csv")

        # DQN training data
        # df_dqn = df_dqn.set_index("date")
        df_dqn.to_csv(self.simulator.directory_simulation() + "dqn_training_data.csv")
