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

        self.target_aoi = defaultdict(list)
        self.cur_second = []

        self.reward = []
        self.epsilon = []
        self.loss = []

    def append_statistics_on_target_reached(self, reward=None, epsilon=None, loss=None):
        """ Saves the current second, aoi, reward, epsilon, loss. For computing final plots. """

        self.cur_second.append(self.simulator.simulated_current_second())

        for t in self.simulator.environment.targets:
            self.target_aoi[t.identifier].append(t.age_of_information())

        if reward is not None and epsilon is not None and loss is not None:
            self.reward.append(reward)
            self.epsilon.append(epsilon)
            self.loss.append(loss)

        if self.targets_threshold is None:
            self.targets_threshold = {t.identifier: t.maximum_tolerated_idleness for t in self.simulator.environment.targets}

    def save_dataframe(self):
        """ Saves data to disk. """

        N_ROWS = len(self.cur_second)
        target_aoi_columns = ["target_{}_aoi".format(t.indtieifer) for t in self.simulator.environment.targets]

        df = pd.DataFrame(columns=["date", "cur_second"] + target_aoi_columns + ["loss", "reward", "epsilon"])

        df["cur_second"] = pd.Series([pd.to_datetime('now').normalize().strftime("%d-%m-%Y %H:%M:%S")]*N_ROWS)
        df["cur_second"] = pd.to_datetime(df["cur_second"])
        df["cur_second"] = pd.to_datetime(df["cur_second"]) + pd.Series([pd.to_timedelta(delta, unit='s') for delta in self.cur_second])

        df = df.set_index("cur_second")
        df["loss"] = pd.Series(self.loss)
        df["reward"] = pd.Series(self.reward)
        df["epsilon"] = pd.Series(self.epsilon)

        for t in self.simulator.environment.targets:
            df["target_{}_aoi".format(t.indtieifer)] = pd.Series(self.target_aoi[t.indtieifer])

        df.to_csv(self.simulator.directory_simulation() + "log_simulation.csv")
