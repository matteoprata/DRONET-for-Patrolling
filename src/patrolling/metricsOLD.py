import numpy as np
import os
from collections import defaultdict
import pandas as pd


class MetricsOLD:
    def __init__(self, simulator=None):
        self.simulator = simulator
        self.targets_threshold = {}

        self.target_aoi = defaultdict(list)
        self.cur_second = []

    def append_statistics_on_target_reached(self, action):
        """ Saves the current second, aoi, reward, epsilon, loss. For computing final plots. """

        for rep in [-1, 0]:
            self.cur_second.append(self.simulator.current_second(rep, cur_second_tot=True))
            for t in self.simulator.environment.targets:
                if rep == -1:
                    if t.identifier == action:
                        # TODO FIX
                        self.target_aoi[t.identifier].append(t.age_of_information(rep))
                    else:
                        self.target_aoi[t.identifier].append(None)
                else:
                    if t.identifier == action:
                        self.target_aoi[t.identifier].append(0)
                    else:
                        # TODO FIX
                        self.target_aoi[t.identifier].append(t.age_of_information())

    def save_dataframe(self):
        """ Saves data to disk. """

        has_header = not os.path.isfile(self.simulator.directory_simulation() + "dqn_training_data.csv")

        # ---- DF 2 ) TARGET VISITS

        N_ROWS = len(self.cur_second)
        target_aoi_columns = ["target_{}_aoi".format(t.identifier) for t in self.simulator.environment.targets]

        df_tar = pd.DataFrame()

        df_tar["cur_second"] = pd.Series(self.cur_second)
        df_tar["date"] = pd.Series([pd.to_datetime('now').normalize().strftime("%d-%m-%Y %H:%M:%S")] * N_ROWS)
        df_tar["date"] = pd.to_datetime(df_tar["date"]) + pd.Series([pd.to_timedelta(delta, unit='s') for delta in self.cur_second])

        for i, t in enumerate(self.simulator.environment.targets):
            df_tar[target_aoi_columns[i]] = pd.Series(self.target_aoi[t.identifier])

        # Dataset with simulation target data
        df_tar = df_tar.set_index("date")
        df_tar.to_csv(self.simulator.directory_simulation() + "log_simulation.csv", mode='a', header=has_header)

        # Resetting the data structure
        self.reset_data_structures()

    def reset_data_structures(self):

        self.target_aoi = defaultdict(list)
        self.cur_second = []


