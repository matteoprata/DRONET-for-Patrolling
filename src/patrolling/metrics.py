import numpy as np
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
        self.is_final = []
        self.is_new_epoch = []

        self.Q_vectors = defaultdict(lambda: (0, None))
        self.N_ACTIONS = 0
        self.N_FEATURES = 0

    def append_statistics_on_target_reached(self, action, learning_tuple=None):
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

        if learning_tuple is not None:
            reward, epsilon, loss, end_epoch, s, q, was_new_epoch = learning_tuple
            # if reward is not None and epsilon is not None and loss is not None and end_epoch is not None:
            self.reward.append(reward)
            self.epsilon.append(epsilon)
            self.loss.append(loss)
            self.is_final.append(1 if end_epoch else None)
            self.is_new_epoch.append(1 if was_new_epoch else None)

            if len(self.targets_threshold.keys()) == 0:
                self.targets_threshold = {t.identifier: t.maximum_tolerated_idleness for t in self.simulator.environment.targets}

            if not (s is None or q is None):
                s = tuple(s.vector())
                self.Q_vectors[s] = (self.Q_vectors[s][0]+1, list(q))   # add occurrence of that state

    def append_statistics_on_target_reached_light(self, learning_tuple=None):
        """ Saves the current second, aoi, reward, epsilon, loss. For computing final plots. """

        reward, epsilon, loss, end_epoch, s, q, was_new_epoch = learning_tuple
        # if reward is not None and epsilon is not None and loss is not None and end_epoch is not None:
        self.reward.append(reward)
        self.epsilon.append(epsilon)
        self.loss.append(loss)
        self.is_final.append(1 if end_epoch else None)
        self.is_new_epoch.append(1 if was_new_epoch else None)

    def save_dataframe_light(self):
        has_header = not os.path.isfile(self.simulator.directory_simulation() + "dqn_training_data.csv")

        # ---- DF 1 ) DQN LEARNING PARAMETERS
        df_dqn = pd.DataFrame()

        df_dqn["loss"] = self.loss
        df_dqn["reward"] = self.reward
        df_dqn["epsilon"] = self.epsilon
        df_dqn["is_end"] = self.is_final
        df_dqn["is_new_epoch"] = self.is_new_epoch

        df_dqn.to_csv(self.simulator.directory_simulation() + "dqn_training_data.csv", mode='a', header=has_header)

        # Resetting the data structure
        self.reset_data_structures()

    def save_dataframe(self):
        """ Saves data to disk. """

        has_header = not os.path.isfile(self.simulator.directory_simulation() + "dqn_training_data.csv")

        # ---- DF 1 ) DQN LEARNING PARAMETERS
        df_dqn = pd.DataFrame()

        df_dqn["loss"] = self.loss
        df_dqn["reward"] = self.reward
        df_dqn["epsilon"] = self.epsilon
        df_dqn["is_end"] = self.is_final
        df_dqn["is_new_epoch"] = self.is_new_epoch

        df_dqn.to_csv(self.simulator.directory_simulation() + "dqn_training_data.csv", mode='a', header=has_header)

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

        # ---- DF 3 ) CONSTANTS

        # Dataset with constants data
        df_const = pd.DataFrame()
        df_const["threshold"] = pd.Series(self.targets_threshold)
        df_const.to_csv(self.simulator.directory_simulation() + "log_simulation_constants.csv")

        # ---- DF 4 ) Q VALUES

        df_qvalues = pd.DataFrame()
        STATES = np.asarray(list(self.Q_vectors.keys()))
        VALUES = np.asarray([q for _, q in self.Q_vectors.values()])
        COUNT = np.asarray([count for count, _ in self.Q_vectors.values()])

        for i in range(STATES[0].shape[0]):
            df_qvalues["s"+str(i)] = STATES[:,i]

        for i in range(VALUES[0].shape[0]):
            df_qvalues["q"+str(i)] = VALUES[:,i]
        df_qvalues["count"] = COUNT

        if has_header:
            df_qvalues.to_csv(self.simulator.directory_simulation() + "qvalues.csv")
        else:
            # MERGE
            df_qvalues_disk = pd.read_csv(self.simulator.directory_simulation() + "qvalues.csv", index_col=0)
            key_columns = list(df_qvalues_disk.columns[:self.N_FEATURES])
            other_columns = list(df_qvalues_disk.columns[self.N_FEATURES:])
            other_columns_x = [el + "_x" for el in other_columns]
            other_columns_y = [el + "_y" for el in other_columns]

            merged = df_qvalues_disk.merge(df_qvalues, on=key_columns, how='outer')
            NEW = ~merged[other_columns_y].isnull().all(1)

            # increase counter
            merged.loc[NEW, other_columns_x[-1]] = merged.loc[NEW, other_columns_x[-1]].fillna(0)
            merged.loc[NEW, other_columns_x[-1]] += merged.loc[NEW, other_columns_y[-1]]

            # replace new qvalues
            merged.loc[NEW, other_columns_x[:-1]] = merged.loc[NEW, other_columns_y[:-1]].values
            merged = merged.drop(other_columns_y, axis=1)
            merged = merged.rename(columns={other_columns_x[i]:other_columns[i] for i in range(len(other_columns_x))})

            merged.to_csv(self.simulator.directory_simulation() + "qvalues.csv")

        # Resetting the data structure
        self.reset_data_structures()

    def reset_data_structures(self):

        self.target_aoi = defaultdict(list)
        self.cur_second = []

        self.reward = []
        self.epsilon = []
        self.loss = []
        self.is_final = []
        self.is_new_epoch = []

        self.Q_vectors = defaultdict(lambda: (0, None))