import numpy as np
import os
from collections import defaultdict
import pandas as pd
import matplotlib.pyplot as plt

class Metrics:
    def __init__(self, simulator=None):
        self.simulator = simulator

        # self.time = []
        self.target_ids = []
        self.idleness = []
        self.violation_factor = []

    # def append_statistics_on_target_reached(self, time, drone, target):
    #     """ Store info what drone visited what target and when. """
    #     if not target.is_depot and target.active:
    #         self.time.append(time)
    #         self.target_ids.append(target.identifier)
    #         self.idleness.append(target.age_of_information(drone_id=drone if drone is not None else 0))
    #         self.violation_factor.append(target.aoi_idleness_ratio(drone_id=drone if drone is not None else 0))

    def append_statistics(self):
        if self.simulator.wandb is None:
            for t in self.simulator.environment.targets:
                if not t.is_depot:
                    self.idleness.append(t.age_of_information(drone_id=0))
                    self.violation_factor.append(t.aoi_idleness_ratio(drone_id=0))

    def save_dataframe(self):
        if self.simulator.wandb is None:
            df_dqn = pd.DataFrame()

            idleness_key = "idleness-{}".format(self.simulator.drone_mobility.value)
            violation_factor_key = "violation_factor-{}".format(self.simulator.drone_mobility.value)

            df_dqn[idleness_key] = self.idleness
            df_dqn[violation_factor_key] = self.violation_factor

            print("Saving @")
            path = self.simulator.directory_simulation() + self.simulator.experiment_name()
            print(path + "-dqn_training_data.csv")
            is_ex = os.path.exists(path + "-dqn_training_data.csv")
            df_dqn.to_csv(path + "-dqn_training_data.csv", mode='w' if is_ex else 'a', header=True)

            print()
            print("final stats:")
            print("idleness:", df_dqn[idleness_key].mean(), df_dqn[idleness_key].max())
            print("violation_factor:", df_dqn[violation_factor_key].mean(), df_dqn[violation_factor_key].max())

            # df_dqn.boxplot(column=[idleness_key], showfliers=False)
            # plt.savefig(path + "-dqn_training_data_idleness.png")
            # plt.clf()
            #
            # df_dqn.boxplot(column=[violation_factor_key], showfliers=False)
            # plt.savefig(path + "-dqn_training_data_violation_factor.png")

    def join_metrics(self, path):
        df_final = None
        modalities = []
        for filename in os.listdir(path):
            if filename.endswith("dqn_training_data.csv"):
                modalities.append(filename.split("dqn_training_data.csv")[0].split("-")[-2])
                df_temp = pd.read_csv(path + filename)
                if df_final is None:
                    df_final = df_temp.copy()
                else:
                    df_final = pd.concat([df_temp, df_final], axis=1, join="inner")

        cols_remove = list(df_final.columns)[0::3]
        df_final = df_final.drop(columns=cols_remove)
        # print(cols_remove)
        # print(df_final.columns)
        cols = df_final.columns

        fig1, ax1 = plt.subplots()
        ax1.set_title('Idleness')
        IDLENESS = np.array([[float(i) for i in df_final.iloc[:, j].to_numpy()]
                                       for j in range(0, len(cols), 2)]).T
        ax1.boxplot(IDLENESS, showfliers=False)
        plt.xlabel("modalities")
        plt.ylabel("seconds")
        plt.xticks(list(range(1, len(modalities)+1)), modalities)
        plt.savefig(path + "idleness.png")
        plt.clf()

        fig1, ax1 = plt.subplots()
        ax1.set_title('Violation Factor')
        REQ = np.array([[float(i) for i in df_final.iloc[:, j].to_numpy()]
                             for j in range(1, len(cols), 2)]).T
        ax1.boxplot(REQ, showfliers=False)
        plt.xlabel("modalities")
        plt.ylabel("factor")
        plt.xticks(list(range(1, len(modalities)+1)), modalities)
        plt.savefig(path + "violation.png")
        plt.clf()

        fig1, ax1 = plt.subplots()
        ax1.set_title('Number Violations')
        n_seconds = df_final.shape[0]

        VIOLS = []
        for j in range(1, len(cols), 2):
            viol = 0
            for ta in range(self.simulator.n_targets):
                viol += sum(df_final.iloc[::self.simulator.n_targets + ta, j] >= 1)
            VIOLS.append(viol / n_seconds)

        ax1.bar(modalities, VIOLS)
        plt.xlabel("modalities")
        plt.ylabel("number of violations")
        plt.xticks(list(range(0, len(modalities))), modalities)
        plt.savefig(path + "violations.png")
        plt.clf()

