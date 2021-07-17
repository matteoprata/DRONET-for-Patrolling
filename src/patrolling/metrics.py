import numpy as np
import os
from collections import defaultdict
import pandas as pd


class Metrics:
    def __init__(self, simulator=None):
        self.simulator = simulator

        self.time = []
        self.target = []
        self.idleness = []
        self.violation_factor = []

    def append_statistics_on_target_reached(self, time, drone, target):
        """ Store info what drone visited what target and when. """
        if not target.is_depot and target.active:
            self.time.append(time)
            self.target.append(target.identifier)
            self.idleness.append(target.age_of_information(drone_id=drone if drone is not None else 0))
            self.violation_factor.append(target.aoi_idleness_ratio(drone_id=drone if drone is not None else 0))

    def save_dataframe(self):
        df_dqn = pd.DataFrame()

        df_dqn["time"] = self.time
        df_dqn["target"] = self.target
        df_dqn["idleness"] = self.idleness
        df_dqn["violation_factor"] = self.violation_factor

        print("Saving @")
        print(self.simulator.directory_simulation() + "dqn_training_data.csv")
        df_dqn.to_csv(self.simulator.directory_simulation() + "dqn_training_data.csv", mode='a', header=False)

        print()
        print("final stats:")
        print("idleness:", df_dqn["idleness"].mean(), df_dqn["idleness"].max())
        print("violation_factor:", df_dqn["violation_factor"].mean(), df_dqn["violation_factor"].max())
