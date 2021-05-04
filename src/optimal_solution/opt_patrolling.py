from abc import ABCMeta, abstractmethod

import math
import pickle
import hashlib

from gurobipy import Model, GRB, quicksum
from gurobipy import *
from collections import OrderedDict
from src.utilities import utilities, config

import numpy as np

# export GUROBI_HOME="/opt/gurobi902/linux64"
# export PATH="${PATH}:${GUROBI_HOME}/bin"
# export LD_LIBRARY_PATH="${LD_LIBRARY_PATH}:${GUROBI_HOME}/lib"

# -----------------------------------------------------------------------------
#
# Abstract Class Model for patrolling
#
# -----------------------------------------------------------------------------
class AbstractPatrollingModel(metaclass=ABCMeta):

    def __init__(self, simulation, debug=True, out_path="data/opt_sol/"):
        """
        Build up the model

        :param simulation: the input simulation
        """
        self.debug = debug
        self.simulation = simulation
        self.targets = self.simulation.environment.targets
        self.ntargets = len(self.targets)
        self.ndrones = self.simulation.n_drones
        self.out_path = out_path
        self.drones = self.simulation.environment.drones
        self.base_stations = self.simulation.environment.base_stations
        self.ndepots = len(self.base_stations)
        self.M = 10000
        self.epsilon = 0.01
        self.mission_time = int(self.simulation.sim_duration_ts * self.simulation.ts_duration_sec)
        self.recharging_time = 1
        self.depots_drones = {u: 0 for u in range(self.ndrones)}  # the drone u is associated to depot: 0
        self.hovering_time = 1
        self.compute_sol()

    def __add_variables(self):
        """ Add all the needed integer variables to the problem """
        # add x^u_ij(t) variables, to move the drones among targets
        self.edge_variables = self.model.addVars([(u, i, j, t)
                                                  for u in range(self.ndrones)
                                                  for i in range(self.ntargets)
                                                  for j in range(self.ntargets)
                                                  for t in range(self.mission_time)],
                                                 vtype=GRB.BINARY, name="x_u_ij_t")

        # update weights for current drone positions
        self.target_weights = self.__target_weights()

        # add visit variables
        self.visit_variables = self.model.addVars([(u, i, t)
                                                   for u in range(self.ndrones)
                                                   for i in range(self.ntargets)
                                                   for t in range(self.mission_time)],
                                                  vtype=GRB.BINARY, name="delta^u_i(t)")

        # add drone usage variable
        self.drone_vars = self.model.addVars([u
                                              for u in range(self.ndrones)],
                                             vtype=GRB.BINARY, name="y_u")

        # deadline variables consumption
        self.deadline_vars = self.model.addVars([(i, t)
                                                 for i in range(self.ntargets)
                                                 for t in range(self.mission_time)],
                                                vtype=GRB.BINARY, name="v_i(t)")

    def __target_weights(self):
        """ compute a dictionary (i, j) with the cost in time to reach that target j from target i,

        """
        target_weights = {(i, j): utilities.euclidean_distance(self.targets[i].coords, self.targets[j].coords)
                                  / self.simulation.drone_speed_meters_sec
                          for i in range(self.ntargets)
                          for j in range(self.ntargets)}

        for i in range(self.ntargets):
            target_weights[i, i] = 1

        return target_weights

    def __add_constraints(self):
        """ Add all the needed integer variables to the problem """
        # exactly one target for drone (consistency of the solution)
        self.__coverage_constraints()

        # energy constraints (to avoid failure of drones, reaching too far points)
        self.__energy_constraints()

        # add constraints on the use of drones
        self.__drone_usage()

        # idleness constraints
        self.idleness_constraints()

        # battery constraints
        self.__energy_constraints()

    @abstractmethod
    def idleness_constraints(self):
        """ add idleness constraints """
        pass

    def __energy_constraints(self):
        """ constraint 18 """
        # add coefficient consumption
        self.model.addConstrs(
            quicksum([self.visit_variables[u, i, t + k]
                      for u in range(self.ndrones)
                      for k in range(0, self.drones[u].max_battery)
                      if t + k < self.mission_time
                      ]) >= 1
            for i in range(self.ndepots)
            for t in range(self.mission_time))

    def __drone_usage(self):
        """ Add drone usage constraints on y variables """
        self.model.addConstrs(
            self.drone_vars[u] >= self.edge_variables.sum(u, '*', '*', '*') / (self.mission_time * self.ntargets)
            for u in range(self.ndrones))

    def __coverage_constraints(self):
        """ exactly one target for drone (consistency of the solution) """
        # cover exactly one target
        self.model.addConstrs(self.visit_variables[u, i, t] <=
                              quicksum([self.edge_variables[u, i, i, k] for k in
                                        range(t - self.hovering_time)]) / self.hovering_time
                              for u in range(self.ndrones)
                              for i in range(self.ndepots, self.ntargets)  # avoid depots here
                              for t in range(self.mission_time)
                              if t > self.hovering_time)

        self.model.addConstrs(self.visit_variables[u, self.depots_drones[u], t] <=
                              quicksum([self.edge_variables[u, self.depots_drones[u], self.depots_drones[u], k] for k in
                                        range(t - self.recharging_time)]) / self.recharging_time
                              for u in range(self.ndrones)
                              for t in range(self.mission_time)
                              if t > self.hovering_time)

        self.model.addConstrs(quicksum([self.edge_variables[u, self.depots_drones[u], i, 0]
                                        for i in range(self.ntargets)]) == 1
                              for u in range(self.ndrones))

        # THIS DOES NOT WORK AS IS, THE DRONE HOVER WHERE HE LIKES, THE LAST AT DEPOT SHOULD COST 0 in THE OBJ FUN
        self.model.addConstrs(quicksum([self.edge_variables[u, i, self.depots_drones[u], self.mission_time - 1]
                                        for i in range(self.ntargets)]) == 1
                              for u in range(self.ndrones))

        # THIS IS NEEDED TO AVOID THE A SINGLE DRONE STARTS MULTIPLE TRAJECTORIES
        self.model.addConstrs(quicksum([self.edge_variables[u, i, j, 0]
                                        for i in range(self.ndepots, self.ntargets)
                                        for j in range(self.ntargets)]) <= 0
                              for u in range(self.ndrones))

        # add cyclic
        self.model.addConstrs(
            quicksum([self.edge_variables[u, i, j, math.ceil(t - self.target_weights[i, j])]
                      for i in range(self.ntargets)
                      if self.target_weights[i, j] <= t
                      ])
            ==
            quicksum([self.edge_variables[u, j, i, t]
                      for i in range(self.ntargets)])
            for j in range(self.ntargets)
            for u in range(self.ndrones)
            for t in range(self.mission_time)
            if t >= 1)

    def next_target(self, i_drone):
        """ return"""
        out_target = self.paths[i_drone][self.current_target_of_drones[i_drone]]
        self.current_target_of_drones[i_drone] += 1
        if self.current_target_of_drones[i_drone] >= len(self.paths[i_drone]):
            self.current_target_of_drones[i_drone] = len(self.paths[i_drone]) - 1
        return out_target

    def disk_filename(self):
        """ return the filename of the pickled solution for optimal model """
        unique_str = ""

        unique_str += str(self.simulation.sim_seed) + "_"
        unique_str += str(self.simulation.ts_duration_sec) + "_"
        unique_str += str(self.simulation.sim_duration_ts) + "_"
        unique_str += str(self.simulation.n_drones) + "_"
        unique_str += str(self.simulation.drone_mobility) + "_"
        unique_str += str(self.simulation.env_width_meters) + "_"
        unique_str += str(self.simulation.env_height_meters) + "_"
        unique_str += str(self.simulation.n_targets) + "_"
        unique_str += str(self.simulation.n_obstacles) + "_"
        unique_str += str(self.simulation.drone_speed_meters_sec) + "_"
        unique_str += str(self.simulation.drone_max_battery) + "_"
        unique_str += str(self.simulation.drone_max_buffer) + "_"
        unique_str += str(self.simulation.drone_com_range_meters) + "_"
        unique_str += str(self.simulation.drone_sen_range_meters) + "_"
        unique_str += str(self.simulation.drone_radar_range_meters) + "_"
        unique_str += str(self.simulation.bs_com_range_meters) + "_"
        unique_str += str(self.simulation.bs_coords) + "_"

        unique_str += ",".join([str(t) for t in self.targets]) + "_"
        unique_str += str(self.ntargets) + "_"
        unique_str += str(self.mission_time) + "_"
        unique_str += str(self.ndepots) + "_"
        unique_str += str(self.recharging_time) + "_"
        unique_str += str(self.hovering_time) + "_"

        # Compute key
        out_filename = hashlib.sha256(unique_str.encode('utf-8')).hexdigest()
        return str(out_filename)

    def load_disk_solution(self):
        """ Try to load the solution (already computed) from disk.
            If the solution does not exists, return None,
            otherwise return a distionary with all the trajectories for drones
        """
        try:
            with open(self.out_path + self.disk_filename() + '_.pickle', 'rb') as f:
                path_data = pickle.load(f)

            return path_data
        except Exception as e:
            print(e)
            return None

    def save_disk_solution(self):
        """
        Save a distionary with all the trajectories for drones to disck
        """
        with open(self.out_path + self.disk_filename() + '_.pickle', 'wb') as f:
            pickle.dump(self.paths, f)

    def compute_sol(self):
        """ Compute the optimal solution for the problem """
        self.current_target_of_drones = {i_dr: self.depots_drones[i_dr]
                                            for i_dr in range(self.ndrones)}

        # IF SOLUTION EXIST ON DISK ---> LOAD
        self.paths = self.load_disk_solution()
        if self.paths is None:
            self.model = Model('Patrolling')
            #self.model.setParam('OutputFlag', 0)
            self.__add_variables()
            self.__add_constraints()
            self.objective_function()
            self.model.optimize()
            if self.model.getAttr('Status') == GRB.OPTIMAL:
                self.paths = self.extract_solution()
                self.save_disk_solution()
            else:
                print("No optimal solution found")
                exit(1)

    @abstractmethod
    def objective_function(self):
        """ objective function of opt model """
        pass

    def extract_solution(self):
        ''' extract the values from variables
            and the tours produced by the optimization
        '''
        self.strsolution = ""
        for variable in self.model.getVars():
            self.strsolution += (str(variable.varName)
                                 + " - "
                                 + str(variable.x)
                                 + "\n")  # x -> results of optimization
        self.strsolution += "Objetive function value: "
        self.strsolution += str(self.model.objVal)

        full_drones_trajectories = {}
        for u in range(self.ndrones):  # drones
            trajectory = []
            for t in range(self.mission_time):  # mission time
                for i in range(self.ntargets):  # targets
                    for j in range(self.ntargets):  # targets
                        if (self.model.getVarByName('x_u_ij_t[{},{},{},{}]'.format(u, i, j, t)).X >= 0.5):
                            trajectory.append((i, j))
                            if self.debug:
                                print("dr:", u, "start:", i, "end:", j, "at time:", t)
            full_drones_trajectories[u] = trajectory

        if self.debug:
            for i, r in self.target_weights.items():
                print("Edge:", i, "Cost:", int(r))

        # COMPUTE SOLUTIONS FOR THE SIMULATOR
        paths = {}
        for i_dr in range(self.ndrones):
            full_traj = np.asarray([j for i,j in full_drones_trajectories[i_dr]])
            full_traj = full_traj[np.insert(np.diff(full_traj).astype(np.bool), 0, True)]
            # 0, 0,0 ,0,0 ,0 ,0, 1,1 ,1, 1, 1, 2,3,3,4,4. 0, 0,0
            ordered_targets = list(full_traj)
            if ordered_targets == []:
                paths[i_dr] = [self.depots_drones[i_dr], self.depots_drones[i_dr]]
            else:
                paths[i_dr] = [self.targets[i] for i in ordered_targets]

        return paths


""" THe following patrolling optimal model aims at finding a solution where all the deadlines should be respected """
class HardConstraintsModel(AbstractPatrollingModel):


    def objective_function(self):
        """ objective function of opt model """
        self.model.setObjective(self.drone_vars.sum('*') + self.epsilon * quicksum([self.edge_variables[u, j, i, t]
                                                                                    for i in range(self.ntargets)
                                                                                    for j in range(self.ntargets)
                                                                                    for u in range(self.ndrones)
                                                                                    for t in range(self.mission_time)
                                                                                    if not (i == j and i < self.ndepots)
                                                                                    # stay at depot does not consume
                                                                                    # energy
                                                                                    ]),
                                GRB.MINIMIZE)

    def idleness_constraints(self):
        """ add idleness constraints """
        self.model.addConstrs(
            quicksum([self.visit_variables[u, i, t + k]
                      for u in range(self.ndrones)
                      for k in range(0, self.targets[i].maximum_tolerated_idleness)
                      if t + k < self.mission_time
                      ]) >= 1
            for i in range(self.ndepots, self.ntargets)
            for t in range(self.mission_time))

    def disk_filename(self):
        outfname = super().disk_filename()
        return "hard_constraint_model_" + outfname


""" THe following patrolling optimal model aims at finding a solution where most of the deadlines are respected """
class SoftConstraintsModel(AbstractPatrollingModel):


    def objective_function(self):
        """ objective function of opt model """
        # read this -> https://math.stackexchange.com/questions/2732897/linear-integer-programming-count-consecutive-ones
        self.model.setObjective(self.deadline_vars.sum('*') + self.epsilon * quicksum([self.edge_variables[u, j, i, t]
                                                                                    for i in range(self.ntargets)
                                                                                    for j in range(self.ntargets)
                                                                                    for u in range(self.ndrones)
                                                                                    for t in range(self.mission_time)
                                                                                    if not (i == j and i < self.ndepots)
                                                                                    # stay at depot does not consume
                                                                                    # energy
                                                                                    ]),
                                GRB.MINIMIZE)

    def idleness_constraints(self):
        """ add idleness constraints """
        self.model.addConstrs(
            self.deadline_vars[i, t] >= 1 - quicksum([self.visit_variables[u, i, t + k]
                                                      for u in range(self.ndrones)
                                                      for k in range(0, self.targets[i].maximum_tolerated_idleness)
                                                      if t + k < self.mission_time
                                                      ])
                        for i in range(self.ndepots, self.ntargets)
                        for t in range(self.mission_time))

    def disk_filename(self):
        outfname = super().disk_filename()
        return "soft_constraint_model_" + outfname

