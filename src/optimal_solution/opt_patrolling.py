from abc import ABCMeta, abstractmethod
from deprecated import deprecated

import networkx as nx
import numpy as np
from gurobipy import Model, GRB, quicksum
from gurobipy import *

import math
from src.utilities import utilities, config

#export GUROBI_HOME="/opt/gurobi902/linux64"
#export PATH="${PATH}:${GUROBI_HOME}/bin"
#export LD_LIBRARY_PATH="${LD_LIBRARY_PATH}:${GUROBI_HOME}/lib"

# -----------------------------------------------------------------------------
#
# Abstract Class Model for patrolling
#
# -----------------------------------------------------------------------------
class AbstractConnectivityModel():

    def __init__(self, simulation):
        """
        Build up the model

        :param simulation: the input simulation
        """
        self.simulation = simulation
        self.targets = self.simulation.environment.targets
        self.ntargets = len(self.targets)
        self.ndrones = self.simulation.n_drones
        self.drones = self.simulation.environment.drones
        self.base_stations = self.simulation.environment.base_stations
        self.ndepots = len(self.base_stations)
        self.M = 10000
        self.epsilon = 0.001
        self.mission_time = int(self.simulation.sim_duration_ts * self.simulation.ts_duration_sec)
        self.recharging_time = 1
        self.depots_drones = {u : 0 for u in range(self.ndrones)}  # the drone u is associated to depot: 0
        self.hovering_time = 1
        self.compute_sol()


    def __add_variables(self):
        """ Add all the needed integer variables to the problem """
        # add x^u_ij(t) variables, to move the drones among targets
        self.edge_variables = self.model.addVars([(u, i, j ,t)
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
                                              vtype=GRB.CONTINUOUS, name="y_u")

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
        self.__idleness_constraints()

        # battery constraints
        self.__energy_constraints()

    def __idleness_constraints(self):
        """ add idleness constraints """
        self.model.addConstrs(
            quicksum([self.visit_variables[u, i, t + k]
                      for u in range(self.ndrones)
                      for k in range(0, self.targets[i].maximum_tolerated_idleness)
                      if t + k < self.mission_time
                      ]) >= 1
                 for i in range(self.ndepots, self.ntargets)
                 for t in range(self.mission_time))

    def __energy_constraints(self):
        """ constraint 18 """
        # add coefficient consumption
        self.model.addConstrs(
            quicksum([self.visit_variables[u, i, t + k]
                      for u in range(self.ndrones)
                      for k in range(0, self.recharging_time)
                      if t + k < self.mission_time
                      ]) >= 1
                 for i in range(self.ndepots)
                 for t in range(self.mission_time))


    def __drone_usage(self):
        """ Add drone usage constraints on y variables """
        self.model.addConstrs(self.drone_vars[u] >= self.edge_variables.sum(u, '*', '*', '*') / (self.mission_time*self.ntargets)
                            for u in range(self.ndrones))


    def __coverage_constraints(self):
        """ exactly one target for drone (consistency of the solution) """
        # cover exactly one target
        self.model.addConstrs(self.visit_variables[u, i, t] <=
                                    quicksum([self.edge_variables[u, i, i, k] for k in range(t - self.hovering_time)]) / self.hovering_time
                            for u in range(self.ndrones)
                            for i in range(self.ndepots, self.ntargets)  # avoid depots here
                            for t in range(self.mission_time)
                                if t > self.hovering_time)

        self.model.addConstrs(self.visit_variables[u, self.depots_drones[u], t] <=
                                    quicksum([self.edge_variables[u, self.depots_drones[u], self.depots_drones[u], k] for k in range(t - self.recharging_time)]) / self.recharging_time
                            for u in range(self.ndrones)
                            for t in range(self.mission_time)
                                if t > self.hovering_time)

        self.model.addConstrs(quicksum([self.edge_variables[u, self.depots_drones[u], i, 0]
                                for i in range(self.ntargets)]) == 1
                        for u in range(self.ndrones))

        self.model.addConstrs(quicksum([self.edge_variables[u, i, i, 0]
                                for i in range(self.ndepots, self.ntargets)]) <= 0
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
            for t in range(self.mission_time))

        #self.model.addConstrs(
        #        quicksum([self.edge_variables[u, i, j, t]
        #                  for i in range(self.ntargets)
        #                  if self.target_weights[i, j] > t
        #                  ]) == 0
        #    for j in range(self.ntargets)
        #    for u in range(self.ndrones)
        #    for t in range(self.mission_time))

    def compute_trial(self):
        """ compute a solution """
        self.paths = {}
        self.current_target_of_drones = {i_dr : 0 for i_dr in range(self.ndrones)}
        for i_dr in range(self.ndrones):
            targets = list(self.targets)
            import random
            random.shuffle(targets)
            self.paths[i_dr] = targets

    def next_target(self, i_drone):
        """ return"""
        out_target = self.paths[i_drone][self.current_target_of_drones[i_drone]]
        self.current_target_of_drones[i_drone] += 1
        if self.current_target_of_drones[i_drone] >= len(self.targets):
            self.current_target_of_drones[i_drone] = 0
        return out_target

    def compute_sol(self):
        # IF SOLUTION EXIST ON DISK ---> LOAD 
        #ELSE:
            # COMPUTE
            # SAVE ON DISK 
            
        self.model = Model('Patrolling')
        #self.model.setParam('OutputFlag', 0)
        self.__add_variables()
        self.__add_constraints()
        self.objective_function()
        self.model.optimize()
        sol = None
        if self.model.getAttr('Status') == GRB.OPTIMAL:
            sol = self.extract_solution()
        #self.model.reset(clearall=1)
        return sol

    def objective_function(self):
        """ objective function of opt model """
        self.model.setObjective(self.drone_vars.sum('*') + self.epsilon*self.edge_variables.sum('*', '*', '*', '*'),
                                GRB.MINIMIZE)

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
        #print(self.strsolution)
        #exit()
        drones_trajectories = {}
        for u in range(self.ndrones):  # drones
            trajectory = []
            for t in range(self.mission_time):  # mission time
                for i in range(self.ntargets):  # targets
                    for j in range(self.ntargets):  # targets
                        if (self.model.getVarByName('x_u_ij_t[{},{},{},{}]'.format(u, i, j, t)).X >= 0.5):
                            print(u,i,j,t)
                            trajectory.append((i, j))
            drones_trajectories[u] = trajectory
        print(drones_trajectories)
        exit()
 #       return drones_to_points
