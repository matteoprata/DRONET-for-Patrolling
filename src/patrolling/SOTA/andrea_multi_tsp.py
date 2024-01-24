from abc import ABCMeta, abstractmethod

import numpy as np
import random
import matplotlib.pyplot as plt


from src.patrolling.SOTA.util import (rndgraph, utility)
import src.patrolling.SOTA.multi_path_generator as multi_path_generator
import networkx as nx
from src.utilities.utilities import euclidean_distance
# -----------------------------------------------------------------
#
# Abstract Class for Greedy path coverage
#
# -----------------------------------------------------------------
class AbstractGreedyCoverage():
    ''' Abstract class for greedy algorithm for coverage
        path based problem with drones and multiple tours/rounds
    '''
    __metaclass__ = ABCMeta

    def __init__(self, graph, uavs_tours, roots, uavs, max_rounds):
        """
            Constructor for the Path Coverage, input the graph that
            must be covered, the list of uavs that will be used, for each
            uav the respective root will be in the roots list and
            the feasible tours in the list_tours: the i-th uav (uavs[i])
            will have uavs_tours[i] feasbile tours and the root in roots[i].
            max_rounds indicates the max number of rounds for the graph cover.

            the tours in uavs_tours[i] must be instance of class utility.Tour
            the uavs must be istance of class utility.Drone
        """
        assert (len(uavs_tours) == len(roots)
                and len(roots) == len(uavs))
        self.graph = graph
        self.uavs_tours = uavs_tours
        self.roots = roots
        self.uavs = uavs
        self.N = max_rounds
        self.nuavs = len(uavs)
        self.points = set(graph.nodes()) - set(roots)  # to visit
        self.visited_points = set()
        self.set_reachable_points = self.reachable_points()
        # indicates for each uav how much tours it still has unassign
        self.residual_ntours_to_assign = [max_rounds] * self.nuavs
        # indicates how much tours has still to be unassign
        self.tot_count_unassign_tours = max_rounds * self.nuavs
        self.solution = utility.MultiRoundSolution(graph,
                                                   roots, uavs)  # dict {round : tours}

    def reachable_points(self):
        """  return a set of all points
             that are reachable by the input tours
        """
        r_points = set()
        for u in range(self.nuavs):
            uuav_tours = self.uavs_tours[u]
            for t in uuav_tours:
                r_points |= t.nodes
        r_points -= set(self.roots)
        return r_points

    @abstractmethod
    def local_optimal_choice(self):
        """ return a tuple (index_uav, index_tour)
            that is optimal in this step
        """
        pass

    def run(self):
        """ run the algorithm until the stop
            condition is reached
        """
        while not self.greedy_stop_condition():
            itd_uav, ind_tour = self.local_optimal_choice()
            self.residual_ntours_to_assign[itd_uav] -= 1
            self.tot_count_unassign_tours -= 1
            opt_tour = self.uavs_tours[itd_uav][ind_tour]
            # no remove of ind_tour (make sense?)
            self.visited_points |= \
                self.points.intersection(opt_tour.nodes)  # update visited points
            self.solution.append_tour(itd_uav, opt_tour)

    def greedy_stop_condition(self):
        """ stop condition of greedy algorithm """
        return (self.tot_count_unassign_tours == 0 or
                len(self.set_reachable_points - self.visited_points) == 0)


# -------------------------------------------------------------------
#
# Cumulative Greedy Coverage Class for path coverage
#
# -------------------------------------------------------------------
class CumulativeGreedyCoverage(AbstractGreedyCoverage):
    ''' Cumulative Greedy Coverage class for cumulative coverage
        path based problem with drones and multiple tours/rounds
    '''

    def local_optimal_choice(self):
        choice_dict = {}
        for ind_uav in range(self.nuavs):
            uav_residual_rounds = self.residual_ntours_to_assign[ind_uav]
            if uav_residual_rounds > 0:
                uav_tours = self.uavs_tours[ind_uav]
                for ind_tour in range(len(uav_tours)):
                    tour = uav_tours[ind_tour]
                    q_tour = self.evaluate_tour(tour, uav_residual_rounds)
                    choice_dict[q_tour] = (ind_uav, ind_tour)

        best_value = max(choice_dict, key=int)
        return choice_dict[best_value]

    def evaluate_tour(self, tour, round_count):
        """ measure of quality of round,
            in terms of cumulative coverage
        """
        new_points = (tour.nodes - self.visited_points) - set(self.roots)
        return round_count * len(new_points)

    # -------------------------------------------------------------------


#
# Simple Greedy Coverage Class for path coverage
#
# -------------------------------------------------------------------
class SimpleGreedyCoverage(AbstractGreedyCoverage):
    ''' Simple Greedy Coverage class for simple coverage
        path based problem with drones and multiple tours/rounds
    '''

    def local_optimal_choice(self):
        choice_dict = {}
        for ind_uav in range(self.nuavs):
            uav_residual_rounds = self.residual_ntours_to_assign[ind_uav]
            if uav_residual_rounds > 0:
                uav_tours = self.uavs_tours[ind_uav]
                for ind_tour in range(len(uav_tours)):
                    tour = uav_tours[ind_tour]
                    q_tour = self.evaluate_tour(tour)
                    choice_dict[q_tour] = (ind_uav, ind_tour)

        best_value = max(choice_dict, key=int)
        return choice_dict[best_value]

    def evaluate_tour(self, tour):
        """ measure of quality of round,
            in terms of cumulative coverage
        """
        new_points = (tour.nodes - self.visited_points) - set(self.roots)
        return len(new_points)


def procedure_INFOCOM(G, battery, nuavs, depot=0):

    utility.save_graph(G)
    tours = multi_path_generator.multipath_subroutine(G, depot, battery)

    # print("DEBUG")
    nodes = []
    for t in tours:
        t.depot = depot
        # print("Edges:", t.edges)
        nodes += t.nodes
    # print("Visitable nnodes:", len(set(nodes)))

    # print("Cumulative Coverage")
    model = CumulativeGreedyCoverage(G, [tours] * nuavs, [0] * nuavs, list(range(nuavs)), 20)
    model.run()
    solution = model.solution
    graph_s = solution.graph

    """
    solution.append_tour(0, utility.Tour(G, solution.tours_at_round(0)[0].edges, depot))
    solution.tours[1] = []
    solution.drones = [1, 2]
    solution.roots.append(2)
    solution.append_tour(1, utility.Tour(G, solution.tours_at_round(0)[0].edges, depot))
    """

    nodes = []
    VISISTED_IPS = set()
    i = 0
    SOL = None
    ALL_ROUNDS = []
    while SOL != [[] * nuavs] * nuavs:
        SOL = [t.edges for t in solution.tours_at_round(i)]
        VISISTED_IPS |= {edge[0] for s in SOL for edge in s} | {edge[1] for s in SOL for edge in s}
        # print("Round:", i, "tour:", [t.edges for t in solution.tours_at_round(i)])
        if SOL != [[] * nuavs] * nuavs:
            ALL_ROUNDS.append(SOL)
            i += 1

    # print(VISISTED_IPS)
    # print("Cumulative Visited nnodes:", len(set(nodes)))
    # solution.plot()

    # print("Simple Model")
    # model = SimpleGreedyCoverage(G, [tours, tours], [9, 9], [1,2], 20)
    # model.run()
    # solution = model.solution
    # for i in range(20):
    #     print("Round:", i, "tour:", [t.edges for t in solution.tours_at_round(i)])
    # for i in range(20):
    #     for t in solution.tours_at_round(i):
    #         nodes += t.nodes
    # print("Simple Visited nnodes:", len(set(nodes)))
    # solution.plot()

    return VISISTED_IPS, ALL_ROUNDS


def compute_from_coordinates(coordinates, weight_node=0):
    graph = nx.Graph(node_weight=weight_node)
    for i, coo in enumerate(coordinates):
        graph.add_nodes_from([(i, {"x": coo[0], "y": coo[1], "pos": coo, "weight": weight_node})])

    for i, coo1 in enumerate(coordinates):
        for j, coo2 in enumerate(coordinates):
            if i > j:
                graph.add_edges_from([(i, j, {"weight": euclidean_distance(coo1, coo2)})])
    return graph


def plan_builder(G, n_drones, n_tars):
    depot = 0
    bss = list(range(0, 6000, 300))
    n_dr = []
    n_vis = []
    ALL_SOLS = []
    for bat in bss:
        a, b = procedure_INFOCOM(G, bat, n_drones, depot)
        ALL_SOLS.append(b)
        s = (np.array([[len(l) for l in round] for round in b]) > 1) * 1
        s = (np.sum(s, axis=0) > 0) * 1
        n_used_dr = np.sum(s)
        n_dr.append(n_used_dr / n_drones)
        n_vis.append(len(a) / n_tars)
    n_dr, n_vis = np.array(n_dr), np.array(n_vis)

    # assert np.max(n_dr) == 1, f"max n_drones was {np.max(n_dr)}"
    # plt.plot(bss, n_dr, label="DRONES")
    # plt.plot(bss, n_vis, label="VISITS")
    #
    # plt.legend()
    # plt.show()

    n_vis[n_dr != np.max(n_dr)] = -np.inf
    max_ind = np.argmax(n_vis)

    # Print Pareto optimal points
    print("Pareto Optimal Points:")
    print(f"Index: {max_ind} so {bss[max_ind]}, Value for Curve1: {n_vis[max_ind]}, Value for Curve2: {n_dr[max_ind]}")

    rounds = ALL_SOLS[max_ind]
    # print(rounds)

    plan = {}
    for dr in range(n_drones):
        tour_drone = []
        for ir, r in enumerate(rounds):
            tour = r[dr]
            if len(tour) > 0:
                # print(tour)
                if ir == 0:
                    tour_drone.append(tour[0][0])
                for e1, e2 in tour:
                    tour_drone.append(e2)
        if len(tour_drone) > 0:
            plan[dr] = tour_drone
        else:
            plan[dr] = [0]
    return plan


from src.patrolling.meta_patrolling import PrecomputedPolicy


class Bartolini(PrecomputedPolicy):
    name = "GaP"
    identifier = 0
    line_tick = 10
    marker = 10

    def __init__(self, set_drones, set_targets):
        super().__init__(set_drones=set_drones, set_targets=set_targets)

    def set_tour(self) -> dict:
        return self.my_solution()  # {0: [0, 1, 2, 4, 2, 3, 0, 3], 1: [3]}

    def my_solution(self) -> dict:
        coords = np.array([np.array(t.coords) for t in self.set_targets])

        G = compute_from_coordinates(coords)
        plan = plan_builder(G, len(self.set_drones), len(self.set_targets))
        return plan


