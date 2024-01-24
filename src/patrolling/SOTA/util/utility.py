from matplotlib import colors
from matplotlib.legend_handler import *
from enum import Enum
# from rndgraph import *
# from deprecated import deprecated

from src.patrolling.SOTA.util.rndgraph import *
# from src.util import rndgraph
import networkx as nx
from src.patrolling.SOTA.util import myplot
import matplotlib.pyplot as plt
import random
import math
import numpy as np
import matplotlib.patches as mpatches
import os
import pickle


# import progressbar

# ------------------------------------------------------------------------------------
#
#										RANDOM GRAPH AND SQUAD FUNCTIONS
#
# ------------------------------------------------------------------------------------

def random_graph_builder(n, diameter=1410, node_weight=1,
                         graphShape=GraphShapes.random,
                         depotPosition=DepotPosition.random, no_edges=False, ndepots=1):
    """ takes in input all the parameters to build a graph and return
        the graph that is made by the input parameters
    """
    builder = RandomGraphBuilder(diameter, n, node_weight,
                                 depotPosition, graphShape, no_edges, ndepots)

    if graphShape in [GraphShapes.quad, GraphShapes.concentric_quad,
                      GraphShapes.double_concentric_quad, GraphShapes.multi_quad]:
        builder = QuadGraphBuilder(diameter, n, node_weight, depotPosition,
                                   graphShape, no_edges, ndepots)
    if graphShape in [GraphShapes.circle, GraphShapes.concentric_circle,
                      GraphShapes.double_concentric_circle, GraphShapes.multi_circle]:
        builder = CircleGraphBuilder(diameter, n, node_weight, depotPosition,
                                     graphShape, no_edges, ndepots)
    if graphShape in [GraphShapes.sphere]:
        builder = SphereGraphBuilder(diameter, n, node_weight, depotPosition,
                                     graphShape, no_edges, ndepots)

    return builder.build()


def advance_graph_builder(n, diameter=1410, node_weight=1,
                          graphShape=GraphShapes.random,
                          depotPosition=DepotPosition.random, no_edges=False, ndepots=1,
                          points=[], seed=None):
    """ takes in input all the parameters to build a graph and return
        a builder that can be used to retrieve the information about the graph
        and to build it
    """
    builder = RandomGraphBuilder(diameter, n, node_weight,
                                 depotPosition, graphShape, no_edges, ndepots, points=points, seed=seed)

    if graphShape in [GraphShapes.quad, GraphShapes.concentric_quad,
                      GraphShapes.double_concentric_quad, GraphShapes.multi_quad]:
        builder = QuadGraphBuilder(diameter, n, node_weight, depotPosition,
                                   graphShape, no_edges, ndepots)
    if graphShape in [GraphShapes.circle, GraphShapes.concentric_circle,
                      GraphShapes.double_concentric_circle, GraphShapes.multi_circle]:
        builder = CircleGraphBuilder(diameter, n, node_weight, depotPosition,
                                     graphShape, no_edges, ndepots)
    if graphShape in [GraphShapes.sphere]:
        builder = SphereGraphBuilder(diameter, n, node_weight, depotPosition,
                                     graphShape, no_edges, ndepots)

    return builder


def euclidean_distance(point1, point2):
    return math.sqrt(math.pow(point2[0] - point1[0], 2)
                     + math.pow(point2[1] - point1[1], 2))


def create_squad_drones(ndrones, speed, autonomies):
    '''
        create a list of object Drone that have the same and the autonomies given
    '''
    assert len(autonomies) == ndrones

    drones = []
    for i in range(0, ndrones):
        drones.append(Drone(autonomies[i], speed, i))

    return drones


# -----------------------------------------------------------------------------------
#
#									PRINT FUNCTIONS
#
# -----------------------------------------------------------------------------------

# @deprecated(version='1', reason="You should use myplot library")
def print_graph(graph, title="Graph", depot=None, edges_color='g',
                infotours=None, plot=True, save=True, save_dir=None):
    '''
        takes in input G : a graph of networkx class, title: a string that identify
        the graph plot, and use matplotlib to plot the graph in an external window
    '''
    G = graph.copy()
    pos = nx.get_node_attributes(G, 'pos')
    labels = dict(map(lambda x: ((x[0], x[1]), str(int(x[2]['weight']))),
                      G.edges(data=True)))

    if depot is None:
        nx.draw_networkx_nodes(G, pos=pos, node_size=35, label='nodes')
    else:
        nd_nodes = filter(lambda x: x is not depot, G.nodes())
        nx.draw_networkx_nodes(G, pos=pos, nodelist=nd_nodes, node_size=35,
                               node_color='r', label='nodes')
        nx.draw_networkx_nodes(G, pos=pos, nodelist=[depot], node_size=50,
                               node_color='b', label='depot')

    # walkaround to have same size window on each graph
    node_border = [10000, 10001, 10002, 10003]
    G.add_node(10000, pos=(-1000, -1000))
    G.add_node(10001, pos=(1000, -1000))
    G.add_node(10002, pos=(-1000, 1000))
    G.add_node(10003, pos=(1000, 1000))
    pos = nx.get_node_attributes(G, 'pos')
    nx.draw_networkx_nodes(G, pos=pos, nodelist=node_border,
                           node_size=0, node_color='b')

    # edge_labels.
    nx.draw_networkx_edge_labels(G, font_size=8, pos=pos, edge_labels=labels)

    if infotours is None:
        nx.draw_networkx_edges(G, pos, width=3, edge_color=edges_color,
                               arrows=False, label="edges")
    else:
        for infotour in infotours:
            paired = plt.get_cmap('Paired')
            c = paired(infotour[1] / float(len(infotours)))
            c = colors.to_hex(c)
            if infotour[0] == []:  # drones with no tours
                nx.draw_networkx_edges(G, pos, width=3,
                                       edgelist=[(depot, depot)], edge_color=c,
                                       arrows=False, label=infotour[2])
            nx.draw_networkx_edges(G, pos, width=3,
                                   edgelist=infotour[0], edge_color=c,
                                   arrows=False, label=infotour[2])

    plt.title(title)
    plt.legend()
    if plot:
        plt.plot()
        plt.show()
    if save:
        sdir = "" if save_dir is None else save_dir
        plt.savefig(sdir + "chart_" + title + ".svg")
    plt.close()


def make_proxy(clr, mappable, **kwargs):
    return Line2D([0, 1], [0, 1], color=clr, **kwargs)


# @deprecated(version='1', reason="You should use myplot library")
def print_tours(graph, depot, drones, tours, title="Trajectories", plot=True, save=True, save_dir=None):
    '''
        N.B. assuming the tours and drones are ordered s.t.
        tours[i] is associated to drones[i]
        takes in input a list of tour (list of list of edges)
        the graph that containt these tours and a title for the plot,
        then uses the print_graph(..) function to plot the circuit
    '''
    tours_graph = nx.Graph()
    infotours = []

    for t in range(0, len(tours)):
        tour = tours[t]
        edgeslist = []
        w_tour = 0
        for e in tour:
            tours_graph.add_node(e[0], pos=graph.nodes[e[0]]["pos"])
            tours_graph.add_node(e[1], pos=graph.nodes[e[1]]["pos"])
            tours_graph.add_edge(e[0], e[1], weight=graph.edges[e[0], e[1]]["weight"])
            edgeslist.append((e[0], e[1]))
            w_tour += graph.edges[e[0], e[1]]["weight"] + graph.nodes[e[1]]["weight"]
        infotours.append((edgeslist, t, "UAV: " + str(t)
                          + ", Cost: " + str(int(w_tour))
                          + ", Autonomy: " + str(drones[t].autonomy)))

    print_graph(tours_graph, title, depot, 'g', infotours, plot=plot, save=save, save_dir=save_dir)


# @deprecated(version='1', reason="You should use myplot library")
def print_circuit(edges, graph, title="Graph", depot=None, legend=None, plot=True, save=True, save_dir=None):
    '''
        takes in input a circuit (list of edges) the graph that containt
        this circuit and a title for the plot,
        then uses the print_graph(..) function to plot the circuit
    '''
    circuit_graph = nx.Graph()

    for e in edges:
        circuit_graph.add_node(e[0], pos=graph.nodes[e[0]]["pos"])
        circuit_graph.add_node(e[1], pos=graph.nodes[e[1]]["pos"])
        circuit_graph.add_edge(e[0], e[1], weight=graph.edges[e[0], e[1]]["weight"])

    print_graph(circuit_graph, title, depot)


# ----------------------------------------------------------------------------
#
#										CHRISTOFIDES FUNCTIONS
#
# -------------------------------------------------------------------------------

def christofides(G, depot=None, debug_print=False):
    '''
        Computes the TSP for the given graph using the well
        know 1.5appoximation of Christofides.
        return a graph with all the nodes in G
        but the only edges that compose the TSP tour.
        the optional depot allow to define the root of the TSP
        return the TSP, a list of edges
    '''
    if debug_print: print_graph(G, "Input Graph")  # print input graph

    ''' return a minimum spannign tree '''
    MST = nx.minimum_spanning_tree(G)

    if debug_print: print_graph(MST, "MST")  # print MST of Graph G

    odd_nodes = []
    for i in list(MST.nodes()):
        if MST.degree(i) % 2 != 0:
            odd_nodes += [i]

    H = G.subgraph(odd_nodes)

    # print subgraph induced on odd nodes
    if debug_print: print_graph(H, "SubGraph of odd nodes")

    # reverse weight edges to compute perfect minimum weight matching
    temp_H = H.copy()
    for edge in temp_H.edges():
        temp_H.edges[edge[0], edge[1]]["weight"] = \
            1.0 / temp_H.edges[edge[0], edge[1]]["weight"]

    # list of edges for a perfect matching graph
    perfect_match = nx.max_weight_matching(temp_H)
    if debug_print:  print_circuit(perfect_match, H, "Minimum Weight Perfect matching")

    # build Eulerian graph - MST + PERFECT MATCH
    eu_Graph = nx.MultiGraph()  #
    makeGraphFromTour(list(MST.edges()), MST, eu_Graph)

    for e in perfect_match:
        eu_Graph.add_edge(e[0], e[1], weight=H.edges[e[0], e[1]]["weight"])

    assert nx.is_eulerian(eu_Graph)

    if debug_print: print_graph(eu_Graph, "Eurelian Graph")

    # eurelian tour
    eu_tour = None
    if depot is None:
        eu_tour = list(nx.eulerian_circuit(eu_Graph, keys=False))
    else:
        eu_tour = list(nx.eulerian_circuit(eu_Graph, source=depot, keys=False))

    if debug_print: print_circuit(eu_tour, G, "Eurelian Tour")

    # pruning of eurelian tour to obtain 1.5TSP approximation
    TSP = shorcutVisit(eu_tour)
    if debug_print: print_circuit(TSP, G, "1.5TSP")

    return TSP


def makeGraphFromTour(tour, inGraph, outGraph):
    '''
        takes in input a tour (list of edges) an inGraph,
        the graph which from the circuit i builded, and
        an empty outGraph where build the representation graph for this tour.
    '''
    for e in tour:
        outGraph.add_node(e[0], pos=inGraph.nodes[e[0]]["pos"])
        outGraph.add_node(e[1], pos=inGraph.nodes[e[1]]["pos"])
        outGraph.add_edge(e[0], e[1], weight=inGraph.edges[e[0], e[1]]["weight"])


def shorcutVisit(eurelianTour):
    '''
        sub-routine for TSP-algorithm of Christofides. Takes input
        an eurelian tour and remove the nodes already visited in the tour..
    '''
    eu_nodes = [edge[0] for edge in eurelianTour]
    shorterTour_nodes = []
    for node in eu_nodes:
        if node not in shorterTour_nodes:
            shorterTour_nodes.append(node)

    shorterTour = []
    for i in range(0, len(shorterTour_nodes)):
        if i == len(shorterTour_nodes) - 1:  # ultimo nodo, va collegato al primo
            shorterTour.append((shorterTour_nodes[i], shorterTour_nodes[0]))
        else:
            shorterTour.append((shorterTour_nodes[i], shorterTour_nodes[i + 1]))

    return shorterTour


# ---------------------------------------------------------------------------------
#
#										UTILITIES
#
# ---------------------------------------------------------------------------------

def add_all_edges_from_node(graph, node):
    """ add all weighted edges between the node
        in input and all the others in the graph
    """
    assert node in graph.nodes
    # add weighted edges between node
    for i in graph.nodes:
        if i != node:
            w_ij = 0
            for coordinate in [0, 1]:
                w_ij += math.pow(graph.nodes[i]["pos"][coordinate]
                                 - graph.nodes[node]["pos"][coordinate], 2)
            graph.add_edge(i, node, weight=math.sqrt(w_ij))


def graph_weight(graph):
    """	weight of a graph
        including cost of edges and points
        each edge is assume undirected and added ones
        from (x,y) and (y,x) only one is added for both
    """
    weight = 0
    for e in graph.edges():
        weight += graph.edges()[e]["weight"]

    weight += cost_points(graph, graph.nodes())
    return weight


def max_distance(graph, nodes, depot):
    maxD = 0
    for n in nodes:
        maxD = max(maxD, euclidean_distance(
            graph.nodes[depot]["pos"],
            graph.nodes[n]["pos"]))
    return maxD


def cost_tour(graph, tour):
    cost = 0
    for edge in tour:
        cost += (graph.edges[edge[0], edge[1]]['weight']
                 + graph.nodes[edge[1]]["weight"])
    return cost


def len_tour(graph, tour):
    tlen = 0
    for edge in tour:
        tlen += graph.edges[edge[0], edge[1]]['weight']
    return tlen


def cost_points(graph, points):
    cost = 0
    for n in points:
        cost += graph.nodes[n]["weight"]
    return cost


def save_graph(graph, name="graph.gpickle", fdir=""):
    nx.write_gpickle(graph, fdir + name)


def load_graph(name="graph.gpickle", fdir=""):
    return nx.read_gpickle(fdir + name)


def createFolder(directory):
    try:
        if not os.path.exists(directory):
            os.makedirs(directory)
    except OSError:
        print('Error: Creating directory. ' + directory)


def save_obj(obj, fname="obj.pickle", fdir=""):
    with open(fdir + fname, 'wb') as output:
        pickle.dump(obj, output)


def load_obj(fname="obj.pickle", fdir=""):
    obj = None
    with open(fdir + fname, 'rb') as _input:
        obj = pickle.load(_input)
    return obj


# ------------------------------------------------------------
#
#										CLASSES
#
# -------------------------------------------------------------

class PathPlanningSolution:
    """ single round solution of path planning
        problem with multiple drones and single depot
    """

    def __init__(self, graph, depot, tours, drones, comment=None):
        self.graph = graph
        self.depot = depot
        self.tours = tours
        self.drones = drones
        self.comment = comment
        self.max_tour = max(t.cost for t in tours)
        self.average_tour = sum(t.cost for t in tours) / len(tours)
        self.nnetours = len([t.cost for t in tours if t.cost > 0])
        self.plot_manager = myplot.ToursPlotManager(self.graph, self.drones,
                                                    [t.edges for t in self.tours], self.depot)

    def plot(self, title="Drones Trajectories"):
        self.plot_manager.title = title
        self.plot_manager.make_plot()
        self.plot_manager.show()
        self.plot_manager.close()

    def save_plot(self, title="Drones Trajectories", fname="", fdir=""):
        self.plot_manager.title = title
        self.plot_manager.make_plot()
        self.plot_manager.save(fname=fname, fdir=fdir)
        self.plot_manager.close()

    def append_comment(self, comment):
        self.comment += comment

    def __str__(self):
        return ("Path Planning solution on: random graph of "
                + str(len(self.graph.nodes()))
                + " points, " + str(len(self.drones)) + " drones: \n"
                + "#Tours: " + str(len([t for t in self.tours if t.cost > 0])) + "\n"
                + "Max tour cost: " + str(self.max_tour) + "\n"
                + "Average tour cost: " + str(self.average_tour) + "\n"
                + "Comment: " + "\n"
                + self.comment)


class Tour:
    def __init__(self, graph, edges, depot=None):
        self.graph = graph
        self.edges = edges
        self.depot = depot
        if depot is not None and edges != []:
            assert self.check_depot()
        self.cost = self.__compute_cost()
        self.nodes = set(x[0] for x in edges)
        self.nnodes = len(self.nodes)

    def point_times(self, speed):
        """ return a list of ordered (time : point) """
        times = []
        cost = 0
        for i in range(len(self.edges)):  # last edge don't partecipates
            edge = self.edges[i]
            cost += (self.graph.edges[edge[0], edge[1]]['weight']
                     + self.graph.nodes[edge[1]]["weight"])
            times.append((int(cost / speed), self.graph.nodes[edge[1]]["pos"]))
        return times

    def inspection_times(self):
        times = []
        cost = 0
        for i in range(len(self.edges) - 1):  # last edge don't partecipates
            edge = self.edges[i]
            cost += (self.graph.edges[edge[0], edge[1]]['weight']
                     + self.graph.nodes[edge[1]]["weight"])
            times.append(cost)
        return times

    def inspection_times_dict(self, base_cost=0):
        times = {}
        cost = 0
        for i in range(len(self.edges) - 1):  # last edge don't partecipates
            edge = self.edges[i]
            cost += (self.graph.edges[edge[0], edge[1]]['weight']
                     + self.graph.nodes[edge[1]]["weight"])
            times[edge[1]] = cost + base_cost
        return times

    def update(self, edges, depot=None):
        self.edges = edges
        self.depot = depot
        if depot is not None and edges != []:
            assert self.check_depot()
        self.cost = self.__compute_cost()
        self.nodes = set(x[0] for x in edges)
        self.nnodes = len(self.nodes)

    def check_depot(self):
        return (self.edges[0][0] == self.depot
                and
                self.edges[-1][1] == self.depot)

    def __compute_cost(self):
        return cost_tour(self.graph, self.edges)

    def __str__(self):
        out = "Tour istance - "
        for e in self.edges:
            out += str(e)
        return out


class Drone:
    def __init__(self, autonomy, speed, key=None):
        self.autonomy = autonomy
        self.speed = speed
        self.key = key

    def __str__(self):
        out = ""
        if self.key is None:
            out += "Drone_with"
        else:
            out += "Drone" + str(self.key) + ": "

        out += str(self.autonomy) + " autonomy, " + str(self.speed) + " speed."
        return out


class Point2D:
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def __str__(self):
        return "Point: ", self.x, self.y


class ProgressBar:
    def __init__(self, max_value):
        self.bar = progressbar.ProgressBar(max_value=max_value)

    def update(self, s):
        self.bar.update(s)


class MultiRoundSolution:
    """ Multi round solution of path planning
        problem with multiple drones and multiple depots
    """

    def __init__(self, graph, roots, drones, comment=None):
        self.graph = graph
        self.roots = roots
        self.roots_set = set(roots)
        self.nnodes = (graph.number_of_nodes()
                       - len(self.roots_set))
        self.tours = {i: [] for i in range(len(drones))}
        self.drones = drones
        self.max_round = 0
        self.visited_nodes = set()
        self.comment = comment
        self.running_time = 0

    def add_tours(self, i_drone, tours):
        self.max_round = max(self.max_round, len(tours))
        for t in tours: self.visited_nodes |= t.nodes
        self.tours[i_drone] = [Tour(t.graph,
                                    [e for e in t.edges],
                                    t.depot)
                               for t in tours]

    def append_tour(self, i_drone, tour):
        self.visited_nodes |= tour.nodes
        self.tours[i_drone].append(Tour(tour.graph,
                                        [e for e in tour.edges], tour.depot))
        self.max_round = max(self.max_round,
                             len(self.tours[i_drone]))

    def tours_at_round(self, n):
        """ return the tours at round n
            the list have len = #drones
            if a drone doesn't partecipate in the round
            if will have empty tour
        """
        round_tours = []
        for i in range(len(self.drones)):
            try:
                round_tours.append(self.tours[i][n])
            except:
                round_tours.append(Tour(self.graph,
                                        [], self.roots[i]))
        return round_tours

    def tours_of_drone(self, i_drone):
        """ return the tours in the rounds for the drone i_drone
            if the drone doesn't partecipate in a round
            if will have empty tour
        """
        return self.tours[i_drone]

    def all_tours_obj(self):
        all_tours = []
        for i in range(len(self.drones)):
            all_tours.extend([t for t in self.tours[i]])
        return all_tours

    def all_tours(self):
        all_tours = []
        for i in range(len(self.drones)):
            all_tours.extend([t.edges for t in self.tours[i]])
        return all_tours

    def plot(self, dim):
        all_tours = self.all_tours()
        chart = myplot.ToursPlotManager(self.graph,
                                        [Drone(0, 0) for t in all_tours],
                                        all_tours, self.roots, dim_plot=dim)
        chart.make_plot()
        chart.show()

    def N_rate(self, rate, N):
        """ return the first round
            for which we have the input rate
            of cov rate
        """
        vis_points = set()
        for i in range(N):
            for t in self.tours_at_round(i):
                vis_points |= t.nodes
            if self.cov_rate(vis_points) >= rate:
                return i
        return N

    def N25(self, N):
        """ return the first round
            for which we have 25% of cov rate
        """
        return self.N_rate(0.25, N)

    def N50(self, N):
        """ return the first round
            for which we have 50% of cov rate
        """
        return self.N_rate(0.50, N)

    def N75(self, N):
        """ return the first round
            for which we have 75% of cov rate
        """
        return self.N_rate(0.75, N)

    def N100(self, N):
        """ return the first round
            for which we have 100% of cov rate
        """
        return self.N_rate(1, N)

    def avg_inspection_time(self, N):
        """ assuming no cost inter-round
            the next round starts at the end
            of the last tour of the previous one
        """
        times = []
        cost = 0
        for r in range(N):
            max_completion_time = 0
            tmp_times = []
            for t in self.tours_at_round(r):
                if t.edges != []:
                    max_completion_time = max(t.cost,
                                              max_completion_time)
                    tmp_times.extend([cost + time
                                      for time in t.inspection_times()])

            times.extend(tmp_times)
            cost += max_completion_time
            if tmp_times == []:
                break

        return sum(times) / float(self.nnodes)

    def cov_rate(self, vis_point):
        """ cov rate of the whole mission """
        return (len(vis_point - self.roots_set)
                / float(self.nnodes))

    def average_tour_cost(self, N):
        n_tours = 0.0
        sum_tours_cost = 0.0
        for i in range(N):
            for t in self.tours_at_round(i):
                n_tours += 1
                sum_tours_cost += t.cost
        return sum_tours_cost / n_tours

    def coverage_rate(self):
        """ return the rate of covered points
            respect all the points in the graph
        """
        return self.cov_rate(self.visited_nodes)

    def last_used_round(self):
        """ return the last round that
            involve non empty tours
        """
        last_used_round = 0
        for i in range(self.max_round):
            for t in self.tours_at_round(i):
                if t.edges != []:
                    last_used_round = i + 1
        return last_used_round

    def number_used_rounds(self):
        """ return the number of used rounds
            that involve non empty tours
        """
        n_used_rounds = 0
        for i in range(self.max_round):
            no_empty_round = False
            for t in self.tours_at_round(i):
                if t.edges != []:
                    no_empty_round = True
            if no_empty_round:
                n_used_rounds += 1
        return n_used_rounds

    def completion_time(self, round_cost=0):
        """ return the cost of last round that
            involve non empty tours
        """
        last_used_round = self.last_used_round()
        completion_time = 0
        for t in self.tours_at_round(last_used_round - 1):
            completion_time = max(t.cost, completion_time)

        return last_used_round * round_cost + completion_time

    def cumulative_coverage(self, N):
        """ return a list of len(list) == N
            s.t. each element of the list
            indicates how many new points are
            visited in that round (expect the depots
            which are assumed already visited at the beginning)
        """
        cumulative_coverage = []
        visited_nodes = set(self.roots)
        for i in range(N):
            round_nodes = set()
            for t in self.tours_at_round(i):
                round_nodes |= t.nodes

            cumulative_coverage.append(
                len(round_nodes - visited_nodes))
            visited_nodes |= round_nodes

        return cumulative_coverage

    def cumulative_coverage_score(self, N):
        """ return the cumulative coverage score
            of the solution
        """
        cumulative_coverage = self.cumulative_coverage(N)
        cumulative_coverage_score = 0
        for i in range(N):
            cumulative_coverage_score += \
                cumulative_coverage[i] * (N - i)
        return cumulative_coverage_score

    def mean_waiting_round(self, N):
        """ return the mean waiting round
            for the points before they get visited
        """
        waited_rounds_sum = 0
        cum_cov = self.cumulative_coverage(N)
        for r in range(len(cum_cov)):
            waited_rounds_sum += cum_cov[r] * (r + 1)
        return waited_rounds_sum / float(self.nnodes)

    def cov_rate_at_round(self, N):
        """ return for each round
            the coverage rate reached"""
        cov_points_sum = [0] * N
        cum_cov = self.cumulative_coverage(N)
        cov_points_sum[0] = cum_cov[0]
        for r in range(1, len(cum_cov)):
            cov_points_sum[r] = cov_points_sum[r - 1] + cum_cov[r]
        for i in range(len(cum_cov)):
            cov_points_sum[r] = cov_points_sum[r] / float(self.nnodes)

        return cov_points_sum

