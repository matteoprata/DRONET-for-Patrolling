from matplotlib import colors
from matplotlib.legend_handler import *
from networkx.drawing.nx_agraph import to_agraph

import csv
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np


def gradient_color(lenght):
    t_colors = []
    paired = plt.get_cmap('Paired')
    for i in range(lenght):
        c = paired(i / float(lenght))
        t_colors += [colors.to_hex(c)]
    return t_colors


# -----------------------------------------------------------------------------------
#
#									PRINT FUNCTIONS
#
# -----------------------------------------------------------------------------------
class GraphPlotManager:
    I_DEPOT = {"size": 50, "color": 'b', "label": "depot"}
    I_NODE = {"size": 35, "color": 'r', "label": "nodes"}
    I_EDGE = {"size": 3, "color": 'g', "label": "edges"}
    ELABELS_SIZE = 8

    def __init__(self, graph, depot=[], title="Graph", personalized_elabels=None, dim_plot=1000):
        self.graph = graph
        self.nnodes = len(graph.nodes())
        self.depot = depot if isinstance(depot, list) else [depot]
        self.title = title
        self.personalized_elabels = personalized_elabels
        self.dim = dim_plot

    def make_plot(self):
        pos = nx.get_node_attributes(self.graph, 'pos')
        self.plot_nodes(pos)
        self.plot_edgelabels(pos)
        self.plot_edges(pos)
        self.fix_plot_dim(pos)
        plt.title(self.title)

    # plt.legend()

    def show(self):
        plt.plot()
        plt.show()

    def save(self, fname="", fdir=""):
        if fname == "": fname = self.title
        plt.savefig(fdir + fname + ".svg")

    def close(self):
        plt.close()

    def plot_edges(self, pos):
        nx.draw_networkx_edges(self.graph, pos,
                               width=GraphPlotManager.I_EDGE["size"],
                               edge_color=GraphPlotManager.I_EDGE["color"],
                               arrows=False,
                               label=GraphPlotManager.I_EDGE["label"])

    def plot_edgelabels(self, pos):
        if self.personalized_elabels is None:
            labels = dict(map(lambda x: ((x[0], x[1]),
                                         str(int(x[2]["weight"]))),
                              self.graph.edges(data=True)))
        else:
            labels = self.personalized_elabels

        nx.draw_networkx_edge_labels(self.graph,
                                     font_size=GraphPlotManager.ELABELS_SIZE,
                                     pos=pos,
                                     edge_labels=labels)

    def plot_nodes(self, pos):
        if self.depot == []:
            nx.draw_networkx_nodes(self.graph, pos=pos,
                                   node_size=GraphPlotManager.I_NODE["size"],
                                   label=GraphPlotManager.I_NODE["label"])
        else:
            nd_nodes = filter(lambda x: x not in self.depot, self.graph.nodes())
            nx.draw_networkx_nodes(self.graph, pos=pos, nodelist=nd_nodes,
                                   node_size=GraphPlotManager.I_NODE["size"],
                                   node_color=GraphPlotManager.I_NODE["color"],
                                   label=GraphPlotManager.I_NODE['label'])
            nx.draw_networkx_nodes(self.graph, pos=pos, nodelist=self.depot,
                                   node_size=GraphPlotManager.I_DEPOT["size"],
                                   node_color=GraphPlotManager.I_DEPOT["color"],
                                   label=GraphPlotManager.I_DEPOT["label"])

    def fix_plot_dim(self, pos):
        plt.axis([-self.dim, self.dim, -self.dim, self.dim])


class ToursPlotManager(GraphPlotManager):
    ''' The class is delegated to plot a set of tours and
        with the associated drones. It is assumed that
        drones are matched against the tours (i.e
        dronei[i] use tours[i])
    '''

    def fix_plot_dim(self, pos):
        plt.axis([0, self.dim, 0, self.dim])

    def __init__(self, graph, drones, tours, depot=[],
                 title="Trajectories", personalized_elabels=None, t_colors=[], dim_plot=None):
        self.drones = drones
        self.tours = tours
        self.info_tours = []
        self.depot = depot
        self.base_graph = graph
        if t_colors == []:
            self.t_colors = gradient_color(len(self.tours))
        else:
            self.t_colors = t_colors
        GraphPlotManager.__init__(self, self.tours_graph(),
                                  depot, title, personalized_elabels, dim_plot=dim_plot)

    def tours_graph(self):
        tours_graph = nx.Graph()
        for d in self.depot:
            tours_graph.add_node(d, pos=self.base_graph.nodes[d]["pos"])
        for t in range(0, len(self.tours)):
            edgeslist = []
            w_tour = 0
            for e in self.tours[t]:
                tours_graph.add_node(e[0], pos=self.base_graph.nodes[e[0]]["pos"])
                tours_graph.add_node(e[1], pos=self.base_graph.nodes[e[1]]["pos"])
                tours_graph.add_edge(e[0], e[1],
                                     weight=self.base_graph.edges[e[0], e[1]]["weight"])
                edgeslist.append((e[0], e[1]))
                w_tour += (self.base_graph.edges[e[0], e[1]]["weight"]
                           + self.base_graph.nodes[e[1]]["weight"])
            self.info_tours.append("UAV: " + str(t)
                                   + ", Cost: " + str(int(w_tour))
                                   + ", Autonomy: " + str(self.drones[t].autonomy))
        return tours_graph

    def plot_edges(self, pos):
        for i in range(0, len(self.tours)):
            c = self.t_colors[i]
            nx.draw_networkx_edges(self.graph, pos, width=GraphPlotManager.I_EDGE["size"],
                                   edgelist=self.tours[i], edge_color=c,
                                   arrows=False, label=self.info_tours[i])
            if self.tours[i] == []:
                edgelist = [(self.depot[0], self.depot[0])]
                if i < len(self.depot):  # different depots for tours tour[i] has depot[i]
                    edgelist = [(self.depot[i], self.depot[i])]
                nx.draw_networkx_edges(self.graph, pos,
                                       width=GraphPlotManager.I_EDGE["size"],
                                       edgelist=edgelist,
                                       edge_color=c,
                                       arrows=False, label=self.info_tours[i])


class RoundsPlotManager(GraphPlotManager):
    ''' The class is delegated to plot a set of tours and
        with the associated drones. It is assumed that
        drones are matched against the tours (i.e
        dronei[i] use tours[i])
    '''

    def __init__(self, graph, drones, tours, depot=[],
                 title="Rounds Trajectories", personalized_elabels=None,
                 t_colors=["r", "black", "g"],
                 styles=['solid', 'dashed', 'dotted'],
                 dim_plot=1000):
        """ made for comparison on tre rounds!! """
        GraphPlotManager.I_NODE["label"] = "points"
        self.drones = drones
        self.styles = styles
        self.tours = tours
        self.info_tours = []
        self.depot = depot
        self.base_graph = graph
        if t_colors == []:
            self.t_colors = gradient_color(len(self.tours))
        else:
            self.t_colors = t_colors
        GraphPlotManager.__init__(self, self.tours_graph(),
                                  depot, title, personalized_elabels, dim_plot=dim_plot)

    def make_plot(self):
        pos = nx.get_node_attributes(self.graph, 'pos')
        self.plot_nodes(pos)
        # self.plot_edgelabels(pos)
        self.plot_edges(pos)
        self.fix_plot_dim(pos)
        plt.title(self.title)
        plt.xticks([])
        plt.yticks([])
        plt.legend()

    def tours_graph(self):
        tours_graph = nx.Graph()
        for d in self.depot:
            tours_graph.add_node(d, pos=self.base_graph.nodes[d]["pos"])
        for t in range(0, len(self.tours)):
            edgeslist = []
            w_tour = 0
            for e in self.tours[t]:
                tours_graph.add_node(e[0], pos=self.base_graph.nodes[e[0]]["pos"])
                tours_graph.add_node(e[1], pos=self.base_graph.nodes[e[1]]["pos"])
                tours_graph.add_edge(e[0], e[1],
                                     weight=self.base_graph.edges[e[0], e[1]]["weight"])
                edgeslist.append((e[0], e[1]))
                w_tour += (self.base_graph.edges[e[0], e[1]]["weight"]
                           + self.base_graph.nodes[e[1]]["weight"])
            self.info_tours.append("Round: " + str(t + 1))
        return tours_graph

    def plot_edges(self, pos):
        for i in range(0, len(self.tours)):
            c = self.t_colors[i]
            nx.draw_networkx_edges(self.graph, pos, width=GraphPlotManager.I_EDGE["size"],
                                   edgelist=self.tours[i], edge_color=c,
                                   style=self.styles[i],
                                   arrows=False, label=self.info_tours[i])
            if self.tours[i] == []:
                edgelist = [(self.depot[0], self.depot[0])]
                if i < len(self.depot):  # different depots for tours tour[i] has depot[i]
                    edgelist = [(self.depot[i], self.depot[i])]
                nx.draw_networkx_edges(self.graph, pos,
                                       width=GraphPlotManager.I_EDGE["size"],
                                       edgelist=edgelist,
                                       edge_color=c,
                                       style=self.styles[i],
                                       arrows=False, label=self.info_tours[i])


class RelaxedToursPlotManager:

    def __init__(self, graph, edgesvar_weight, drones,
                 tours, depot=None, title="Trajectories"):
        self.edgesvar_weight = edgesvar_weight
        self.plots = []
        self.figures = []
        self.tours = tours
        self.t_colors = gradient_color(len(self.tours))
        for i in range(len(tours)):
            self.plots.append(ToursPlotManager(graph, [drones[i]], [tours[i]],
                                               depot, title="Drone: " + str(i),
                                               personalized_elabels=self.edgeslabels(i),
                                               t_colors=[self.t_colors[i]]))

    def edgeslabels(self, index):
        labels = dict()
        for e in self.tours[index]:
            labels[e[0], e[1]] = round(self.edgesvar_weight[index, e[0], e[1]], 2)
        return labels

    def make_plot(self):
        for i in range(len(self.plots)):
            self.figures += [plt.figure(i)]
            self.plots[i].make_plot()

    def show(self):
        plt.plot()
        plt.show()


class RelaxedToursGraphziv:
    def __init__(self, graph, edgesvar_weight, drones,
                 tours, depot=None, title="Trajectories"):
        self.graph = graph
        self.edgesvar_weight = edgesvar_weight
        self.tours = tours
        self.var_multigraph = nx.MultiDiGraph()
        self.var_multigraph.add_node(depot, label="DEPOT", pos=graph.nodes[depot]['pos'])
        self.add_nodes()
        self.t_colors = gradient_color(len(self.tours))
        self.multigraph()

    def add_nodes(self):
        for i in range(len(self.graph.nodes()) - 1):
            self.var_multigraph.add_node(i, label=str(i),
                                         pos=str(self.graph.nodes[i]['pos']) + "!", pin="true")

    def multigraph(self, decimal=2):
        for i in range(len(self.tours)):
            tour = self.tours[i]
            for edge in tour:
                self.var_multigraph.add_edge(edge[0], edge[1],
                                             color=self.t_colors[i],
                                             label=round(self.edgesvar_weight[i, edge[0], edge[1]], decimal))
        self.var_multigraph.graph['edge'] = {'arrowsize': '0.6', 'splines': 'curved'}
        self.var_multigraph.graph['graph'] = {'scale': '3'}

    def save(self, fname="relaxedTours.png", fdir=""):
        A = to_agraph(self.var_multigraph)
        A.layout('dot')
        A.draw(fdir + fname)

