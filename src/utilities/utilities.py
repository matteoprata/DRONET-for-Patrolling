
""" To clean. """

import pathlib
import time
import json
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import pickle
from ast import literal_eval as make_tuple
import os
from shapely.geometry import LineString
import signal
from multiprocessing import Pool
import networkx as nx
from sklearn.cluster import KMeans
from collections import defaultdict

def flip_biased_coin(p, random_gen):
    """ Return true with probability p, false with probability 1-p. """
    return random_gen.random() < p


def log(message_to_log, is_to_log=True, current_ts=1, log_every=1):
    """ Logs message_to_log, if is_to_log or every log_every steps (given current_ts). """
    if not is_to_log or not (current_ts % log_every == 0):
        return
    print(message_to_log)


def current_date():
    return str(time.strftime("%d%m%Y-%H%M%S"))


def euclidean_distance(p1, p2):
    """ Given points p1, p2 in R^2 it returns the norm of the vector connecting them.  """
    return np.linalg.norm(np.array(p1)-np.array(p2))


def xor(a, b):
    return (a and not b) or (not a and b)


def clustering_kmeans(set_coords, n_clusters):
    kmeans_vals = KMeans(n_clusters=n_clusters, random_state=0, n_init="auto").fit(set_coords)
    clusters = np.array(kmeans_vals.labels_)
    return clusters


def generate_random_coordinates_in_circle(x, y, r, num_points):
    points = []
    for _ in range(num_points):
        # Generate a random angle between 0 and 2pi
        angle = np.random.uniform(0, 2 * np.pi)

        # Generate a random radius between 0 and r
        radius = np.random.uniform(0, r)

        # Convert polar coordinates to Cartesian coordinates
        new_x = x + radius * np.cos(angle)
        new_y = y + radius * np.sin(angle)

        # Add the coordinates to the list of points
        points.append([new_x, new_y])
    return points


def min_max_normalizer(value, startLB, startUB, endLB=0, endUB=1, soft=False):
    # Figure out how 'wide' each range is
    value = np.asarray(value)
    if not soft:
        assert((value <= startUB).all() and (value >= startLB).all())

    leftSpan = startUB - startLB
    rightSpan = endUB - endLB

    # Convert the left range into a 0-1 range (float)
    valueScaled = (value - startLB) / leftSpan

    # Convert the 0-1 range into a value in the right range.
    return ((valueScaled * rightSpan) + endLB)


def angle_between_three_points(p1, p2, p3):
    """ Given points p1, p2, p3 returns the angle between them.  """
    point_a = p1 - p2
    point_b = p3 - p2

    ang_a = np.arctan2(*point_a[::-1])
    ang_b = np.arctan2(*point_b[::-1])

    return np.rad2deg((ang_a - ang_b) % (2 * np.pi))


def pickle_data(data, filename):
    """ save the metrics on file """
    with open(filename, 'wb') as out:
        pickle.dump(data, out)


def unpickle_data(filename):
    """ load the metrics from a file """
    with open(filename, 'rb') as handle:
        obj = pickle.load(handle)
    return obj


def save_txt(text, file):
    with open(file, "w") as f:
        f.write(text)


def distance_point_segment(ps1, ps2, external_p):
    """Returns the distance between segment and point. Don't ask why."""
    x3, y3 = external_p
    x1, y1 = ps1
    x2, y2 = ps2

    px = x2-x1
    py = y2-y1

    norm = px*px + py*py

    u =  ((x3 - x1) * px + (y3 - y1) * py) / float(norm)

    if u > 1:
        u = 1
    elif u < 0:
        u = 0

    x = x1 + u * px
    y = y1 + u * py

    dx = x - x3
    dy = y - y3

    dist = (dx*dx + dy*dy)**.5

    return dist


def is_segments_intersect(A, B, C, D):
    """ Return true if line segments AB and CD intersect """
    segment1 = LineString([A, B])
    segment2 = LineString([C, D])
    point = segment1.intersection(segment2)

    return not point.is_empty

# from scipy.stats import norm
# def skew_norm_pdf(x, m=0, s=1, a=0):
#     # adapated from:
#     # http://stackoverflow.com/questions/5884768/skew-normal-distribution-in-scipy
#     t = (x-m) / s
#     return 2.0 * s * norm.pdf(t) * norm.cdf(a*t)


# def rand_skew_norm(alpha, mean, std, rand_generator=None):
#     sigma = alpha / np.sqrt(1.0 + alpha ** 2)
#
#     afRN = np.random.randn(2)
#     u0 = afRN[0]
#     v = afRN[1]
#     u1 = sigma*u0 + np.sqrt(1.0 -sigma**2) * v
#
#     if u0 >= 0:
#         return u1 * std + mean
#     return (-u1) * std + mean

# ------------------ Event (Traffic) Generator ----------------------
class EventGenerator:

    def __init__(self, simulator):
        """
        :param simulator: the main sim object
        """
        self.simulator = simulator
        self.rnd_drones = np.random.RandomState(self.simulator.seed)
        # for now no random on number of event generated
        # self.rnd_event = np.random.RandomState(self.sim.seed)

    def handle_events_generation(self, cur_step : int, drones : list):
        """
        at fixed time randomly select a drone from the list and sample on it a packet/event.

        :param cur_step: the current step of the simulation to decide whenever sample an event or not
        :param drones: the drones where to sample the event
        :return: nothing
        """
        if cur_step > 0 and cur_step % self.simulator.event_generation_delay == 0:  # if it's time to generate a new packet
            print(self.simulator.event_generation_delay, cur_step, cur_step % self.simulator.event_generation_delay)
            # drone that will receive the packet:
            drone_index = self.rnd_drones.randint(0, len(drones))
            drone = drones[drone_index]
            drone.feel_event(cur_step)


def json_to_paths(json_file_path):
    """ load the tour for drones
        and return a dictionary {drone_id : list of waypoint}

        e.g.,
        accept json that contains:
        {"drones": [{"index": "0", "tour": ["(1500, 0)", "(1637, 172)", ...
                    (1500, 0)"]}, {"index": "1", "tour": ["(1500, 0)",

        TOURS = {
            0 : [(0,0), (2000,2000), (1500, 1500), (200, 2000)],
            1 : [(0,0), (2000, 200), (200, 2000), (1500, 1500)]
        }
    """
    out_data = {}
    with open(json_file_path, 'r') as in_file:
        data = json.load(in_file)
        for drone_data in data["drones"]:
            drone_index = int(drone_data["index"])
            drone_path = []
            for waypoint in drone_data["tour"]:
                drone_path.append(make_tuple(waypoint))
            out_data[drone_index] = drone_path
    return out_data


def box_plot(data, pos, edge_color="red", fill_color="white"):
    bp = plt.boxplot(data, positions=pos, patch_artist=True, widths=0.5, showmeans=True)

    for element in ['boxes', 'whiskers', 'fliers', 'means', 'medians', 'caps']:
        plt.setp(bp[element], color=edge_color)

    for patch in bp['boxes']:
        patch.set(facecolor=fill_color, alpha=.2)
    return bp


def sample_color(index, cmap='tab10'):
    # 1. Choose your desired colormap
    cmap = plt.get_cmap(cmap)

    # 2. Segmenting the whole range (from 0 to 1) of the color map into multiple segments
    colors = [cmap(x) for x in range(cmap.N)]
    assert index < cmap.N

    # 3. Color the i-th line with the i-th color, i.e. slicedCM[i]
    color = colors[index]
    return color


def initializer():
    """Ignore CTRL+C in the worker process."""
    signal.signal(signal.SIGINT, signal.SIG_IGN)


def execute_parallel(process, arguments, n_cores=1):
    with Pool(initializer=initializer, processes=n_cores) as pool:
        try:
            pool.starmap(process, arguments)

        except KeyboardInterrupt:
            pool.terminate()
            pool.join()

    print("COMPLETED SUCCESSFULLY")


def write_json(msg, fname):
    with open(fname, 'w') as fp:
        json.dump(msg, fp)


def read_json(fname):
    data = None
    if os.path.exists(fname):
        with open(fname, 'r') as fp:
            data = json.load(fp)
    return data


class LimitedList:
    """ Time window """
    def __init__(self, threshold=None):
        self.llist = []
        self.threshold = threshold

    def append(self, el):
        if self.threshold and self.threshold < len(self.llist) + 1:
            self.llist = self.llist[1:]
        self.llist.append(el)

    def __len__(self):
        return len(self.llist)

    def __getitem__(self, index):
        return self.llist[index]


def make_path(fname):
    path = pathlib.Path(fname)
    path.parent.mkdir(parents=True, exist_ok=True)


def plot_X(X, plt_title, plt_path, window_size=30, is_avg=True):
    if len(X) >= window_size:
        df = pd.Series(X)
        scatter_print = X[window_size:]
        to_plot_data = df.rolling(window_size).mean()[window_size:]

        plt.clf()
        plt.plot(range(len(scatter_print)), to_plot_data, label="Moving Average-" + str(window_size))
        if is_avg:
            plt.plot(range(len(scatter_print)), [np.average(scatter_print)] * len(scatter_print), label="avg")

        plt.legend()
        plt.title(plt_title)
        plt.savefig(plt_path)
        plt.clf()


""" This class handle the return to depot for
    the drones, such that they return to the depot in a coordinated fashion
    currently is based on channel -> in future can also handle cluster head/waypoints
"""


class PathToDepot:

    def __init__(self, x_position, simulator):
        """ for now just a middle channel in the area used by all the drones """
        self.x_position = x_position
        self.simulator = simulator

    def next_target(self, drone_pos):
        """ based on the drone position return the next target:
            |-> channel position or cluster head position
            |-> the depot if the drones are already in the channel or have overpass the cluster head
        """
        # only channel mode
        if abs(drone_pos[0] - self.x_position) < 1:  # the drone is already on the channel with an error of 1 meter
            return self.simulator.depot_coordinates
        else:
            return self.x_position, drone_pos[1]  # the closest point to the channel


def measure_scaler(measure, dom_start, dom_target):
    """ Scales the measure value in the start domain [Type, min, max], in the target domain. """
    return (measure - dom_start[1]) / (dom_start[2] - dom_start[1]) * (dom_target[2] - dom_target[1]) + dom_target[1]

# -------------------- all cells computation ---------------------#


class TraversedCells:

    @staticmethod
    def cells_in_travel(size_cell, width_area, start, end):
        """ return the cell number in which the pos (x, y) lay """

        start_cell, coords_cell_start = TraversedCells.coord_to_cell(size_cell, width_area, start[0], start[1])  # int, lower left coordinates
        end_cell, coords_cell_end = TraversedCells.coord_to_cell(size_cell, width_area, end[0], end[1])

        out_cells = []

        if coords_cell_end[1] == coords_cell_start[1]:  # vertical alignment
            min_x = min(coords_cell_start[0], coords_cell_end[0])
            max_x = max(coords_cell_start[0], coords_cell_end[0])
            for x_ in range(min_x, max_x + 1):
                out_cells.append((x_, coords_cell_end[1]))
            return out_cells

        if coords_cell_end[0] == coords_cell_start[0]:  # horizontal alignment
            min_y = min(coords_cell_start[1], coords_cell_end[1])
            max_y = max(coords_cell_start[1], coords_cell_end[1])
            for y_ in range(min_y, max_y + 1):
                out_cells.append((coords_cell_end[0], y_))
            return out_cells

        # Diagonal line
        # Boundaries of the rectangle
        min_x, max_x = min(coords_cell_start[0], coords_cell_end[0]), max(coords_cell_start[0], coords_cell_end[0])
        min_y, max_y = min(coords_cell_start[1], coords_cell_end[1]), max(coords_cell_start[1], coords_cell_end[1])

        # All the cells of the rectangle, indices
        coords_index = [(i, j) for i in range(min_x, max_x+1) for j in range(min_y, max_y+1)]
        for cell in coords_index:

            ll = cell[0]*size_cell, cell[1]*size_cell
            lr = cell[0]*size_cell + size_cell, cell[1]*size_cell
            ul = cell[0]*size_cell, cell[1]*size_cell + size_cell
            ur = cell[0]*size_cell + size_cell, cell[1]*size_cell + size_cell

            if TraversedCells.intersect_quad(start, end, ll, lr, ul, ur):
                out_cells.append(cell)

        return out_cells  # list of lower-lefts

    @staticmethod
    def intersect_quad(start, end, ll, lr, ul, ur):

        return (TraversedCells.intersect_segments(start, end, ll, lr)
                or TraversedCells.intersect_segments(start, end, ul, ur)
                or TraversedCells.intersect_segments(start, end, ul, ll)
                or TraversedCells.intersect_segments(start, end, lr, ur))

    @staticmethod
    def intersect_segments(start1:tuple, end1:tuple, start2:tuple, end2:tuple):
        if end1 == start2:
            return True
        if end2 == start1:
            return True
        if start2 == start1:
            return True
        if end2 == end1:
            return True

        a = np.asarray(end1) - np.asarray(start1)  # direction of line a
        b = np.asarray(start2) - np.asarray(end2)  # direction of line b, reversed
        d = np.asarray(start2) - np.asarray(start1)  # right-hand side
        det = a[0] * b[1] - a[1] * b[0]

        if det == 0:
            return False

        t = (a[0] * d[1] - a[1] * d[0]) / det
        return 0 <= t <= 1

    @staticmethod
    def all_centers(widht_area, height_area, size_cell):
        """ return all cell along their centers """
        all_cells_and_centers = []
        for x in range(0, widht_area, size_cell):
            for y in range(0, height_area, size_cell):
                all_cells_and_centers.append(
                    (TraversedCells.coord_to_cell(size_cell, widht_area, x, y),
                        (x + (size_cell/2.0), y + (size_cell/2.0)))
                )
        return all_cells_and_centers

    @staticmethod
    def coord_to_cell(size_cell, width_area, x_pos, y_pos):
        """ return the cell number in which the pos (x"abs", y"abs") lay """
        x_cell_coords = int(x_pos / size_cell)
        y_cell_coords = int(y_pos / size_cell)
        return TraversedCells.cell_coord_to_cell_number(size_cell, width_area, x_cell_coords,
                                                        y_cell_coords), (x_cell_coords, y_cell_coords)

    @staticmethod
    def cell_coord_to_cell_number(size_cell, width_area, x_cell_coords, y_cell_coords):
        """ return the number o the cells given the indexes """

        x_cells = np.ceil(width_area / size_cell)  # numero di celle su X
        return x_cell_coords + (x_cells * y_cell_coords)


class Christofides:
    """
    A class for compute the approximated solution of TSP by Christofides
    """

    def compute_from_coordinates(cls, coordinates, depot_index):
        graph = nx.Graph()
        for i, coo in enumerate(coordinates):
            graph.add_nodes_from([(i, {"x": coo[0], "y": coo[1]})])

        for i, coo1 in enumerate(coordinates):
            for j, coo2 in enumerate(coordinates):
                if i > j:
                    graph.add_edges_from([(i, j, {"weight": euclidean_distance(coo1, coo2)})])

        return cls.compute_from_graph(graph, depot_index)

    @classmethod
    def compute_from_graph(cls, graph : nx.Graph, depot_index: int):
        """
        :param graph: the graph where compute the TSP tour
        :param: depot_index : the index of node which is referred as depot (start and end of the tour)
        :return: the tsp Tour [e1,e2,...,en] with e1 with indexes of graph
        """
        graph = graph.copy()

        # first step -> MST of graph
        mst = nx.minimum_spanning_tree(graph)

        # even
        odd_nodes = Christofides.odd_nodes(mst)

        # induced subgraph of odd nodes
        odd_graph = graph.subgraph(odd_nodes).copy()

        # minimum weighted matching
        perfect_match = Christofides.min_weight_matching(odd_graph)

        # build Eulerian Graph: mst + perfect match
        eu_graph = nx.MultiGraph()
        for e0, e1 in list(mst.edges) + list(perfect_match):
            eu_graph.add_node(e0)
            eu_graph.add_node(e1)
            eu_graph.add_edge(e0, e1, weight=graph.edges[e0, e1]["weight"])

        # Assert a eulerian graph
        assert nx.is_eulerian(eu_graph), "The mst + perfect matching of Christofides -> not an eulerian graph "

        # eulerian tour
        eu_tour = list(nx.eulerian_circuit(eu_graph, source=depot_index, keys=False))

        # shortcut tour to have a 1.5-TSP
        tsp_tour = Christofides.shorted_tour(eu_tour)
        return tsp_tour

    @classmethod
    def min_weight_matching(cls, graph: nx.Graph):
        """
        :param graph: NetworkX Graph -  the graph where compute the minium weight matching .
        :return: a list of edges of the perfect match
        """
        temp_graph = graph.copy()
        # reverse weight to use built-in function of networkx -> max_weight_matching
        for edge in temp_graph.edges():
            temp_graph.edges[edge[0], edge[1]]["weight"] = 1.0 / temp_graph.edges[edge[0], edge[1]]["weight"]

        # list of edges for a perfect matching graph
        return nx.max_weight_matching(temp_graph)

    @classmethod
    def odd_nodes(cls, mst: nx.Graph):
        """

        :param mst: NetworkX Graph -  A minimum spanning tree.
        :return:
        """
        odd_vert = []  # list containing vertices with odd degree
        for i in mst.nodes():
            if mst.degree(i) % 2 != 0:
                odd_vert.append(i)  # if the degree of the vertex is odd, then append it to odd_vert list
        return odd_vert

    @classmethod
    def shorted_tour(cls, eu_tour: list) -> list:
        """
        sub-routine for TSP-algorithm of Christofides. Takes input
        an eulerian tour and remove the nodes already visited in the tour
        and return an instance of tour [e1,e2,...,en].

        :param eu_tour: Takes input an eulerian tour [e1,e2,... ] edges.
        :return: the 1.5-TSP from the shortcut of eulerian tour
        """
        ordered_unique_nodes_in_tour = []
        for node0, node1 in eu_tour:
            if node0 not in ordered_unique_nodes_in_tour:
                ordered_unique_nodes_in_tour.append(node0)

        return ordered_unique_nodes_in_tour

    @classmethod
    def build_tour_from_ordered_nodes(cls, nodes: list):
        """

        :param nodes: a list of nodes
        :return: a tour made of [(node[0], node[1]), (node[1], node[2]), .... , (node[n], node[0])]
        """
        tour = []
        l_nodes = len(nodes)
        for t_i in range(l_nodes):
            if t_i == l_nodes - 1:
                tour += [(nodes[t_i], nodes[0])]
            else:
                tour += [(nodes[t_i], nodes[t_i + 1])]
        return tour


def sample_marker(index):
    MARKERS = ["p", "s", "P", "*", "h", "H", "+", "x", "X", "D", "d", "|", "_", 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, ".", ",", "o", "v", "^", "<", ">", "1", "2", "3", "4", "8"]
    return MARKERS[index]


def sample_pattern(index):
    MARKERS = ['/', '\\', '|', '-', '+', 'x', 'o', 'O', '.', '*'] + ['/o', '\\|', '|*', '-\\', '+o', 'x*', 'o-', 'O|', 'O.', '*-']
    return MARKERS[index]


def sample_line(index):
    MARKERS = ['-', '--', '-.', ':', 'None', ' ', '', 'solid', 'dashed', 'dashdot', 'dotted', 'loosely dotted', 'densely dotted', 'loosely dashed', 'densely dashed', 'loosely dashdotted', 'densely dashdotted', 'loosely dashdotdotted', 'dashdotdotted', 'densely dashdotdotted']
    return MARKERS[index]


def sample_color(index, cmap='tab10'):
    # 1. Choose your desired colormap
    cmap = plt.get_cmap(cmap)

    # 2. Segmenting the whole range (from 0 to 1) of the color map into multiple segments
    colors = [cmap(x) for x in range(cmap.N)]
    assert index < cmap.N

    # 3. Color the i-th line with the i-th color, i.e. slicedCM[i]
    color = colors[index]
    return color
