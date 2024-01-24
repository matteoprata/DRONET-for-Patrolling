from abc import ABCMeta, abstractmethod
from enum import Enum

import networkx as nx
import random
import math


class GraphShapes(Enum):
    random = 0
    circle = 1
    concentric_circle = 2
    double_concentric_circle = 3
    multi_circle = 4
    quad = 5
    concentric_quad = 6
    double_concentric_quad = 7
    multi_quad = 8
    sphere = 9


class DepotPosition(Enum):
    '''  to specify the position of the depot in the graph   '''
    random = 0
    centered = 1
    centered_down = 2
    left_corner = 3
    left_centered = 4
    down_line = 5  # depots alligned in a line, down the graph
    incr_down_line = 6  # incremental method
    all_around = 7  # depots random in a quad around the graph
    incr_all_round = 8  # incremental method
    down_line_partition = 9


class GraphBuilder:
    __metaclass__ = ABCMeta

    def __init__(self, diameter, nnodes, node_weight,
                 depotPosition=DepotPosition.random,
                 graphShape=None, no_edge=False, ndepots=1,
                 points=[], seed=None):
        ''' n: numbers of nodes. (should be > 0 )
    	    diameter: specify the max diameter for the graph. (should be > 0)
	        node_weight: specify the weight of each node, the amount of time
                         required to complete visit that node. (should be > 0)
	        ndepots = indicates multi-depot
            points = allow to defined a set of points to use in the building
            Return a random graph implemented with networkx library, with the
                triangle inequality property, n-nodes and max_diameter diameter.
	    '''
        assert diameter > 0 and nnodes > 0
        assert ndepots >= 1
        self.diameter = diameter * 0.9  # to spacing the depots
        self.seed = seed
        self.grid_side = int(self.diameter / 1.41)
        self.depot_grid_side = int(diameter / 1.41)
        self.nnodes = nnodes
        self.node_weight = node_weight
        self.depotPosition = depotPosition
        self.ndepots = ndepots
        self.depots = self.sample_depots(ndepots)
        self.ind_depots = self.index_depots(ndepots)
        self.graphShape = graphShape
        self.no_edge = no_edge
        self.points = points

    def graph_from_2d_points(self, points, depots):
        # generate the Graph
        G = nx.Graph()
        G.add_nodes_from(range(0, len(points) + len(depots)))

        # add nodes
        for i in range(0, len(points)):
            G.nodes[i]["pos"] = points[i]
            G.nodes[i]["weight"] = self.node_weight
            G.nodes[i]["depot"] = 0

        # add depot with no weight
        for i in range(len(depots)):
            i_node = i + len(points)
            G.nodes[i_node]["pos"] = depots[i]
            G.nodes[i_node]["weight"] = 0
            G.nodes[i_node]["depot"] = 1

        # add depot to all nodes
        points.extend(depots)

        if self.no_edge:  # no add edges and return the graph with only nodes
            return G

        # add weighted edges between node
        for i in range(0, len(points)):
            for j in range(i + 1, len(points)):
                w_ij = math.sqrt(math.pow(points[j][0] - points[i][0], 2)
                                 + math.pow(points[j][1] - points[i][1], 2))
                G.add_edge(i, j, weight=w_ij)

        # the graph, by construction, respect the triangle inequality
        return G

    def build(self):
        G = None
        if self.ndepots <= 1:
            if self.points == []:
                self.points = self.sample_points(self.nnodes - 1)
            G = self.graph_from_2d_points(self.points, self.depots)
        else:
            if self.points == []:
                self.points = self.sample_points(self.nnodes)
            G = self.graph_from_2d_points(self.points, self.depots)
        return G

    @abstractmethod
    def sample_points(self, npoints):
        '''
            return a list of nnodes - 1 lenght. Each element is a
            2-tuple of integer that represente the
            position of a point in a 2d grid.
            return type: list of 2-tuple of int
            return es. [(0,0), (1,0), ... ,(0,1)]
        '''
        pass

    def sample_depots(self, ndepots):
        '''
            return a 2-tuple of integer that represent the
            position in a 2-grid of the depot.
            return type: 2-tuple of integer -> es. (0,0)
        '''
        depots = set()
        while len(depots) < ndepots:
            if self.depotPosition == DepotPosition.centered:
                depots.add((0, 0))

            if self.depotPosition == DepotPosition.left_corner:
                depots.add((-self.depot_grid_side, -self.depot_grid_side))

            if self.depotPosition == DepotPosition.centered_down:
                depots.add((0, -self.depot_grid_side))

            if self.depotPosition == DepotPosition.left_centered:
                depots.add((-self.depot_grid_side, 0))

            if self.depotPosition == DepotPosition.random:
                depots.add(random.randint(-self.depot_grid_side,
                                          self.depot_grid_side),
                           random.randint(-self.depot_grid_side,
                                          self.depot_grid_side))

            if self.depotPosition == DepotPosition.down_line:
                depots.add((random.randint(-self.depot_grid_side,
                                           self.depot_grid_side),
                            -self.depot_grid_side))

            if self.depotPosition == DepotPosition.incr_down_line:
                depots = self.incremental_samples_depots(ndepots,
                                                         y=-self.depot_grid_side)

            if self.depotPosition == DepotPosition.all_around:
                sign = +1 if random.random() > 0.5 else -1
                x = random.randint(-self.depot_grid_side,
                                   self.depot_grid_side)
                y = sign * self.depot_grid_side
                if random.random() > 0.5:
                    depots.add((x, y))
                else:
                    depots.add((y, x))

            if self.depotPosition == DepotPosition.incr_all_round:
                depots_on_each_side = [0, 0, 0, 0]
                for i in range(ndepots):
                    depots_on_each_side[i % 4] += 1

                for sign in [1, -1]:
                    depots |= self.incremental_samples_depots(
                        depots_on_each_side[1 - sign],
                        y=sign * self.depot_grid_side)
                    depots |= self.incremental_samples_depots(
                        depots_on_each_side[sign + 2],
                        x=sign * self.depot_grid_side)

            if self.depotPosition == DepotPosition.down_line_partition:
                step_size = (2 * self.depot_grid_side) / float(ndepots + 1)
                for i in range(1, ndepots + 1):
                    depots.add((-self.depot_grid_side + (step_size) * i, -self.depot_grid_side))

        return list(depots)

    def incremental_samples_depots(self, ndepots, space=20, x=None, y=None):
        """ sample increamental points from the center of
            non selected axis. The input x and y must be in XOR, only one
            at times can be different from None. The axis that is None
            will be used for sampling points from its center,
            from 0 to space*ndepots/2.

        es: if x is None:
            the method samples (0,y), (20,y), (-20,y), (40,y), etc...
        es: if y is None:
            the method samples (x, 0), (x, 20), (x, -20), (x, 40), etc...
        """
        assert ((x is None and y is not None)
                or (x is not None and y is None))

        depots = set()
        sign = 1
        spacing = 0
        for i in range(0, ndepots):
            if x is None:
                depots.add((sign * spacing, y))
            else:
                depots.add((x, sign * spacing))
            sign *= -1
            if i % 2 == 0:
                spacing += space

        return depots

    def index_depots(self, ndepots):
        """ return the index of the depots"""
        if ndepots == 1:
            return [self.nnodes - 1]
        else:
            return [i + self.nnodes
                    for i in range(0, ndepots)]


class RandomGraphBuilder(GraphBuilder):

    def sample_points(self, npoints):
        points = []
        if self.seed is not None:
            random.seed(self.seed)
        # sample points over a 2D grid
        while len(points) < npoints:
            point = (random.randint(-self.grid_side, self.grid_side),
                     random.randint(-self.grid_side, self.grid_side))
            if (point not in points
                    and point not in self.depots):  # no duplicates
                points.append(point)

        return points


class QuadGraphBuilder(GraphBuilder):

    def sample_points(self, npoints):
        side = npoints * 6
        # if the radius go beyond the grid area
        if side > self.grid_side / 2.0:
            side = self.grid_side / 2

        if self.graphShape == GraphShapes.quad:
            return self.sample_points_on_side(npoints, side)
        elif self.graphShape == GraphShapes.concentric_quad:
            points = []
            npoints_1 = int(npoints * (2 / 3.0))
            npoints_2 = npoints - npoints_1
            points += self.sample_points_on_side(npoints_1, side)
            points += self.sample_points_on_side(npoints_2, side / 2)
            return points
        elif self.graphShape == GraphShapes.double_concentric_quad:
            points = []
            npoints_1 = int(npoints * (4 / 7.0))
            npoints_2 = int(npoints * (2 / 7.0))
            npoints_3 = npoints - (npoints_1 + npoints_2)
            points += self.sample_points_on_side(npoints_1, side)
            points += self.sample_points_on_side(npoints_2, side / 2)
            points += self.sample_points_on_side(npoints_3, side / 4)
            return points
        elif self.graphShape == GraphShapes.multi_quad:
            points = []
            npoints_1 = npoints / 2
            npoints_2 = npoints - npoints_1
            points_1 = self.sample_points_on_side(npoints_1, side / 2)
            points_2 = self.sample_points_on_side(npoints_2, side / 2)
            # translate the quad and returns their points
            for p in points_1:
                t_p = (p[0] - side / 2, p[1])  # reduce side to squeeze the circles
                points.append(t_p)
            for p in points_2:
                t_p = (p[0] + side / 2, p[1])
                points.append(t_p)
            return points
        else:
            print("Error: no graphShapes is correct")
            assert False

    def sample_points_on_side(self, npoints, side):
        points = []
        bound = side / 2
        # sample points over a 2D grid
        while len(points) < npoints:
            x = random.randint(-bound, bound)
            y = bound
            if random.random() >= 0.5:  # sign negative
                y = -y

                # reverse point, to cover all the sides
            point = (x, y)
            if len(points) > npoints / 2:
                point = (y, x)

            if (point not in points
                    and point not in self.depots):  # no duplicates
                points.append(point)

        return points


class CircleGraphBuilder(GraphBuilder):

    def sample_points(self, npoints):
        r = npoints * 6
        # if the radius go beyond the grid area
        if r > self.grid_side / 2.0:
            r = self.grid_side / 2

        if self.graphShape == GraphShapes.circle:
            return self.sample_points_on_radius(npoints, r)
        elif self.graphShape == GraphShapes.concentric_circle:
            points = []
            npoints_1 = int(npoints * (2 / 3.0))
            npoints_2 = npoints - npoints_1
            points += self.sample_points_on_radius(npoints_1, r)
            points += self.sample_points_on_radius(npoints_2, r / 2)
            return points
        elif self.graphShape == GraphShapes.double_concentric_circle:
            points = []
            npoints_1 = int(npoints * (4 / 7.0))
            npoints_2 = int(npoints * (2 / 7.0))
            npoints_3 = npoints - (npoints_1 + npoints_2)
            points += self.sample_points_on_radius(npoints_1, r)
            points += self.sample_points_on_radius(npoints_2, r / 2)
            points += self.sample_points_on_radius(npoints_3, r / 4)
            return points
        elif self.graphShape == GraphShapes.multi_circle:
            points = []
            npoints_1 = npoints / 2
            npoints_2 = npoints - npoints_1
            points_1 = self.sample_points_on_radius(npoints_1, r / 2)
            points_2 = self.sample_points_on_radius(npoints_2, r / 2)
            # translate the circles and returns their points
            for p in points_1:
                t_p = (p[0] - r, p[1])  # reduce r to squeeze the circles
                points.append(t_p)
            for p in points_2:
                t_p = (p[0] + r, p[1])
                points.append(t_p)
            return points
        else:
            print("Error: no graphShapes is correct")
            assert False

    def sample_points_on_radius(self, npoints, radius):
        points = []
        # sample points over a 2D grid
        while len(points) < npoints:
            x = random.randint(-radius, radius)
            y = int(math.sqrt(radius ** 2 - (x ** 2)))
            if random.random() >= 0.5:  # sign negative
                y = -y
            point = (x, y)
            if (point not in points
                    and point not in self.depots):  # no duplicates
                points.append(point)

        return points


class SphereGraphBuilder(GraphBuilder):
    '''
        a Graph with a sphere shape where
        all the points respect the follow equation: x^2 + y^2 <= r^2
    '''

    def sample_points(self, npoints):
        r = npoints * 6
        # if the radius go beyond the grid area
        if r > self.grid_side / 2.0:
            r = self.grid_side / 2

        # square of the radius -> used in the equation of points
        r_2 = r ** 2

        points = []
        # sample points over a 2D grid
        while len(points) < npoints:
            x = random.randint(-r, r)
            y_bound = int(math.sqrt(r_2 - (x ** 2)))
            # in order to have the point always inside the circumference
            y = random.randint(-y_bound, y_bound)
            point = (x, y)
            if (point not in points
                    and point not in self.depots):  # no duplicates
                points.append(point)

        return points


