import networkx as nx
import matplotlib.pyplot as plt
import random
import sys

from argparse import ArgumentParser
from src.patrolling.SOTA.util import rndgraph, utility


def multipath(graph, roots, drones):
    """ compute multipath for each drone
        assuming len(roots) == len(drones)
        and that drones[i] start at roots[i]
        the roots are not points! only the
        ith-drone can visit the ith-root

        return a list of len(drones) elements
        each element i-th is a list of tours for the
        drones i-th that start and end at the depot i-th

        the tours are made by utility.Tour class
    """
    assert len(roots) == len(drones)
    multi_tours = []
    for i in range(len(drones)):
        depot = roots[i]
        drone = drones[i]
        single_depot_graph = uniqdepot_graph(graph,  # remove other roots
                                             roots[i], roots)
        multi_tours.append(multipath_subroutine(single_depot_graph,
                                                depot, drone.autonomy))

    return multi_tours


def uniqdepot_graph(graph, uniq_depot, roots):
    """ creates a copy of the input graph and remove
        all the roots in input except the uniq_depot

        return the new graph with only a root -> uniq_depot
    """
    graph_out = graph.copy()
    graph_out.remove_nodes_from(
        [r for r in roots if r != uniq_depot])
    return graph_out


def multipath_subroutine(graph, depot, bound):
    """ compute the possible tours in the graph
        from the depot with at most the input bound
        the path are splitted away from the single tsp on
        graph, without the not reachable nodes due to
        constraint bound

        return a list of tours which are made by utility.Tour class
    """
    G = graph.copy()
    tours = [utility.Tour(graph, [], depot)]  # contains always empty tour
    for n in graph.nodes():
        if (n != depot and
                graph.nodes[n]["weight"]
                + graph.edges[depot, n]["weight"] > bound):
            G.remove_node(n)  # remove non reachable nodes

    if G.number_of_nodes() < 2:
        return tours

    tsp = utility.christofides(G, depot)
    tsp_nodes = [x[0] for x in tsp]
    assert len(set(tsp_nodes)) == G.number_of_nodes()

    n = len(tsp_nodes)
    for i in range(1, n):  # index of first nodes of sub-tour
        current_subtour = [(depot, tsp_nodes[i])]
        if not add_tour(graph, tours, current_subtour,
                        (tsp_nodes[i], depot), bound):
            continue
        for j in range(i + 1, n):  # last nodes in sub-tour
            current_subtour += [(tsp_nodes[j - 1], tsp_nodes[j])]
            if not add_tour(graph, tours, current_subtour,
                            (tsp_nodes[j], depot), bound):
                break  # bigger subtours will have cost > bound

    return tours


def add_tour(graph, tours, current_subtour, return_edge, bound):
    """ add the returning edge to the current_subtour (tour without
        the returning edge), check if the size is less then bound
        if true add the tour to tours list and return True
        otherwise return False
    """
    tour = utility.Tour(graph, current_subtour + [return_edge])
    if tour.cost > bound:
        return False
    tours.append(tour)
    return True



