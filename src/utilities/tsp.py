""" Traveling salesman problem instance"""

"""Author: William Borgeaud <williamborgeaud@gmail.com>"""

import numpy as np
from scipy.spatial.distance import pdist, squareform
import matplotlib.pyplot as plt

""" 2-opt heuristic for TSP """

import numpy as np

""" Nearest neighbor heuristic for TSP """


class NN_solver:
    """ Class for Nearest Neighbor Solver """

    def __init__(self, starting_point='best'):
        """
        Parameters
        ----------
        starting_point : int or str
                         The starting node for the solution. If starting_point
                         is 'best' returns the best solution over all nodes.
        """
        self.starting_point = starting_point
        if starting_point == 'best':
            self.starting_point = -1

    def solve(self, tsp):
        ans = np.inf
        N = tsp.N
        wanted = self.starting_point
        mat = tsp.mat
        if wanted != -1:
            visited = set(range(N))
            s = wanted
            hist = [s]
            costs = 0
            new_mat = np.copy(mat[:, :])
            visited.remove(wanted)
            while len(visited) > 0:
                new_mat[:, s] = np.inf
                t = np.argmin(new_mat[s])
                hist.append(t)
                costs += new_mat[s, t]
                visited.remove(t)
                s = t
            costs += mat[t, wanted]
            hist.append(wanted)
            ans = costs
        else:
            for i in range(N):
                visited = set(range(N))
                s = i
                hist = [s]
                costs = 0
                new_mat = np.copy(mat[:, :])
                visited.remove(i)
                while len(visited) > 0:
                    new_mat[:, s] = np.inf
                    t = np.argmin(new_mat[s])
                    hist.append(t)
                    costs += new_mat[s, t]
                    visited.remove(t)
                    s = t
                costs += mat[t, i]
                hist.append(i)
                if wanted == i:
                    return hist
                if costs < ans:
                    ans = costs
                    best_hist = hist
            hist = best_hist
        return hist


def get_cost(tour, tsp):
    ans = 0
    for i in range(len(tour) - 1):
        ans += tsp.mat[tour[i], tour[i + 1]]
    return ans


class TwoOpt_solver:

    def __init__(self, initial_tour, iter_num=100):
        """
        Parameters
        ----------
        initial_tour : permutation of the nodes
                       Starting tour on which to apply 2-opt
        iter_num : int
                   Number of iterations in the local 2-opt search
        """
        self.initial_tour = initial_tour
        self.iter_num = iter_num

    def solve(self, tsp):
        if self.initial_tour == 'NN':
            self.initial_tour = NN_solver().solve(tsp)
        best_tour = self.initial_tour
        old_best = np.inf
        for _ in range(self.iter_num):
            best = get_cost(best_tour, tsp)
            tour = best_tour[:]
            for i in range(tsp.N):
                maxx = tsp.N if i != 0 else tsp.N - 1
                for j in range(i + 2, maxx):
                    ftour = tour[:]
                    j1 = j + 1
                    if tsp.mat[ftour[i], ftour[i + 1]] + tsp.mat[ftour[j], ftour[j1]] > tsp.mat[
                        ftour[i], ftour[j]] + tsp.mat[ftour[i + 1], ftour[j1]]:
                        ftour[i + 1: i + j - i + 1] = list(reversed(ftour[i + 1:j1]))
                        cost = get_cost(ftour, tsp)
                        if cost < best:
                            best = cost
                            best_tour = ftour
            if best == old_best:
                return best_tour
            else:
                old_best = best
        return best_tour


class TSP:
    """ Base class for a TSP instance"""

    def __init__(self):
        self.tours = {}
        self.lower_bounds = {}

    def read_mat(self, mat):
        """ Reads a distance matrix
        Parameters
        ----------
        mat : NxN numpy matrix
              Distance matrix for the TSP instance.
        """
        self.N = len(mat)
        self.mat = mat
        for i in range(len(mat)):
            self.mat[i, i] = np.inf

    def read_data(self, data, dist='euclidean'):
        """ Reads a data matrix
        Parameters
        ----------
        data : NxP numpy matrix
               Data matrix containing the N P-dimensional data points

        dist : f: x,y -> float
               Distance function to use in the TSP instance.
        """
        self.data = data
        mat = squareform(pdist(data, dist))
        self.read_mat(mat)

    def plot_data(self):
        """ Plots the data if it has been specified"""
        if hasattr(self, 'data'):
            plt.scatter(*self.data.T)
            plt.show()
        else:
            raise Exception('No 2d data of the instance has been loaded')

    def get_approx_solution(self, solver, star_node=None):
        """ Compute an approximate solution of the instance
        Parameters
        ----------
        solver : TSP solver
                 Instance of a TSP solver class in the module solvers.

        Returns
        ----------
        A permutation of the nodes giving an approximate solution to the
        instance.
        """
        tour = solver.solve(self)
        if star_node is not None:
            start_node_index = tour.index(star_node)
            tour = tour[:-1]
            tour = tour[start_node_index:] + tour[:start_node_index]
            tour = tour + [tour[0]]

        cost = get_cost(tour, self)
        # print('The cost is {}.'.format(get_cost(tour,self)))
        self.tours[solver.__class__.__name__] = tour
        return tour, cost

    def plot_solution(self, which):
        """ Plots a solution"""
        if isinstance(which, int):
            which = list(self.tours.keys())[which]
        tour = self.tours[which]
        plt.scatter(*self.data.T)
        for i in range(self.N):
            plt.plot(*self.data[[tour[i], tour[i + 1]]].T, 'b')
        plt.show()

    def get_lower_bound(self, method):
        """ Compute a lower bound of the instance
        Parameters
        ----------
        method : Lower bound method
                 Instance of a lower bound class in the module lower_bounds.

        Returns
        ----------
        A lower bound of the instance.
        """
        sol = method.bound(self)
        print('The lower bound is {}'.format(sol['primal objective']))
        self.lower_bounds[method.__class__.__name__] = sol['primal objective']
        return sol

    def get_best_solution(self):
        """ Returns the best solution computed so far"""
        if not self.tours:
            raise Exception('No solution has been computed yet')
        scores = {s: get_cost(self.tours[s], self) for s in self.tours}
        best = min(scores, key=scores.get)
        print('The best solution is given by {} with score {}'.format(best, scores[best]))
        return self.tours[best]

    def get_best_lower_bound(self):
        """ Returns the best lower bound computed so far"""
        if not self.tours:
            raise Exception('No lower bound has been computed yet')
        best = max(self.lower_bounds, key=self.lower_bounds.get)
        print('The best lower bound is given by {} with score {}'.format(best, self.lower_bounds[best]))
        return self.lower_bounds[best]
