import argparse

import numpy as np
from scipy.spatial.distance import pdist

from mogapy import ndsa1, utils
import matplotlib.pyplot as plt


def plot_results(solutions, cost):
    plt.matshow(cost)
    for solution in solutions:
        plt.plot(range(cost.shape[1]), solution)


def fitness(S, V):
    """Fitness Calculation
    Given S, V
    1. Calculate S'
    2. Calculate w(S) = S' + V(S)
    3. D = int_c(S)ds
    4. E = int_c(||w||^2)ds
    return D, E"""
    # S'
    S_prime = np.gradient(S, axis=1)
    # Get the number, length and Dimension of solutions
    N, L, Dim = S.shape
    # Break out the X and Y coordinates of S
    X = S[:, :, 0].flatten()
    Y = S[:, :, 1].flatten()
    # Break out the u and v components of V
    u = V[:, :, 0].squeeze()
    v = V[:, :, 1].squeeze()
    u_s = u[X, Y].reshape((N, L, 1))
    v_s = v[X, Y].reshape((N, L, 1))
    # V(s)
    V_s = np.concatenate((u_s, v_s), axis=-1)
    # w
    w = S_prime + V_s
    # curve integral of all solutions in S
    D = np.sqrt((np.diff(S, axis=1)**2).sum(axis=-1)).sum(axis=-1).reshape((N,1))
    # energy integral of all solutions in w
    E = (w**2).sum(axis=-1).sum(axis=-1)
    return D, E


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Script to build and find optimal solutions for Zermelo's Navigation Problem.")
    args = parser.parse_args()
    # Build the search space
    GRID_WIDTH = 101
    GRID_HEIGHT = 101
    START = 50
    FINISH = 50
    X = np.array(range(GRID_WIDTH))
    Y = np.array(range(GRID_HEIGHT))
    # A tile has coordinates (x, y).
    # GRID is a list, element 0 is the x coordinate of a tile. Element 1 is the y coordinate of a tile.
    GRID = np.meshgrid(X, Y)
    # CURRENT is a list, element 0 is the u velocity of a tile. Element 1 is the v velocity of a tile.
    CURRENT = [np.zeros((GRID_WIDTH, GRID_HEIGHT)),
               GRID[0]]
    # Number of solutions
    N = 100
    # Initialise the solutions array, contains GRID_HEIGHT alleles.
    chromosones = np.random.randint(0, GRID_HEIGHT - 1, (N, GRID_WIDTH), np.int)
    # Enforce starting condition
    chromosones[:, 0] = START
    # Enforce finishing condition
    chromosones[:, -1] = FINISH
    # A test case, a straight line
    chromosones[0, :] = START
    # The Solver Class
    solver = ndsa1.NDSA1(N, fitness=fitness)
    # The Logger Class
    objective_logger = utils.ResultsManager()
    constraint_logger = utils.ResultsManager()
    # Number of generations
    for i in range(1000):
        print("Iteration {:04d}".format(i + 1))
        chromosones, fitnesses, constraints = solver.update(chromosones, cost=CURRENT[1], maxdx=1)
        objective_logger.update(fitnesses, ["Distance", "Energy"], linestyle="None", marker=".", markersize=10, color="green")
        constraint_logger.update(constraints, ["Overtravel"], linestyle="None", marker=".", markersize=10, color="green")
    plot_results(chromosones, CURRENT[1])
