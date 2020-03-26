import argparse

import numpy as np
from scipy.spatial.distance import cdist

from mogapy import ndsa1, utils
import matplotlib.pyplot as plt


def plot_results(solutions, cost):
    plt.matshow(cost)
    for solution in solutions:
        plt.plot(range(cost.shape[1]), solution)


def fitness(solutions, cost):
    # The transect is a range from 0 to GRID_HEIGHT-1
    Y = np.array(range(cost.shape[1]))
    # There are two fitnesses, time and energy
    fitnesses = np.zeros((solutions.shape[0], 2))
    for i, X in enumerate(solutions):
        # The time cost is proportional to the distance travelled along the path.
        fitnesses[i, 0] = np.diag(cdist(np.vstack((X[:-1], Y[:-1])).T,
                                        np.vstack((X[1:], Y[1:])).T)).sum()
        # The estimated grid frame speed (x component)
        # Negative U means travelling downstream
        U = np.hstack((0, -np.diff(X)))
        # The current velocity at each visited tile
        v = cost[X, Y]
        # The effort is U*v with a condition that U*v == 0 is v
        fitnesses[i, 1] = np.where(U * v == 0, v, U * v).sum()
    return fitnesses


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
    logger = utils.ResultsManager()
    # Number of generations
    for i in range(10):
        print("Iteration {:04d}".format(i + 1))
        chromosones, fitnesses = solver.update(chromosones, cost=CURRENT[1])
        logger.update(fitnesses, ["Distance", "Energy"], linestyle="None", marker=".", markersize=10, color="green")
    plot_results(chromosones, CURRENT[1])