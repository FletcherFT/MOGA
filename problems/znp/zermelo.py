import argparse

import matplotlib.pyplot as plt
import numpy as np
from numpy.matlib import repmat

from mogapy import ndsa1, utils

np.random.seed(42)


def fitness(S, V, **kwargs):
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
    u = V[:, :, 0]
    v = V[:, :, 1]
    u_s = u[Y, X].reshape((N, L, 1))
    v_s = v[Y, X].reshape((N, L, 1))
    # V(s)
    V_s = np.concatenate((u_s, v_s), axis=-1)
    # w
    w = S_prime - V_s
    # curve integral of all solutions in S
    D = np.sqrt((np.diff(S, axis=1) ** 2).sum(axis=-1)).sum(axis=-1).reshape((N, 1))
    # energy integral of all solutions in w
    E = (w ** 2).sum(axis=-1).sum(axis=-1).reshape((N, 1))
    return np.hstack((D, E))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Script to build and find optimal solutions for Zermelo's Navigation Problem.")
    args = parser.parse_args()
    # Build the search space
    WIDTH = 301
    DEPTH = 101
    START = 150
    FINISH = 150
    # X coordinate corresponds to horizontal position in water column
    X = np.array(range(WIDTH))
    # Y coordinate corresponds to altitude position in water column
    Y = np.array(range(DEPTH))
    # A tile has coordinates (x, y).
    # GRID is a list, element 0 is the x coordinate of a tile. Element 1 is the y coordinate of a tile.
    GRID_X, GRID_Y = np.meshgrid(X, Y)
    # The velocity vector field, comprised of u and v components
    # u component varies with y only
    # v component is constant 0
    V = np.stack((GRID_Y, np.zeros((DEPTH, WIDTH))), axis=2)
    # Number of solutions
    N = 50
    # Initialise the solutions array, alleles for the moment are just random elements in X
    chromosomes = np.stack((np.random.randint(0, WIDTH, (N, DEPTH), np.int),
                            repmat(np.expand_dims(np.arange(DEPTH), 0), N, 1)), axis=2)
    chromosomes[0, :, 0] = START
    # Define bounds
    bounds = np.array([[0, WIDTH - 1],
                       [0, DEPTH - 1]])
    # Define linear equality constraints
    A = np.zeros((2, DEPTH, 2))
    A[0, 0, 0] = 1
    A[1, -1, 0] = 1
    b = np.array([[[START, 0],
                   [FINISH, 0]]])
    # Enforce starting X condition
    chromosomes[:, 0, 0] = START
    # Enforce finishing X condition
    chromosomes[:, -1, 0] = FINISH
    # The Solver Class
    solver = ndsa1.NDSA1(N, fitness=fitness)
    # The Logger Class
    logger = utils.ResultsManager()
    # Solution Display
    sol = utils.ResultPlotter()
    # Vector Field
    COST = np.concatenate((np.expand_dims(GRID_X, 2), np.expand_dims(GRID_Y, 2), V), axis=2)
    # Number of generations
    for i in range(10000):
        print("Iteration {:04d}".format(i + 1))
        chromosomes, fitnesses = solver.update(chromosomes, V=V, bounds=bounds, lineq=(A, b))
        #logger.update(fitnesses, ["Distance", "Energy"], linestyle="None", marker=".", markersize=10, color="green")
        #sol.update(chromosomes, COST)
    logger.update(fitnesses, ["Distance", "Energy"], linestyle="None", marker=".", markersize=10, color="green")
    sol.update(chromosomes, COST)
