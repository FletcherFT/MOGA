import argparse
from pathlib import Path

import numpy as np
from numpy.matlib import repmat

from mogapy import ndsa1, utils

np.random.seed(42)


def constant_V(x, y, c, theta):
    V = np.ones((x.shape[0], x.shape[1], 2))
    V[:, :, 0] = c * np.cos(theta)
    V[:, :, 1] = c * np.sin(theta)
    return V


def linear_shear_V(x, y, m, c):
    V = np.zeros((x.shape[0], x.shape[1], 2))
    V[:, :, 0] = m * y + c
    return V

def cubic_shear_V(x, y, m, c):
    V = np.zeros((x.shape[0], x.shape[1], 2))
    V[:, :, 0] = m * y**3 + c
    return V

def fun_shear_V(x, y, cx, cy):
    V = np.zeros((x.shape[0], x.shape[1], 2))
    V[:, :, 0] = (x - 1) / ((x - 1) ** 2 + y ** 2 + 0.1) - (x + 1) / ((x + 1) ** 2 + y ** 2 + 0.1) + cx
    V[:, :, 1] = y / ((x - 1) ** 2 + y ** 2 + 0.1) - y / ((x + 1) ** 2 + y ** 2 + 0.1) + cy
    return V


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
    # Speed overstepping
    w_mag = ((np.sqrt((w**2).sum(axis=2))) > 5).sum(axis=1).reshape((N,1))
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
    WIDTH = 61
    DEPTH = 15
    START = 30
    FINISH = 30
    # X coordinate corresponds to horizontal position in water column
    X = np.array(range(WIDTH))
    # Y coordinate corresponds to altitude position in water column
    Y = np.array(range(DEPTH))
    # A tile has coordinates (x, y).
    # GRID is a list, element 0 is the x coordinate of a tile. Element 1 is the y coordinate of a tile.
    GRID_X, GRID_Y = np.meshgrid(X, Y)
    # The velocity vector field, comprised of u and v components
    #V = linear_shear_V(GRID_X, GRID_Y, 3, 0)
    #V = constant_V(GRID_X, GRID_Y, 1, -np.pi/4)
    #V = cubic_shear_V(GRID_X, GRID_Y, 1, 0)
    V = fun_shear_V(GRID_X, GRID_Y, 1, 0)
    # Vector Field
    COST = np.concatenate((np.expand_dims(GRID_X, 2), np.expand_dims(GRID_Y, 2), V), axis=2)
    # Number of solutions
    N = 50
    # Pareto Front Length
    NP = N
    # Initialise the solutions array, alleles for the moment are just random elements in X
    chromosomes = np.stack((np.random.randint(0, WIDTH, (N, DEPTH), np.int),
                            repmat(np.expand_dims(np.arange(DEPTH), 0), N, 1)), axis=2)
    # Straight line seed
    #chromosomes[0, :, 0] = START
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
    outfile = Path("./results").resolve()
    outfile.mkdir(parents=True, exist_ok=True)
    f = list(outfile.glob("log_run*"))
    outfile = outfile.joinpath("log_run_{:03d}.mp4".format(len(f) + 1))
    # If you want a video logged, then pass FFMpegWriter keyword arguments and an outfile keyword for the video output path.
    #logger = utils.ResultsManager(fps=60, outfile=str(outfile))
    # If you don't want a video logged, then don't pass arguments to ResultsManager
    logger = utils.ResultsManager()
    # Solution Display
    outfile = Path("./results").resolve()
    outfile.mkdir(parents=True, exist_ok=True)
    f = list(outfile.glob("sol_run*"))
    outfile = outfile.joinpath("sol_run_{:03d}.mp4".format(len(f) + 1))
    # If you want a video logged, then pass FFMpegWriter keyword arguments and an outfile keyword for the video output path.
    #sol = utils.ResultPlotter(NP, COST, fps=60, outfile=str(outfile))
    # If you don't want a video logged, then don't pass arguments to ResultPlotter
    sol = utils.ResultPlotter(NP, COST)
    # Number of generations
    GBEST = None
    for i in range(3000):
        chromosomes, fitnesses = solver.update(chromosomes, V=V, bounds=bounds, lineq=(A, b))
        if GBEST is None:
            rankings = ndsa1.ndsa(fitnesses)
            GBEST = chromosomes[rankings[0], :, :]
            GFBEST = fitnesses[rankings[0], :]
        else:
            tmpS = np.vstack((GBEST, chromosomes))
            tmpF = np.vstack((GFBEST, fitnesses))
            rankings = ndsa1.ndsa(tmpF)
            GBEST = tmpS[rankings[0], :, :]
            GFBEST = tmpF[rankings[0], :]
            if GBEST.shape[0] > NP:
                d = ndsa1.crowding_distance([[0 for _ in range(GBEST.shape[0])]], GFBEST)
                sur_idx = np.flipud(np.argsort(d))
                sur_idx = sur_idx[:NP]
                GBEST = GBEST[sur_idx, :, :]
                GFBEST = GFBEST[sur_idx, :]
        print("Iteration {:04d}\t GBEST Size: {:03d}".format(i + 1, GBEST.shape[0]))
        sol.update(GBEST)
        logger.update(GFBEST, ["Distance", "Energy"], linestyle="None", marker=".", markersize=10, color="green")
    if sol.flag:
        sol.finish()
    if logger.flag:
        logger.finish()
