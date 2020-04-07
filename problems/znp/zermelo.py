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
    V[:, :, 0] = m * y ** 3 + c
    return V


def fun_shear_V(x, y, cx, cy):
    V = np.zeros((x.shape[0], x.shape[1], 2))
    V[:, :, 0] = (x - 1) / ((x - 1) ** 2 + y ** 2 + 0.1) - (x + 1) / ((x + 1) ** 2 + y ** 2 + 0.1) + cx
    V[:, :, 1] = y / ((x - 1) ** 2 + y ** 2 + 0.1) - y / ((x + 1) ** 2 + y ** 2 + 0.1) + cy
    return V


def stoch_shear(x, y):
    u = 0.1 * y * np.random.randn(*x.shape) + y
    v = 0.1 * y * np.random.randn(*x.shape)
    return u, v


def stoch_func(ku, kv):
    def stoch_shear(x, y):
        #u = ku * y * np.random.randn(*x.shape) + y
        #v = kv * y * np.random.randn(*x.shape) + 0
        u = ku * np.random.randn(*x.shape) / (x+1) + y
        v = kv * y**2 * np.random.randn(*x.shape) + 0
        return u, v
    return stoch_shear


def stoch_constraints(S, V, maxw=0, n_samp=1, **kwargs):
    """Constraint Fitness Calculation (with Monte Carlo Sampling)
        Given S, V, maxw, n_samp
        1. Calculate w n_samp times
        2. Calculate maximum speed violation across all simulations."""
    assert callable(V), "V must be a function for Monte Carlo Simulation"
    # S'
    S_prime = np.gradient(S, axis=1)
    # Get the number, length and Dimension of solutions
    N, L, Dim = S.shape
    # Break out the X and Y coordinates of S
    X = S[:, :, 0].flatten()
    Y = S[:, :, 1].flatten()
    V_s = [V(X, Y) for _ in range(n_samp)]
    u_s = np.concatenate([i[0].reshape((N, L, 1)) for i in V_s],axis=2)
    v_s = np.concatenate([i[1].reshape((N, L, 1)) for i in V_s],axis=2)
    s_x = np.expand_dims(S_prime[:,:,0], 2)
    w_x = s_x - u_s
    s_y = np.expand_dims(S_prime[:,:,1], 2)
    w_y = s_y - v_s
    w_mag = np.sqrt((w_x**2 + w_y**2))
    # Constraint violation score
    c = np.where(maxw - w_mag < 0, w_mag - maxw, 0).sum(axis=1).sum(axis=1).reshape((N, 1))
    return c


def stoch_fitness(S, V, dt=1, n_samp=1, **kwargs):
    """Fitness Calculation with Monte Carlo simulation
    Given S, V and n_samp
    1. Calculate S'
    2. Calculate w(S) = S' + V(S)
    3. D = int_c(S)ds
    4. E = int_c(||w||^2)ds
    return D, E"""
    assert callable(V), "V must be a function for Monte Carlo Simulation"
    # S'
    S_prime = np.gradient(S, axis=1)
    # Get the number, length and Dimension of solutions
    N, L, Dim = S.shape
    # Break out the X and Y coordinates of S
    X = S[:, :, 0].flatten()
    Y = S[:, :, 1].flatten()
    V_s = [V(X, Y) for _ in range(n_samp)]
    u_s = np.concatenate([i[0].reshape((N, L, 1)) for i in V_s], axis=2)
    v_s = np.concatenate([i[1].reshape((N, L, 1)) for i in V_s], axis=2)
    s_x = np.expand_dims(S_prime[:, :, 0], 2)
    w_x = s_x - u_s
    s_y = np.expand_dims(S_prime[:, :, 1], 2)
    w_y = s_y - v_s
    # curve integral of all solutions in S
    D = np.sqrt((np.diff(S, axis=1) ** 2).sum(axis=-1)).sum(axis=-1).reshape((N, 1))
    # energy integral of all solutions in w
    E_mu = (w_x**2 + w_y**2).sum(axis=1).mean(axis=-1).reshape((N, 1))
    # energy integral of all solutions in w
    E_sig = (w_x ** 2 + w_y ** 2).sum(axis=1).std(axis=-1).reshape((N, 1))
    return np.hstack((D, E_mu, E_sig))
    #return np.hstack((E_mu, E_sig))



def constraint(S, V, maxw=0, **kwargs):
    """Constraint Fitness Calculation
    Given S, V, maxw
    1. Calculate w
    2. Calculate maximum speed violation."""
    # S'
    S_prime = np.gradient(S, axis=1)
    # Get the number, length and Dimension of solutions
    N, L, Dim = S.shape
    # Break out the X and Y coordinates of S
    X = S[:, :, 0].flatten()
    Y = S[:, :, 1].flatten()
    if type(V) is np.ndarray:
        # Break out the u and v components of V
        u = V[:, :, 0]
        v = V[:, :, 1]
        u_s = u[Y, X].reshape((N, L, 1))
        v_s = v[Y, X].reshape((N, L, 1))
        # V(s)
        V_s = np.concatenate((u_s, v_s), axis=-1)
    elif callable(V):
        u_s, v_s = V(X, Y)
        u_s = u_s.reshape((N, L, 1))
        v_s = v_s.reshape((N, L, 1))
        V_s = np.concatenate((u_s, v_s), axis=-1)
    # w
    w = S_prime - V_s
    # Speed magnitude
    w_mag = np.sqrt((w ** 2).sum(axis=2))
    # Constraint violation score
    c = np.where(maxw - w_mag < 0, w_mag - maxw, 0).sum(axis=1).reshape((N, 1))
    return c


def fitness(S, V, dt=1, **kwargs):
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
    if type(V) is np.ndarray:
        # Break out the u and v components of V
        u = V[:, :, 0]
        v = V[:, :, 1]
        u_s = u[Y, X].reshape((N, L, 1))
        v_s = v[Y, X].reshape((N, L, 1))
        # V(s)
        V_s = np.concatenate((u_s, v_s), axis=-1)
    elif callable(V):
        u_s, v_s = V(X, Y)
        u_s = u_s.reshape((N, L, 1))
        v_s = v_s.reshape((N, L, 1))
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
    WIDTH = 61
    DEPTH = 15
    START = 30
    FINISH = 30
    # The constraints
    maxw = np.inf
    # Number of solutions
    N = 50
    # Pareto Front Length (Need to keep equal to N for now).
    NP = N
    # For stochastic functions, how many simulations to run
    n_samp = 10
    # Gain on the u std. deviation.
    ku = 0.5
    # Gain on the v std. deviation.
    kv = 0.1
    # Start up the shear function
    shear_f = stoch_func(ku, kv)
    # X coordinate corresponds to horizontal position in water column
    X = np.array(range(WIDTH))
    # Y coordinate corresponds to altitude position in water column
    Y = np.array(range(DEPTH))
    # A tile has coordinates (x, y).
    # GRID is a list, element 0 is the x coordinate of a tile. Element 1 is the y coordinate of a tile.
    GRID_X, GRID_Y = np.meshgrid(X, Y)
    # The velocity vector field, comprised of u and v components
    # V = linear_shear_V(GRID_X, GRID_Y, 1, 0)
    # V = constant_V(GRID_X, GRID_Y, 1, -np.pi/4)
    # V = cubic_shear_V(GRID_X, GRID_Y, 1, 0)
    # V = fun_shear_V(GRID_X, GRID_Y, 1, 0)
    V = stoch_shear(GRID_X, GRID_Y)
    V = np.concatenate([np.expand_dims(i, 2) for i in V], axis=2)
    #V = linear_shear_V(GRID_X, GRID_Y, 1, 0)
    # Vector Field
    COST = np.concatenate((np.expand_dims(GRID_X, 2), np.expand_dims(GRID_Y, 2), V), axis=2)
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
    # The Solver Class (Deterministic)
    #solver = ndsa1.NDSA1(N, fitness=fitness, constraint=constraint)
    # The Solver Class (Stochastic)
    solver = ndsa1.NDSA1(N, fitness=stoch_fitness, constraint=stoch_constraints)
    # The Logger Class
    outfile = Path("./results").resolve()
    outfile.mkdir(parents=True, exist_ok=True)
    f = list(outfile.glob("log_run*"))
    outfile = outfile.joinpath("log_run_{:03d}.mp4".format(len(f) + 1))
    # If you want a video logged, then pass FFMpegWriter keyword arguments and an outfile keyword for the video output path.
    logger = utils.ResultsManager(fps=30, outfile=str(outfile))
    # If you don't want a video logged, then don't pass arguments to ResultsManager
    #logger = utils.ResultsManager()
    # Solution Display
    outfile = Path("./results").resolve()
    outfile.mkdir(parents=True, exist_ok=True)
    f = list(outfile.glob("sol_run*"))
    outfile = outfile.joinpath("sol_run_{:03d}.mp4".format(len(f) + 1))
    # If you want a video logged, then pass FFMpegWriter keyword arguments and an outfile keyword for the video output path.
    sol = utils.ResultPlotter(NP, COST, fps=30, outfile=str(outfile))
    # If you don't want a video logged, then don't pass arguments to ResultPlotter
    #sol = utils.ResultPlotter(NP, COST)
    GBEST = None
    rankings = None
    fitnesses = None
    for i in range(1000):
        # Get the next generation of chromosomes, their fitnesses and constraints
        V = shear_f(GRID_X, GRID_Y)
        V = np.concatenate([np.expand_dims(i, 2) for i in V], axis=2)
        # Vector Field
        COST = np.concatenate((np.expand_dims(GRID_X, 2), np.expand_dims(GRID_Y, 2), V), axis=2)
        #V = linear_shear_V(GRID_X, GRID_Y, 1, 0)
        # Deterministic Example
        #chromosomes, fitnesses, constraints = solver.update(chromosomes, V=V, bounds=bounds, lineq=(A, b), maxw=maxw)
        # Stochastic Function Example
        chromosomes, fitnesses, constraints = solver.update(chromosomes, V=shear_f, bounds=bounds, lineq=(A, b), maxw=maxw, n_samp=n_samp)
        # If the GBEST population hasn't been initialised
        if GBEST is None:
            # Get the best performers
            rankings = ndsa1.cndsa2(fitnesses, constraints)
            # Update the archives solution, fitness and constraints
            GBEST = chromosomes[rankings[0], :, :]
            GFBEST = fitnesses[rankings[0], :]
            GCBEST = constraints[rankings[0], :]
        # Otherwise, perform survival on the archive and latest batch
        else:
            tmpS = np.vstack((GBEST, chromosomes))
            tmpF = np.vstack((GFBEST, fitnesses))
            tmpC = np.vstack((GCBEST, constraints))
            rankings = ndsa1.cndsa2(tmpF, tmpC)
            sur_idx = solver._survival(tmpS, rankings, tmpF)
            GBEST = tmpS[sur_idx, :, :]
            GFBEST = tmpF[sur_idx, :]
            GCBEST = tmpC[sur_idx, :]
        print("Iteration {:04d}\tGBEST Size {:03d}".format(i + 1, GBEST.shape[0]))
        sol.update(GBEST, cost=COST)
        #sol.update(GBEST)
        logger.update(np.hstack((GFBEST,GCBEST)), ["Distance","Energy Mean", "Energy Std. Dev.", "Speed Violation"], linestyle="None", marker=".", markersize=10, color="green")
        #logger.update(np.hstack((GFBEST,GCBEST)), ["Distance", "Energy", "Speed Violation"], linestyle="None", marker=".", markersize=10, color="green")
    if sol.flag:
        sol.finish()
    if logger.flag:
        logger.finish()
