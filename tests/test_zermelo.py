import numpy as np
import matplotlib.pyplot as plt

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
    E = (w ** 2).sum(axis=-1).sum(axis=-1).reshape((N,1))
    return np.hstack((D, E))


GRID_WIDTH = 5
GRID_HEIGHT = 5
X = np.array(range(GRID_WIDTH))
Y = np.array(range(GRID_HEIGHT))
# A tile has coordinates (x, y).
# GRID is a list, element 0 is the x coordinate of a tile. Element 1 is the y coordinate of a tile.
GRID = np.meshgrid(X, Y)
V = np.stack((np.zeros((GRID_WIDTH, GRID_HEIGHT)), GRID[0]), axis=2)
S = np.stack((np.array([range(5), [1, 1, 1, 1, 1]]).T,
              np.array([[1, 1, 1, 1, 1], range(5)]).T,
              np.array([[0, 0, 0, 0, 0], range(5)]).T
              ), axis=0)

plt.quiver(GRID[0],GRID[1],V[:,:,0],V[:,:,1])
for s in S:
    plt.plot(s[:,0],s[:,1])
F = fitness(S,V)