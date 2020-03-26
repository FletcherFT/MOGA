import numpy as np
import random
from mogapy.solver import Solver


def ndsa(fitnesses):
    """Take in a list of fitnesses, rank according to level of non-dominatedness.
    Input:
    fitnesses: A MxN numpy array, each row is a solution, each column is a fitness variable.
    Outputs:
    An ordered list, each element contains a list of row indices of fitnesses that correspond
    to a non-dominated ranking: 0 - 1st rank, 1 - 2nd rank and so on.
    1st rank is completely non-Dominated (the Pareto front for the given solutions)."""
    M, N = fitnesses.shape
    # Dominated Counter
    d = np.zeros((M, 1))
    # Dominating Tracker, a M length list showing what solutions are dominated by a solution.
    s = [[] for _ in range(M)]
    # The current front ranking (initialised at 0)
    f = 0
    # Rankings list (theoretically there can be up to M rankings)
    F = [[] for _ in range(M)]
    # For every solution, check if it dominates the rest
    for i in range(M):
        # select solution p
        p = fitnesses[i, :]
        for j in range(i + 1, M):
            # select solution q
            q = fitnesses[j, :]
            # If p dominates q
            if np.all(p <= q) and np.any(p < q):
                # Increase the domination counter of q
                d[j] = d[j] + 1
                # Add index of q to s[i]
                s[i].append(j)
            # If q dominates p
            elif np.all(q <= p) and np.any(q < p):
                # Increase the domination counter of p
                d[i] = d[i] + 1
                # Add index of p to s[j]
                s[j].append(i)
        # If solution p is non-dominated, then assign first non-dominated rank (0 indexed)
        if d[i] == 0:
            F[f].append(i)
    # Loop through solutions to find the non-dominated points
    while len(F[f]) > 0:
        # For each solution in rank f
        for i in F[f]:
            # For each solution dominated by i
            for j in s[i]:
                d[j] = d[j] - 1
                if d[j] == 0:
                    F[f+1].append(j)
        # Increment the rank
        f = f + 1
    # Remove empty rankings from F and return
    return [i for i in F if len(i) > 0]


def crowding_distance(solutions, rankings, fitnesses):
    D = np.zeros((solutions.shape[0],))
    # for each ranking
    for r in rankings:
        # Get the fitness for the current rank
        fitness = fitnesses[r, :]
        # Sort each fitness
        c = np.sort(fitness, axis=0)
        # Get the mapping
        s = np.argsort(fitness, axis=0)
        # Init the Normalized crowding distance
        d = np.inf * np.ones(fitness.shape)
        for i in range(1, d.shape[0] - 1):
            d[s[i, :], range(d.shape[1])] = np.abs(c[i + 1, :] - c[i - 1, :]) / (c.max(axis=0) - c.min(axis=0))
        D[r] = d.sum(axis=-1)
    return D


class NDSA1(Solver):
    """This class implements Non-Dominated Sorting Algorithm for Genetic Algorithm."""
    def __init__(self, popsize, **kwargs):
        super().__init__(popsize, **kwargs)

    def _selection(self, solutions, *args):
        """Values are drawn based on rank"""
        rankings = args[0]
        # Unroll the indices into an ordered list according to rank
        ordered = np.array([j for i in rankings for j in i])
        # Get the mapping between the ranked order and the solutions
        mapping = np.argsort(ordered)
        # Also unroll the rankings
        ranks = np.array([k for i, j in enumerate(rankings) for k in [i] * len(j)])
        # Invert the rankings to give the highest weighting to the best rank
        p = ranks.max()-ranks
        # Sort according to the order
        p = p[mapping]
        # Normalise the weightings for each rank to get p distribution for each value.
        # Special case: all ranks are 0
        if p.sum() == 0:
            p = p+1
        p = p/p.sum()
        # draw parent indices from the distribution
        pidx = np.random.choice(range(solutions.shape[0]), size=(solutions.shape[0], 1), p=p)
        return np.squeeze(solutions[pidx, :])

    def _xover(self, solutions):
        """Generate children by 2 point crossover"""
        children = solutions.copy()
        for i in range(children.shape[0]):
            # Pick two different parents
            pidx = np.random.choice(range(solutions.shape[0]), size=(2, 1), replace=False)
            # Set template chromosone to be first choice
            children[i, :] = solutions[pidx[0], :]
            # Pick two indices
            cidx = np.sort(np.random.choice(range(solutions.shape[1]), size=(2, 1), replace=False), axis=0).squeeze()
            # Replace child indices with parent 2's indices
            children[i, cidx[0]:cidx[1]] = solutions[pidx[1], cidx[0]:cidx[1]]
        return children

    def _mutate(self, solutions):
        """Mutate children by random inversion."""
        children = solutions.copy()
        for i in range(children.shape[0]):
            # Pick two indices between start+1 and finish-1
            cidx = np.sort(np.random.choice(range(1, solutions.shape[1]-1), size=(2, 1), replace=False), axis=0).squeeze()
            # Invert child indices
            children[i, cidx[0]:cidx[1]] = children[i, cidx[1]:cidx[0]:-1]
        return children

    def _survival(self, solutions, *args):
        N = int(solutions.shape[0] / 2)
        rankings = args[0]
        fitnesses = args[1]
        # Add the indices of the top k rankings until the length of the indices >= N.
        sur_idx =[]
        k = 0
        while len(sur_idx) < solutions.shape[0]/2:
            sur_idx += rankings[k]
            k += 1
        sur_idx = np.array(sur_idx)
        # Trim the solutions to create the survivors
        survivors = solutions[sur_idx, :]
        # Trim the fitnesses
        fitnesses = fitnesses[sur_idx, :]
        # Trim the rankings
        rankings = rankings[:k]
        # Refactor the ranking indices
        # Unroll the indices into an ordered list according to rank
        ordered = np.array([j for i in rankings for j in i])
        # Get the mapping between the ranked order and the solutions
        mapping = np.argsort(ordered)
        # Also unroll the rankings
        ranks = np.array([k for i, j in enumerate(rankings) for k in [i] * len(j)])
        # Repackage the rankings with the indices mapped to the survivors
        rankings = [mapping[ranks == i].tolist() for i in range(k)]
        ranks = np.array([k for i, j in enumerate(rankings) for k in [i] * len(j)])
        # If there are more than N surviving solutions, then trim according to crowding distance.
        if len(sur_idx) > N:
            D = crowding_distance(survivors, rankings, fitnesses)
            # get the indices of the solutions sorted by crowding distance.
            d_idx = np.flipud(np.argsort(D))
            survivors = survivors[d_idx[:N], :]
            fitnesses = fitnesses[d_idx[:N], :]
        return survivors, fitnesses

    def update(self, solutions, **kwargs):
        # Step 1: calculate fitness step has already been done.
        fitnesses = self._fitness(solutions, **kwargs)
        # Step 2: identify the Pareto rankings
        rankings = ndsa(fitnesses)
        # Step 3: Selection based on rank
        parents = self._selection(solutions, rankings)
        # Step 4: Generate children from parents
        children = self._mutate(self._xover(parents))
        # Step 5: Get the fitness of the children
        c_fitness = self._fitness(children, **kwargs)
        # Step 6: Get the rankings of the complete set of children and original population
        rankings = ndsa(np.vstack((fitnesses, c_fitness)))
        # Step 7: Survival according to crowding distance and ranking
        return self._survival(np.vstack((solutions, children)), rankings, np.vstack((fitnesses, c_fitness)))
