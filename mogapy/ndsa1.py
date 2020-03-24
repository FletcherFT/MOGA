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


class NDSA1(Solver):
    """This class implements Non-Dominated Sorting Algorithm for Genetic Algorithm."""
    def __init__(self, popsize, **kwargs):
        super().__init__(popsize, **kwargs)

    def _mutate(self, solutions):
        """Override the mutate function."""
        return [weight + self._m * np.random.randn() for weight in solutions]

    def _xover(self, solutions):
        """Add a cross-over function."""
        return solutions

    def update(self, solutions):
        # Step 1: calculate fitness step has already been done.
        weights = [i[1] for i in solutions]
        fitnesses = np.array([i[2:] for i in solutions])
        # Step 2: identify the Pareto rankings
        rankings = self._ndsa(fitnesses)
        # Unroll the rankings
        rankings = [j for i in rankings for j in i]
        # Step 3: Survival
        # indices for solutions that survive to be mutated
        parents = rankings[:self._e]
        children = rankings[self._e:]
        # Step 4: Mutation
        for i in range(len(children)):
            weights[children[i]] = self._mutate(weights[random.choice(parents)])
        # Return the new solutions
        return weights
