import numpy as np

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
                    F[f + 1].append(j)
        # Increment the rank
        f = f + 1
    # Remove empty rankings from F and return
    return [i for i in F if len(i) > 0]


def crowding_distance(rankings, fitnesses):
    D = np.zeros((fitnesses.shape[0],))
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
        # nans occur when c.max - c.min == 0, set to zero.
        d = np.nan_to_num(d, nan=0, posinf=np.inf, neginf=-np.inf)
        D[r] = d.sum(axis=-1)
    return D


def moving_average(a, n=3):
    ret = np.cumsum(a, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1:] / n


def k_tournament_generator(n, k):
    """Create 2*n k-tournaments with sample size n."""
    a = np.vstack([np.random.choice(np.arange(n), size=2*k, replace=False) for _ in range(n)])
    b = a[:, k:]
    a = a[:, :k]
    T = np.zeros((2*n, k), dtype=np.int)
    T[0::2,:] = a
    T[1::2,:] = b
    return T


class NDSA1(Solver):
    """This class implements Non-Dominated Sorting Algorithm for Genetic Algorithm."""

    def __init__(self, popsize, **kwargs):
        super().__init__(popsize, **kwargs)

    def _selection(self, solutions, *args):
        """Two parent indices are selected for each child based on rank and crowding distance (k-tournament selection)"""
        # Non-dominated sorting rank
        rankings = args[0]
        # Crowding distance
        D = args[1]
        # Unroll the indices into an ordered list according to rank
        ordered = np.array([j for i in rankings for j in i])
        # Unroll the ranks
        ranks = np.array([i for i,j in enumerate(rankings) for _ in j])
        # Get the mapping between the ranked order and the solutions
        mapping = np.argsort(ordered)
        # Mapped ranks
        mrank = ranks[mapping]
        # Generate k-tournament
        k = np.clip(int(self._n/10), 3, self._n)
        T = k_tournament_generator(self._n, k)
        # Get the performance of each competitor
        r = mrank[T.flatten()].reshape(T.shape)
        d = D[T.flatten()].reshape(T.shape)
        # Perform the tournament
        t_idx = -np.ones((2*self._n, 1), dtype=np.int)
        t_idx[:, 0] = r.argmin(axis=1)
        # Check if all competitors are of same rank
        tst = np.abs(np.diff(r, axis=1)).sum(axis=1) == 0
        # In this case perform tournament on crowding distance
        t_idx[tst, 0] = d[tst, :].argmax(axis=1)
        # winners are T indexed by t_idx
        p_idx = T[range(2*self._n), t_idx.flatten()]
        # parents are paired off
        return p_idx[0::2], p_idx[1::2]

    def _xover(self, solutions, *args):
        """Generate children by 2 point crossover"""
        p1 = args[0]
        p2 = args[1]
        children = solutions.copy()
        p = np.random.randint(0, 20, self._n)
        for i in range(self._n):
            # Select the parents
            pidx = np.array([[p1[i]], [p2[i]]])
            # Exploitation and Exploration by recombination
            if p[i] == 0:
                # Set template chromosome to be first choice
                children[i, :] = solutions[pidx[0], :]
                # Pick two indices
                cidx = np.sort(np.random.choice(range(solutions.shape[1]), size=(2, 1), replace=False),
                               axis=0).squeeze()
                # Replace child indices with parent 2's indices
                children[i, cidx[0]:cidx[1]] = solutions[pidx[1], cidx[0]:cidx[1]]
            # Exploration by Seeking Minimum
            elif p[i] == 1:
                # Find the minimum and assign to the child
                children[i, :, :] = np.min(solutions[pidx, :, :], axis=0).squeeze()
            elif p[i] == 2:
                # Find the maximum and assign to the child
                children[i, :, :] = np.max(solutions[pidx, :, :], axis=0).squeeze()
            # Exploitation and Exploration by mean averaging
            elif p[i] == 3:
                # Get the mean of the parents chromosome
                children[i, 1:solutions.shape[1] - 1, :] = np.mean(solutions[pidx, 1:solutions.shape[1] - 1, :], axis=0,
                                                                   dtype=np.int).squeeze()
        return children

    def _mutate(self, solutions, bounds=None, lineq=None, **kwargs):
        """Mutate children by random inversion (exploit)
        or by random signed increment vector (exploration)."""
        children = solutions.copy()
        N, L, Dim = children.shape
        p = np.random.randint(0, 20, N)
        for i in range(N):
            if p[i] == 0:
                # Exploitation Mechanism
                # Pick two indices between start+1 and finish-1
                cidx = np.sort(np.random.choice(range(1, L - 1), size=(2, 1), replace=False), axis=0).squeeze()
                # Invert child indices
                children[i, cidx[0]:cidx[1], 0] = children[i, cidx[1]:cidx[0]:-1, 0]
            elif p[i] == 1:
                # Exploration Mechanism
                # mutate the child by by adding a vector of integers in range [-1, 1] to a segment
                # Get the segment
                idx = np.sort(np.random.randint(1, L, (2,)))
                # Pick a magnitude
                mag = np.random.randint(1, 2, 1)
                children[i, idx[0]:idx[1], 0] = children[i, idx[0]:idx[1], 0] + np.random.randint(-mag, mag + 1,
                                                                                                  (np.diff(idx)))
            elif p[i] == 2:
                # Standard Exchange Mutation
                # Get two elements
                idx = np.random.randint(1, L-1, (2,))
                # Exchange them
                children[i, idx, 0] = children[i, np.flipud(idx), 0]
            elif p[i] == 3:
                # Exploitation Mechanism
                # Apply moving average filter to child
                children[i, 1:L - 1, 0] = moving_average(children[i, :, 0], 3)
            elif p[i] == -1:
                # Scramble Mechanism
                # Just replace child with new random vector
                children[i, 1:L - 1, 0] = np.random.randint(-L, L+1, (L-2,))
            # apply bounds if given
            if bounds is not None:
                children[i, 1:L, :] = np.clip(children[i, 1:L, :], bounds[:, 0], bounds[:, 1])
        return children

    def _survival(self, solutions, *args):
        rankings = args[0]
        fitnesses = args[1]
        # Add the indices of the top k rankings until the length of the indices >= N.
        sur_idx = []
        k = 0
        while len(sur_idx) < self._n:
            sur_idx += rankings[k]
            k += 1
        sur_idx = np.array(sur_idx)
        # Trim the rankings
        rankings = rankings[:k]
        # If there are more than N surviving solutions, then trim according to crowding distance.
        if len(sur_idx) > self._n:
            D = crowding_distance(rankings, fitnesses)
            # get the indices of the solutions sorted by crowding distance.
            sur_idx = np.flipud(np.argsort(D))
            sur_idx = sur_idx[:self._n]
        return sur_idx

    def update(self, solutions, **kwargs):
        # Step 1: calculate fitness step has already been done.
        fitnesses = self._fitness(solutions, **kwargs)
        # Step 2: identify the Pareto rankings
        rankings = ndsa(fitnesses)
        # Identify the crowding distance
        D = crowding_distance(rankings, fitnesses)
        # Step 3: Selection based on rank and crowding distance
        p1, p2 = self._selection(solutions, rankings, D)
        # Step 4: Generate children from parents
        children = self._mutate(self._xover(solutions, p1, p2), **kwargs)
        # Step 5: Get the fitness of the children
        c_fitness = self._fitness(children, **kwargs)
        # Step 6: Get the rankings of the complete set of children and original population
        # Get constraints first
        combined_fitness = np.vstack((fitnesses, c_fitness))
        rankings = ndsa(combined_fitness)
        # Step 7: Survival according to crowding distance and ranking
        # Previous solutions and children combined
        combined_solutions = np.vstack((solutions, children))
        # Get the indices of the survivors
        idx = self._survival(combined_solutions, rankings, combined_fitness)
        return combined_solutions[idx, :], combined_fitness[idx, :]
