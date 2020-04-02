import numpy as np


class Solver:
    """Base class for running MOGA. Default behaviour is to evolved solutions through
    adding white noise to solutions according to user-defined fitness, survival and selection functions.
    Fitness, survival and selection functions are either expected to be overrided by inheritance, or have a
    function handle passed into the fitness, survival, and selection arguments."""
    def __init__(self, pop_size, elitism=0.9, mutation_rate=0.02, fitness=None, survival=None, selection=None):
        assert 0 < elitism < 1, "Elitism must be bounded (0, 1). At least one solution must be survive!"
        assert 0 <= mutation_rate <= 1, "Mutation rate must be bounded [0, 1]."
        if fitness is not None:
            self._fitness = fitness
        if survival is not None:
            self._survival = survival
        if selection is not None:
            self._selection = selection
        # Number of solutions
        self._n = pop_size
        # Number of solutions to keep
        self._e = int(pop_size * elitism)
        # The mutation rate default from https://arxiv.org/pdf/1712.06567.pdf
        self._m = mutation_rate
        # Make sure the elitism is between 1 and N-1
        if self._e >= self._n:
            self._e = self._n - 1
        if self._e < 1:
            self._e = 1

    def _mutate(self, solutions):
        """Default mutation is to add small zero mean Gaussian noise to the solution.
        Mutation rate is used to adjust the variance of the Gaussian.
        Inputs
        solutions: an MxN numpy array containing M solutions with N solution variables.
        Outputs
        solutions with white noise added."""
        return solutions + self._m * np.random.randn()

    def _xover(self, solutions, *args):
        """Default behaviour is just to return the solutions."""
        return solutions

    def _selection(self, solutions):
        """Must be replaced by a function that outputs a subset of fitnesses and solutions as parents."""
        raise Exception("The selection method needs to be either overrided or a valid function handle needs to be "
                        "passed as selection keyword during construction.")

    def _survival(self, solutions):
        """Must be replaced by a function that outputs a subset of solutions and fitnesses as survivors."""
        raise Exception("The survival method needs to be either overrided or a valid function handle needs to be "
                        "passed as survival keyword during construction.")

    def _fitness(self, solutions):
        """Must be replaced by a function that outputs an MxN array of fitnesses."""
        raise Exception("The fitness method needs to be either overrided or a valid function handle needs to be "
                        "passed as fitness keyword during construction.")

    def update(self, solutions):
        """Default behaviour is to return the next generation of solutions. Should return"""
        # Get the fitness of the proposed solutions
        fitnesses = self._fitness(solutions)
        # Select the parents
        parents = self._selection(fitnesses, solutions)
        # Obtain children from the parents
        children = self._mutate(self._xover(parents))
        # Get the fitness of the children
        c_fitness = self._fitness(children)
        # Apply survival and return the updated solutions.
        return self._survival(np.hstack((parents, children)), np.hstack((fitnesses, c_fitness)))

