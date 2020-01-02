import numpy as np
import time

class metaheuristic:
    """
    aisearch class
    The base class of the metaheuristics implementations for python
    """
    version = "0.1.0"

    def __init__(self, fitness_function, dimensions):
        self.max_iterations = 1000
        self.population_size = 30
        self.population = []
        self.fitness = []
        self.fitness_function = fitness_function
        self.dimensions = dimensions

        # Records
        self.best_solution = []
        self.best_solution_changes = 0
        self.best_solution_progresion = []
        self.best_fitness = np.inf
        self.best_fitness_hist = []
        self.number_of_calls = 0
        self.iteration = 0
        self.iteration_time_hist = []

        # extend
        self.at_each_iteration = None
        self.at_each_evaluation = None

        # visualization
        self.plot = False

    def start(self):
        "Start the optimization process"
        if not len(self.population):
            self.init_population()
            self.eval_population()
        if not len(self.fitness):
            self.eval_population()
        self.update_best()
        for i in range(self.max_iterations):
            self.iteration = i
            begging = time.clock()
            self.operators()
            self.update_best()
            end = time.clock()
            self.iteration_time_hist.append(end - begging)
            # keep population record?
            self.best_fitness_hist.append(self.best_fitness)
            if callable(self.at_each_iteration):
                self.at_each_iteration(self)
            # Visualization

    def init_population(self):
        "Set population to random solutions"
        self.population = np.random.rand(self.population_size, self.dimensions)

    def eval_population(self, solutions=None):
        "Evualuate the solutions on the fitness function"
        local = True
        if solutions is None:
            # When no solutions are given the population is evaluated
            solutions = self.population
            local = False
        fit = self.fitness_function(solutions)
        if not local:
            self.fitness = fit
        if callable(self.at_each_evaluation):
            self.at_each_evaluation(self, solutions, fit)
        return fit

    def update_best(self):
        temp = np.argmin(self.fitness)
        if self.best_fitness > self.fitness[temp]:
            self.best_fitness = self.fitness[temp]
            self.best_solution = self.population[temp]
            self.best_solution_changes += 1
            self.best_solution_progresion.append(self.best_solution)

    # General Utilities
    def check_bounds(self, solutions=None):
        # this is the saturation method
        if solutions is None:
            solutions = self.population
        solutions[solutions > 1] = 1
        solutions[solutions < 0] = 0
        return solutions

    def check_bounds_toroidal(self, solutions=None):
        if solutions is None:
            solutions = self.population
        solutions -= np.floor(solutions)
        return solutions

    def sort_population(self):
        "Sort the population fitness wise"

    def get_shuffled_population(self):
        randIndexing = np.random.permutation(self.population_size)
        return self.population[randIndexing]

    def accept_improvements(self, trial_population, trial_fitnesses):
        improved = self.fitness > trial_fitnesses
        self.population[improved] = trial_population[improved]
        self.fitness[improved] = trial_fitnesses[improved]
