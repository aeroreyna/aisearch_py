from metaheuristic import metaheuristic
import numpy as np


class WOA(metaheuristic):

    def __init__(self, fitness_function, dimensions):
        self.parameters = {
        }
        self.algorithm_name = "WOA"
        self.random_pop_dim = self.empty_array()
        metaheuristic.__init__(self, fitness_function, dimensions)

    def operators(self):
        # Random mixture of solutions
        self.random_pop_dim = np.random.permutation(self.population_size)
        for i in range(self.dimensions - 1):
            self.random_pop_dim = np.vstack((self.random_pop_dim, np.random.permutation(self.population_size)))

        t_progress = self.iteration / self.max_iterations
        # Circular Attraction
        l_var = -(2 + t_progress) * self.rand(self.population_size, 1) + 1
        l_var = np.exp(l_var) * np.cos(l_var * 2 * np.pi)
        p = self.rand(self.population_size) < 0.5
        distance2Leader = abs(self.best_solution - self.population)
        tial_population = distance2Leader * l_var + self.best_solution
        # Linear Attraction
        A = 2 * (1 - t_progress) * self.rand(self.population_size)
        # A2 = repmat(A', 1, self.noDimensions) %broadcast should do it
        A2 = A.reshape(self.population_size, 1)
        C = 2 * self.rand(self.population_size, 1)

        temp_helper = np.array([range(self.dimensions)])
        X_rand = self.population[np.transpose(self.random_pop_dim), temp_helper]
        D_X_rand = abs(C * X_rand - self.population)
        A_population = X_rand - A2 * D_X_rand

        D_Leader = abs(C * self.best_solution - self.population)
        nonA_population = self.best_solution - A2 * D_Leader

        # Select Between Linear Attractions
        A_population[abs(A) < 1] = nonA_population[abs(A) < 1]

        # Select Final Trian Population
        tial_population[p] = A_population[p]

        self.population = self.check_bounds_toroidal(tial_population)
        self.eval_population()
        self.update_best()
