from metaheuristic import metaheuristic
import numpy as np


class DE(metaheuristic):

    def __init__(self, fitness_function, dimensions):
        self.parameters = {
            "crossover_rate": 0.7,
            "differential_weight": 0.5
        }
        self.algorithm_name = "DE"
        self.temp_arange = []
        metaheuristic.__init__(self, fitness_function, dimensions)

    def operators(self):
        crossover_rate = self.parameters["crossover_rate"]
        differential_weight = self.parameters["differential_weight"]
        vector1 = self.get_shuffled_population()
        vector2 = self.get_shuffled_population()
        vector3 = self.get_shuffled_population()
        vector4 = self.get_shuffled_population()
        vector5 = self.get_shuffled_population()
        trial_population = self.population.copy()
        doners = vector1 + differential_weight * (vector2 - vector3) + differential_weight * (vector4 - vector5)
        crossoverProbs = self.rand_population() < crossover_rate
        trial_population[crossoverProbs] = doners[crossoverProbs]
        trial_population = self.check_bounds_toroidal(trial_population)
        trial_fitnesses = self.eval_population(trial_population)
        self.accept_improvements(trial_population, trial_fitnesses)

    def operators_old(self):
        crossover_rate = self.parameters["crossover_rate"]
        differential_weight = self.parameters["differential_weight"]

        def selectNDifSolutions(N):
            if len(self.temp_arange) == 0:
                self.temp_arange = np.arange(0, self.population_size)
            np.random.shuffle(self.temp_arange)
            return self.temp_arange[0:N]

        for s in self.population:
            selected_parents = selectNDifSolutions(5)
            selected_dim = np.random.randint(0, self.dimensions)
            solution_base = self.population[selected_parents[0]].copy()
            for d in range(self.dimensions):
                if d == selected_dim or np.random.rand() < crossover_rate:
                    solution_base[d] += differential_weight * (self.population[selected_parents[1], d] - self.population[selected_parents[2], d])
                    # solution_base[d] += differential_weight * (self.population[selected_parents[3], d] - self.population[selected_parents[4], d])
            solution_base = self.check_bounds_toroidal(solution_base)
            newFit = self.eval_population(solution_base)
            if newFit < self.fitness[selected_parents[0]]:
                # better solution
                self.population[selected_parents[0]] = solution_base
                self.fitness[selected_parents[0]] = newFit
