from metaheuristic import metaheuristic


class PSO(metaheuristic):

    def __init__(self, fitness_function, dimensions):
        self.parameters = {
            "velocity": 0.4,
            "best_personal": 2,
            "best": 2
        }
        self.algorithm_name = "PSO"
        self.velocity = self.empty_array()
        self.best_personal = self.empty_array()
        self.best_personal_fitness = self.empty_array()
        metaheuristic.__init__(self, fitness_function, dimensions)

    def operators(self):
        velocity_p = self.parameters["velocity"]
        best_personal_p = self.parameters["best_personal"]
        best_p = self.parameters["best"]
        if self.velocity.shape[0] == 0 or self.best_personal.shape[0] == 0:
            self.velocity = self.zeros_population()
            self.best_personal = self.population.copy()
            self.best_personal_fitness = self.fitness.copy()
        # atraction to the best personal solution
        pb_atraction = self.rand_population()
        pb_atraction *= (self.best_personal - self.population)
        # atraction to the best global solution
        best_atraction = self.rand_population()
        best_atraction *= (self.best_solution - self.population)
        # update velocity trayectories.
        self.velocity *= velocity_p
        self.velocity += best_personal_p * pb_atraction + best_p * best_atraction
        # update solutions
        self.population += self.velocity
        self.check_bounds()
        self.eval_population()
        # Update Best Personal
        tempIndx = self.fitness < self.best_personal_fitness
        self.best_personal_fitness[tempIndx] = self.fitness[tempIndx]
        self.best_personal[tempIndx] = self.population[tempIndx]
