from metaheuristic import metaheuristic
import numpy as np


class HS(metaheuristic):

    def __init__(self, fitness_function, dimensions):
        self.parameters = {
            "new_melody_rate": 0.7,
            "pitch_adjustRate": 0.5,
            "step_adjust": 0.1
        }
        self.algorithm_name = "HS"
        self.temp_arange = []
        metaheuristic.__init__(self, fitness_function, dimensions)

    def operators(self):
        new_melody_rate = self.parameters["new_melody_rate"]
        pitch_adjustRate = self.parameters["pitch_adjustRate"]
        step_adjust = self.parameters["step_adjust"]

        new_melody_indx = self.rand_population() < new_melody_rate
        pitch_adjust_indx = self.rand_population() < pitch_adjustRate
        pitch_adjust_indx = pitch_adjust_indx & new_melody_indx
        pitch_adjustments = self.rand_population() * step_adjust * pitch_adjust_indx

        trial_population = self.get_shuffled_population() * new_melody_indx
        trial_population += self.rand_population() * np.logical_not(new_melody_indx)
        trial_population += pitch_adjustments
        trial_population = self.check_bounds_toroidal(trial_population)

        trial_fitnesses = self.eval_population(trial_population)
        self.keep_the_best(trial_population, trial_fitnesses)
