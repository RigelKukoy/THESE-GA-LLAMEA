import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl

class FuzzyAdaptiveDE:
    def __init__(self, budget=10000, dim=10, initial_pop_size=50, min_pop_size=10, max_pop_size=100, stagnation_threshold=100):
        self.budget = budget
        self.dim = dim
        self.initial_pop_size = initial_pop_size
        self.min_pop_size = min_pop_size
        self.max_pop_size = max_pop_size
        self.pop_size = initial_pop_size
        self.stagnation_threshold = stagnation_threshold
        self.stagnation_counter = 0
        self.best_fitness_history = []
        self.population = None
        self.fitness = None
        self.best_index = None
        self.f_opt = np.inf
        self.x_opt = None
        self.F = 0.5  # Initial mutation factor
        self.CR = 0.7 # Initial Crossover rate

        # Fuzzy Logic Controller for F and CR adaptation
        self.antecedent_stagnation = ctrl.Antecedent(np.linspace(0, self.stagnation_threshold, 1000), 'stagnation')
        self.antecedent_fitness_variation = ctrl.Antecedent(np.linspace(0, 1, 1000), 'fitness_variation') # Normalized fitness variation
        self.consequent_F = ctrl.Consequent(np.linspace(0.1, 0.9, 100), 'F')
        self.consequent_CR = ctrl.Consequent(np.linspace(0.1, 0.9, 100), 'CR')
        
        # Define membership functions (example: triangular)
        self.antecedent_stagnation['low'] = fuzz.trimf(self.antecedent_stagnation.universe, [0, 0, self.stagnation_threshold/2])
        self.antecedent_stagnation['medium'] = fuzz.trimf(self.antecedent_stagnation.universe, [0, self.stagnation_threshold/2, self.stagnation_threshold])
        self.antecedent_stagnation['high'] = fuzz.trimf(self.antecedent_stagnation.universe, [self.stagnation_threshold/2, self.stagnation_threshold, self.stagnation_threshold])

        self.antecedent_fitness_variation['low'] = fuzz.trimf(self.antecedent_fitness_variation.universe, [0, 0, 0.5])
        self.antecedent_fitness_variation['medium'] = fuzz.trimf(self.antecedent_fitness_variation.universe, [0, 0.5, 1])
        self.antecedent_fitness_variation['high'] = fuzz.trimf(self.antecedent_fitness_variation.universe, [0.5, 1, 1])
        
        self.consequent_F['low'] = fuzz.trimf(self.consequent_F.universe, [0.1, 0.1, 0.5])
        self.consequent_F['medium'] = fuzz.trimf(self.consequent_F.universe, [0.1, 0.5, 0.9])
        self.consequent_F['high'] = fuzz.trimf(self.consequent_F.universe, [0.5, 0.9, 0.9])

        self.consequent_CR['low'] = fuzz.trimf(self.consequent_CR.universe, [0.1, 0.1, 0.5])
        self.consequent_CR['medium'] = fuzz.trimf(self.consequent_CR.universe, [0.1, 0.5, 0.9])
        self.consequent_CR['high'] = fuzz.trimf(self.consequent_CR.universe, [0.5, 0.9, 0.9])

        # Define rules
        rule1 = ctrl.Rule(self.antecedent_stagnation['low'] & self.antecedent_fitness_variation['high'], [self.consequent_F['low'], self.consequent_CR['high']])
        rule2 = ctrl.Rule(self.antecedent_stagnation['medium'] & self.antecedent_fitness_variation['medium'], [self.consequent_F['medium'], self.consequent_CR['medium']])
        rule3 = ctrl.Rule(self.antecedent_stagnation['high'] & self.antecedent_fitness_variation['low'], [self.consequent_F['high'], self.consequent_CR['low']])

        self.control_system = ctrl.ControlSystem([rule1, rule2, rule3])
        self.parameter_control = ctrl.ControlSystemSimulation(self.control_system)

    def initialize_population(self, func):
        self.population = np.random.uniform(func.bounds.lb, func.bounds.ub, size=(self.pop_size, self.dim))
        self.fitness = np.array([func(x) for x in self.population])
        self.budget -= self.pop_size
        self.best_index = np.argmin(self.fitness)
        self.f_opt = self.fitness[self.best_index]
        self.x_opt = self.population[self.best_index].copy()
        self.lb = func.bounds.lb
        self.ub = func.bounds.ub
        self.best_fitness_history.append(self.f_opt)

    def adjust_population_size(self):
        if self.stagnation_counter > self.stagnation_threshold:
            self.pop_size = max(self.min_pop_size, self.pop_size // 2)  # Reduce population
            self.stagnation_counter = 0
            print(f"Population size reduced to {self.pop_size}")
        elif self.stagnation_counter > self.stagnation_threshold/2 and self.pop_size < self.max_pop_size:
            self.pop_size = min(self.max_pop_size, self.pop_size * 2) # Increase population
            print(f"Population size increased to {self.pop_size}")
            
    def __call__(self, func):
        self.initialize_population(func)

        while self.budget > 0:
            # Parameter Adaptation using Fuzzy Logic
            if len(self.best_fitness_history) > 1:
                fitness_variation = abs(self.best_fitness_history[-1] - self.best_fitness_history[-2]) / max(abs(self.best_fitness_history[-1]), 1e-9)  # Normalize the fitness variation
            else:
                fitness_variation = 0.0
            
            self.parameter_control.input['stagnation'] = self.stagnation_counter
            self.parameter_control.input['fitness_variation'] = fitness_variation
            self.parameter_control.compute()

            self.F = self.parameter_control.output['F']
            self.CR = self.parameter_control.output['CR']
            

            for i in range(self.pop_size):
                # Mutation
                indices = np.random.choice(self.pop_size, 3, replace=False)
                x_r1, x_r2, x_r3 = self.population[indices]
                v = self.population[i] + self.F * (x_r2 - x_r3)
                v = np.clip(v, self.lb, self.ub)

                # Crossover
                j_rand = np.random.randint(self.dim)
                u = self.population[i].copy()
                for j in range(self.dim):
                    if np.random.rand() < self.CR or j == j_rand:
                        u[j] = v[j]

                # Evaluation
                f_u = func(u)
                self.budget -= 1

                # Selection
                if f_u < self.fitness[i]:
                    self.fitness[i] = f_u
                    self.population[i] = u
                    if f_u < self.f_opt:
                        self.f_opt = f_u
                        self.x_opt = u.copy()
                        self.best_index = np.argmin(self.fitness)
                        self.stagnation_counter = 0  # Reset stagnation counter
                    else:
                        self.stagnation_counter += 1
                else:
                    self.stagnation_counter += 1
            
            self.best_fitness_history.append(self.f_opt)
            self.adjust_population_size() # Adjust population size based on stagnation

        return self.f_opt, self.x_opt