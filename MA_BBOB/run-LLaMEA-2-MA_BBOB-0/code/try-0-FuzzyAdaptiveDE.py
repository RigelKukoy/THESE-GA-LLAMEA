import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl
from scipy.optimize import minimize

class FuzzyAdaptiveDE:
    def __init__(self, budget=10000, dim=10, pop_size=50, local_search_iterations=5):
        self.budget = budget
        self.dim = dim
        self.pop_size = pop_size
        self.local_search_iterations = local_search_iterations
        self.population = None
        self.fitness = None
        self.best_fitness = np.inf
        self.best_solution = None

        # Fuzzy Logic Controller Setup
        self.success_rate = ctrl.Antecedent(np.arange(0, 1.01, 0.01), 'success_rate')
        self.diversity = ctrl.Antecedent(np.arange(0, 1.01, 0.01), 'diversity')
        self.F = ctrl.Consequent(np.arange(0.1, 1.01, 0.01), 'F')
        self.CR = ctrl.Consequent(np.arange(0.1, 1.01, 0.01), 'CR')

        # Membership functions (example - can be tuned)
        self.success_rate['low'] = fuzz.trimf(self.success_rate.universe, [0, 0, 0.5])
        self.success_rate['medium'] = fuzz.trimf(self.success_rate.universe, [0.25, 0.5, 0.75])
        self.success_rate['high'] = fuzz.trimf(self.success_rate.universe, [0.5, 1, 1])

        self.diversity['low'] = fuzz.trimf(self.diversity.universe, [0, 0, 0.5])
        self.diversity['medium'] = fuzz.trimf(self.diversity.universe, [0.25, 0.5, 0.75])
        self.diversity['high'] = fuzz.trimf(self.diversity.universe, [0.5, 1, 1])

        self.F['small'] = fuzz.trimf(self.F.universe, [0.1, 0.1, 0.6])
        self.F['medium'] = fuzz.trimf(self.F.universe, [0.3, 0.6, 0.9])
        self.F['large'] = fuzz.trimf(self.F.universe, [0.6, 1, 1])

        self.CR['small'] = fuzz.trimf(self.CR.universe, [0.1, 0.1, 0.6])
        self.CR['medium'] = fuzz.trimf(self.CR.universe, [0.3, 0.6, 0.9])
        self.CR['large'] = fuzz.trimf(self.CR.universe, [0.6, 1, 1])

        # Rules (example - can be tuned)
        rule1 = ctrl.Rule(self.success_rate['low'] & self.diversity['low'], (self.F['large'], self.CR['small']))
        rule2 = ctrl.Rule(self.success_rate['low'] & self.diversity['medium'], (self.F['medium'], self.CR['small']))
        rule3 = ctrl.Rule(self.success_rate['low'] & self.diversity['high'], (self.F['small'], self.CR['medium']))
        rule4 = ctrl.Rule(self.success_rate['medium'] & self.diversity['low'], (self.F['large'], self.CR['medium']))
        rule5 = ctrl.Rule(self.success_rate['medium'] & self.diversity['medium'], (self.F['medium'], self.CR['medium']))
        rule6 = ctrl.Rule(self.success_rate['medium'] & self.diversity['high'], (self.F['small'], self.CR['large']))
        rule7 = ctrl.Rule(self.success_rate['high'] & self.diversity['low'], (self.F['medium'], self.CR['large']))
        rule8 = ctrl.Rule(self.success_rate['high'] & self.diversity['medium'], (self.F['small'], self.CR['large']))
        rule9 = ctrl.Rule(self.success_rate['high'] & self.diversity['high'], (self.F['small'], self.CR['large']))

        self.parameter_control = ctrl.ControlSystem([rule1, rule2, rule3, rule4, rule5, rule6, rule7, rule8, rule9])
        self.parameter_simulation = ctrl.ControlSystemSimulation(self.parameter_control)

    def initialize_population(self, func):
        self.population = np.random.uniform(func.bounds.lb, func.bounds.ub, size=(self.pop_size, self.dim))
        self.fitness = np.array([func(x) for x in self.population])
        self.budget -= self.pop_size

        best_index = np.argmin(self.fitness)
        self.best_fitness = self.fitness[best_index]
        self.best_solution = self.population[best_index].copy()

    def calculate_diversity(self):
        # Simple diversity measure: average distance from centroid
        centroid = np.mean(self.population, axis=0)
        distances = np.linalg.norm(self.population - centroid, axis=1)
        diversity = np.mean(distances)
        # Normalize diversity to [0, 1]
        diversity = diversity / (func.bounds.ub[0] - func.bounds.lb[0])  # Assuming bounds are the same for all dimensions
        return np.clip(diversity, 0, 1)


    def evolve(self, func):
        F = 0.5 * np.ones(self.pop_size)  # Initialize F and CR
        CR = 0.9 * np.ones(self.pop_size)

        success_count = 0
        for i in range(self.pop_size):
            # Mutation
            idxs = np.random.choice(self.pop_size, 3, replace=False)
            x_r1, x_r2, x_r3 = self.population[idxs]
            mutant = self.population[i] + F[i] * (x_r1 - x_r2)
            mutant = np.clip(mutant, func.bounds.lb, func.bounds.ub)

            # Crossover
            crossover_mask = np.random.rand(self.dim) < CR[i]
            trial = np.where(crossover_mask, mutant, self.population[i])

            # Evaluation
            trial_fitness = func(trial)
            self.budget -= 1

            if trial_fitness < self.best_fitness:
                self.best_fitness = trial_fitness
                self.best_solution = trial.copy()
                success_count += 1
            
            # Selection
            if trial_fitness < self.fitness[i]:
                self.population[i] = trial
                self.fitness[i] = trial_fitness
            
        # Fuzzy Parameter Adaptation
        success_rate = success_count / self.pop_size if self.pop_size > 0 else 0
        diversity = self.calculate_diversity()

        self.parameter_simulation.input['success_rate'] = success_rate
        self.parameter_simulation.input['diversity'] = diversity
        self.parameter_simulation.compute()

        F = self.parameter_simulation.output['F']
        CR = self.parameter_simulation.output['CR']

        return F, CR
    
    def local_search(self, func):
        bounds = [(func.bounds.lb[i], func.bounds.ub[i]) for i in range(self.dim)]
        result = minimize(func, self.best_solution, method='Nelder-Mead', bounds=bounds, options={'maxiter': self.local_search_iterations})
        if result.fun < self.best_fitness:
            self.best_fitness = result.fun
            self.best_solution = result.x

        self.budget -= result.nit # Account for function evaluations by Nelder-Mead


    def __call__(self, func):
        self.initialize_population(func)

        while self.budget > self.local_search_iterations:
            F, CR = self.evolve(func)
        
        self.local_search(func)

        return self.best_fitness, self.best_solution