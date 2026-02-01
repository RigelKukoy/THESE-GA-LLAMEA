import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl

class FuzzyDE:
    def __init__(self, budget=10000, dim=10, pop_size=50, initial_F=0.5, initial_CR=0.9, stagnation_limit=100):
        self.budget = budget
        self.dim = dim
        self.pop_size = pop_size
        self.F = initial_F
        self.CR = initial_CR
        self.initial_F = initial_F
        self.initial_CR = initial_CR
        self.stagnation_limit = stagnation_limit
        self.population = None
        self.fitness = None
        self.best_fitness_history = []
        self.stagnation_counter = 0
        self.improvement_rate_history = []

        # Fuzzy Logic Controller Setup
        self.setup_fuzzy_controller()

    def setup_fuzzy_controller(self):
        # Define fuzzy variables
        diversity = ctrl.Antecedent(np.linspace(0, 1, 100), 'diversity')
        improvement_rate = ctrl.Antecedent(np.linspace(0, 1, 100), 'improvement_rate')
        mutation_factor = ctrl.Consequent(np.linspace(0.1, 1.0, 100), 'mutation_factor')
        crossover_rate = ctrl.Consequent(np.linspace(0.1, 1.0, 100), 'crossover_rate')

        # Define fuzzy membership functions
        diversity['low'] = fuzz.trimf(diversity.universe, [0, 0, 0.5])
        diversity['medium'] = fuzz.trimf(diversity.universe, [0, 0.5, 1])
        diversity['high'] = fuzz.trimf(diversity.universe, [0.5, 1, 1])

        improvement_rate['low'] = fuzz.trimf(improvement_rate.universe, [0, 0, 0.5])
        improvement_rate['medium'] = fuzz.trimf(improvement_rate.universe, [0, 0.5, 1])
        improvement_rate['high'] = fuzz.trimf(improvement_rate.universe, [0.5, 1, 1])

        mutation_factor['low'] = fuzz.trimf(mutation_factor.universe, [0.1, 0.1, 0.5])
        mutation_factor['medium'] = fuzz.trimf(mutation_factor.universe, [0.1, 0.5, 1])
        mutation_factor['high'] = fuzz.trimf(mutation_factor.universe, [0.5, 1, 1])

        crossover_rate['low'] = fuzz.trimf(crossover_rate.universe, [0.1, 0.1, 0.5])
        crossover_rate['medium'] = fuzz.trimf(crossover_rate.universe, [0.1, 0.5, 1])
        crossover_rate['high'] = fuzz.trimf(crossover_rate.universe, [0.5, 1, 1])


        # Define fuzzy rules
        rule1 = ctrl.Rule(diversity['low'] & improvement_rate['low'], (mutation_factor['high'], crossover_rate['low']))
        rule2 = ctrl.Rule(diversity['low'] & improvement_rate['medium'], (mutation_factor['medium'], crossover_rate['medium']))
        rule3 = ctrl.Rule(diversity['low'] & improvement_rate['high'], (mutation_factor['low'], crossover_rate['high']))

        rule4 = ctrl.Rule(diversity['medium'] & improvement_rate['low'], (mutation_factor['high'], crossover_rate['medium']))
        rule5 = ctrl.Rule(diversity['medium'] & improvement_rate['medium'], (mutation_factor['medium'], crossover_rate['medium']))
        rule6 = ctrl.Rule(diversity['medium'] & improvement_rate['high'], (mutation_factor['low'], crossover_rate['high']))

        rule7 = ctrl.Rule(diversity['high'] & improvement_rate['low'], (mutation_factor['high'], crossover_rate['low']))
        rule8 = ctrl.Rule(diversity['high'] & improvement_rate['medium'], (mutation_factor['medium'], crossover_rate['medium']))
        rule9 = ctrl.Rule(diversity['high'] & improvement_rate['high'], (mutation_factor['low'], crossover_rate['high']))

        # Control System Creation and Simulation
        self.tipping_ctrl = ctrl.ControlSystem([rule1, rule2, rule3, rule4, rule5, rule6, rule7, rule8, rule9])
        self.tipping = ctrl.ControlSystemSimulation(self.tipping_ctrl)

    def initialize_population(self, func):
        self.population = np.random.uniform(func.bounds.lb, func.bounds.ub, size=(self.pop_size, self.dim))
        self.fitness = np.array([func(x) for x in self.population])
        self.best_fitness_history.append(np.min(self.fitness))
        self.budget -= self.pop_size

    def mutate(self, x_i):
        indices = np.random.choice(self.pop_size, size=3, replace=False)
        x_r1 = self.population[indices[0]]
        x_r2 = self.population[indices[1]]
        x_r3 = self.population[indices[2]]
        
        # Cauchy mutation
        cauchy_mutation = np.random.standard_cauchy(size=self.dim)
        v_i = x_r1 + self.F * (x_r2 - x_r3) + 0.01 * cauchy_mutation  # Scale Cauchy noise for fine-tuning

        return v_i

    def crossover(self, x_i, v_i):
        u_i = np.copy(x_i)
        j_rand = np.random.randint(self.dim)
        for j in range(self.dim):
            if np.random.rand() <= self.CR or j == j_rand:
                u_i[j] = v_i[j]
        return u_i

    def repair(self, x, func):
        return np.clip(x, func.bounds.lb, func.bounds.ub)

    def check_stagnation(self):
        if len(self.best_fitness_history) > self.stagnation_limit:
            if np.abs(self.best_fitness_history[-1] - np.mean(self.best_fitness_history[-self.stagnation_limit:])) < 1e-6:
                self.stagnation_counter += 1
            else:
                self.stagnation_counter = 0

        if self.stagnation_counter >= self.stagnation_limit:
            return True
        else:
            return False

    def restart(self, func):
        """Restart the population with new random individuals."""
        self.population = np.random.uniform(func.bounds.lb, func.bounds.ub, size=(self.pop_size, self.dim))
        self.fitness = np.array([func(x) for x in self.population])
        self.F = self.initial_F
        self.CR = self.initial_CR
        self.stagnation_counter = 0
        self.best_fitness_history = [np.min(self.fitness)]
        self.improvement_rate_history = []
        self.budget -= self.pop_size

    def __call__(self, func):
        self.f_opt = np.Inf
        self.x_opt = None
        self.initialize_population(func)

        while self.budget > 0:
            # Calculate Diversity
            diversity = np.std(self.fitness) / (np.max(self.fitness) - np.min(self.fitness) + 1e-8)
            diversity = np.clip(diversity, 0, 1)

            # Calculate Improvement Rate
            if len(self.best_fitness_history) > 1:
                improvement_rate = (self.best_fitness_history[-2] - self.best_fitness_history[-1]) / (self.best_fitness_history[-2] + 1e-8)
                improvement_rate = np.clip(improvement_rate, 0, 1)
            else:
                improvement_rate = 0.0
            self.improvement_rate_history.append(improvement_rate)
            if len(self.improvement_rate_history) > self.stagnation_limit:
                self.improvement_rate_history.pop(0)


            # Fuzzy Logic Controller Evaluation
            self.tipping.input['diversity'] = diversity
            self.tipping.input['improvement_rate'] = improvement_rate
            self.tipping.compute()
            
            self.F = self.tipping.output['mutation_factor']
            self.CR = self.tipping.output['crossover_rate']
            
            for i in range(self.pop_size):
                # Mutation
                v_i = self.mutate(self.population[i])

                # Crossover
                u_i = self.crossover(self.population[i], v_i)

                # Repair
                u_i = self.repair(u_i, func)

                # Selection
                f_u_i = func(u_i)
                self.budget -= 1

                if f_u_i < self.fitness[i]:
                    self.population[i] = u_i
                    self.fitness[i] = f_u_i
                    if f_u_i < self.f_opt:
                        self.f_opt = f_u_i
                        self.x_opt = u_i

            # Stagnation Check and Restart
            if self.check_stagnation():
                self.restart(func)

            self.best_fitness_history.append(np.min(self.fitness))

            if self.budget <= 0:
                break

        return self.f_opt, self.x_opt