import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl

class FuzzyAdaptiveDE:
    def __init__(self, budget=10000, dim=10, pop_size=20, initial_mutation_factor=0.5, initial_crossover_rate=0.7):
        self.budget = budget
        self.dim = dim
        self.pop_size = pop_size
        self.mutation_factor = initial_mutation_factor
        self.crossover_rate = initial_crossover_rate
        self.population = np.random.uniform(-5, 5, size=(pop_size, dim))
        self.fitness = np.zeros(pop_size)
        self.best_positions = self.population.copy()
        self.best_fitness = np.full(pop_size, np.inf)
        self.global_best_position = None
        self.global_best_fitness = np.inf
        self.eval_count = 0
        
        # Fuzzy Logic Controller Setup
        self.setup_fuzzy_controller()

    def setup_fuzzy_controller(self):
        # Antecedent (Input) variables
        diversity = ctrl.Antecedent(np.linspace(0, 1, 100), 'diversity') # Population diversity (0: low, 1: high)
        improvement = ctrl.Antecedent(np.linspace(-1, 0, 100), 'improvement') # Fitness improvement (-1: large, 0: none)

        # Consequent (Output) variables
        mutation_change = ctrl.Consequent(np.linspace(-0.2, 0.2, 100), 'mutation_change') # Change in mutation factor
        crossover_change = ctrl.Consequent(np.linspace(-0.2, 0.2, 100), 'crossover_change') # Change in crossover rate

        # Membership functions (adjust as needed)
        diversity['low'] = fuzz.trimf(diversity.universe, [0, 0, 0.5])
        diversity['medium'] = fuzz.trimf(diversity.universe, [0, 0.5, 1])
        diversity['high'] = fuzz.trimf(diversity.universe, [0.5, 1, 1])

        improvement['low'] = fuzz.trimf(improvement.universe, [-1, -1, -0.5])
        improvement['medium'] = fuzz.trimf(improvement.universe, [-1, -0.5, 0])
        improvement['high'] = fuzz.trimf(improvement.universe, [-0.5, 0, 0])

        mutation_change['decrease'] = fuzz.trimf(mutation_change.universe, [-0.2, -0.2, 0])
        mutation_change['no_change'] = fuzz.trimf(mutation_change.universe, [-0.1, 0, 0.1])
        mutation_change['increase'] = fuzz.trimf(mutation_change.universe, [0, 0.2, 0.2])

        crossover_change['decrease'] = fuzz.trimf(crossover_change.universe, [-0.2, -0.2, 0])
        crossover_change['no_change'] = fuzz.trimf(crossover_change.universe, [-0.1, 0, 0.1])
        crossover_change['increase'] = fuzz.trimf(crossover_change.universe, [0, 0.2, 0.2])

        # Rules (adjust as needed)
        rule1 = ctrl.Rule(diversity['low'] & improvement['low'], [mutation_change['increase'], crossover_change['decrease']])
        rule2 = ctrl.Rule(diversity['low'] & improvement['medium'], [mutation_change['increase'], crossover_change['no_change']])
        rule3 = ctrl.Rule(diversity['low'] & improvement['high'], [mutation_change['no_change'], crossover_change['increase']])
        rule4 = ctrl.Rule(diversity['medium'] & improvement['low'], [mutation_change['increase'], crossover_change['decrease']])
        rule5 = ctrl.Rule(diversity['medium'] & improvement['medium'], [mutation_change['no_change'], crossover_change['no_change']])
        rule6 = ctrl.Rule(diversity['medium'] & improvement['high'], [mutation_change['decrease'], crossover_change['increase']])
        rule7 = ctrl.Rule(diversity['high'] & improvement['low'], [mutation_change['no_change'], crossover_change['decrease']])
        rule8 = ctrl.Rule(diversity['high'] & improvement['medium'], [mutation_change['decrease'], crossover_change['no_change']])
        rule9 = ctrl.Rule(diversity['high'] & improvement['high'], [mutation_change['decrease'], crossover_change['increase']])

        # Control System
        self.tipping_ctrl = ctrl.ControlSystem([rule1, rule2, rule3, rule4, rule5, rule6, rule7, rule8, rule9])
        self.tipping = ctrl.ControlSystemSimulation(self.tipping_ctrl)
        
    def calculate_diversity(self):
        # Calculate population diversity (e.g., standard deviation of positions)
        return np.std(self.population) / 5.0 # Normalize to [0, 1]

    def __call__(self, func):
        self.eval_count = 0
        
        # Initialize fitness values
        for i in range(self.pop_size):
            if self.eval_count < self.budget:
                f = func(self.population[i])
                self.eval_count += 1
                self.fitness[i] = f
                if f < self.best_fitness[i]:
                    self.best_fitness[i] = f
                    self.best_positions[i] = self.population[i].copy()
                    if f < self.global_best_fitness:
                        self.global_best_fitness = f
                        self.global_best_position = self.population[i].copy()

        while self.eval_count < self.budget:
            previous_best_fitness = self.global_best_fitness
            for i in range(self.pop_size):
                # Differential Evolution
                r1, r2, r3 = np.random.choice(self.pop_size, 3, replace=False)
                mutant_vector = self.population[r1] + self.mutation_factor * (self.population[r2] - self.population[r3])
                mutant_vector = np.clip(mutant_vector, func.bounds.lb, func.bounds.ub)

                # Crossover
                trial_vector = np.zeros(self.dim)
                for j in range(self.dim):
                    if np.random.rand() < self.crossover_rate or j == np.random.randint(self.dim):
                        trial_vector[j] = mutant_vector[j]
                    else:
                        trial_vector[j] = self.population[i][j]

                # Evaluate trial vector
                f_trial = func(trial_vector) if self.eval_count < self.budget else np.inf
                if self.eval_count < self.budget:
                    self.eval_count += 1
                    if f_trial < self.fitness[i]:
                        self.population[i] = trial_vector
                        self.fitness[i] = f_trial
                        if f_trial < self.best_fitness[i]:
                            self.best_fitness[i] = f_trial
                            self.best_positions[i] = self.population[i].copy()
                            if f_trial < self.global_best_fitness:
                                self.global_best_fitness = f_trial
                                self.global_best_position = self.population[i].copy()

            # Calculate diversity and improvement
            diversity_value = self.calculate_diversity()
            improvement_value = (previous_best_fitness - self.global_best_fitness) / abs(previous_best_fitness) if previous_best_fitness != 0 else 0
            improvement_value = np.clip(improvement_value, -1, 0)

            # Fuzzy Inference
            self.tipping.input['diversity'] = diversity_value
            self.tipping.input['improvement'] = improvement_value
            self.tipping.compute()
            
            mutation_change = self.tipping.output['mutation_change']
            crossover_change = self.tipping.output['crossover_change']
            
            # Update mutation and crossover rates
            self.mutation_factor = np.clip(self.mutation_factor + mutation_change, 0.1, 1.0)
            self.crossover_rate = np.clip(self.crossover_rate + crossover_change, 0.1, 0.95)
            

        return self.global_best_fitness, self.global_best_position