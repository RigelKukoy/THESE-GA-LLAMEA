import numpy as np

class AdaptiveDEwithRestart:
    def __init__(self, budget=10000, dim=10, pop_size=20, mutation_factor=0.5, initial_crossover_rate=0.7, stagnation_threshold=1000):
        self.budget = budget
        self.dim = dim
        self.pop_size = pop_size
        self.mutation_factor = mutation_factor
        self.crossover_rate = initial_crossover_rate
        self.initial_crossover_rate = initial_crossover_rate
        self.stagnation_threshold = stagnation_threshold
        self.population = np.random.uniform(-5, 5, size=(pop_size, dim))
        self.fitness = np.zeros(pop_size)
        self.best_positions = self.population.copy()
        self.best_fitness = np.full(pop_size, np.inf)
        self.global_best_position = None
        self.global_best_fitness = np.inf
        self.eval_count = 0
        self.stagnation_counter = 0

    def __call__(self, func):
        self.eval_count = 0
        self.stagnation_counter = 0
        self.crossover_rate = self.initial_crossover_rate
        
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
        
        last_best_fitness = self.global_best_fitness

        while self.eval_count < self.budget:
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
                                self.stagnation_counter = 0 # Reset stagnation counter
                    else:
                        self.stagnation_counter += 1

            # Adjust crossover rate
            if self.stagnation_counter > self.stagnation_threshold:
                # If stagnation is detected, increase crossover rate to explore more
                self.crossover_rate = min(1.0, self.crossover_rate + 0.1)
            else:
                # Reduce crossover rate to exploit the current best solutions
                self.crossover_rate = max(0.1, self.crossover_rate - 0.05)

            # Restart mechanism
            if self.stagnation_counter > 2 * self.stagnation_threshold:
                # Re-initialize population around the best solution
                self.population = np.random.normal(loc=self.global_best_position, scale=0.5, size=(self.pop_size, self.dim))
                self.population = np.clip(self.population, func.bounds.lb, func.bounds.ub)
                
                #Re-evaluate the population
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
                self.stagnation_counter = 0
                self.crossover_rate = self.initial_crossover_rate

        return self.global_best_fitness, self.global_best_position