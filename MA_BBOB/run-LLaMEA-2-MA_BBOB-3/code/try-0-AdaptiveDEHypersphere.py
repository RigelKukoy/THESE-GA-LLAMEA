import numpy as np

class AdaptiveDEHypersphere:
    def __init__(self, budget=10000, dim=10, pop_size=20, mutation_factor=0.5, crossover_rate=0.7, stagnation_threshold=500, sphere_reduction_factor=0.95):
        self.budget = budget
        self.dim = dim
        self.pop_size = pop_size
        self.mutation_factor = mutation_factor
        self.crossover_rate = crossover_rate
        self.stagnation_threshold = stagnation_threshold
        self.sphere_reduction_factor = sphere_reduction_factor
        self.population = np.random.uniform(-5, 5, size=(pop_size, dim))
        self.fitness = np.zeros(pop_size)
        self.best_position = None
        self.best_fitness = np.inf
        self.eval_count = 0
        self.stagnation_counter = 0
        self.sphere_radius = 1.0  # Initial radius for hypersphere search

    def __call__(self, func):
        self.eval_count = 0
        self.stagnation_counter = 0
        self.sphere_radius = 1.0

        # Initialize fitness values
        for i in range(self.pop_size):
            if self.eval_count < self.budget:
                f = func(self.population[i])
                self.eval_count += 1
                self.fitness[i] = f
                if f < self.best_fitness:
                    self.best_fitness = f
                    self.best_position = self.population[i].copy()

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
                        if f_trial < self.best_fitness:
                            self.best_fitness = f_trial
                            self.best_position = self.population[i].copy()
                            self.stagnation_counter = 0
                    else:
                        self.stagnation_counter += 1

            # Stagnation check and Hypersphere Search
            if self.stagnation_counter > self.stagnation_threshold:
                # Generate new samples within a hypersphere around the best solution
                for i in range(self.pop_size):
                    if self.eval_count < self.budget:
                        # Generate a random point within the hypersphere
                        direction = np.random.randn(self.dim)
                        direction /= np.linalg.norm(direction)  # Normalize to unit vector
                        sample = self.best_position + direction * np.random.rand() * self.sphere_radius
                        sample = np.clip(sample, func.bounds.lb, func.bounds.ub)
                        
                        f_sample = func(sample)
                        self.eval_count += 1

                        if f_sample < self.best_fitness:
                            self.best_fitness = f_sample
                            self.best_position = sample.copy()
                            self.stagnation_counter = 0
                        
                        self.population[i] = sample
                        self.fitness[i] = f_sample

                self.sphere_radius *= self.sphere_reduction_factor #Reduce radius
                self.stagnation_counter = 0  # Reset stagnation counter

        return self.best_fitness, self.best_position