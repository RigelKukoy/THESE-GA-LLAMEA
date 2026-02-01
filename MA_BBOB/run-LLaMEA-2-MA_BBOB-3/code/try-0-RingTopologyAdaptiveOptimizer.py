import numpy as np

class RingTopologyAdaptiveOptimizer:
    def __init__(self, budget=10000, dim=10, pop_size=20, de_mutation_factor=0.5, local_search_prob=0.1, diversity_threshold=0.01):
        self.budget = budget
        self.dim = dim
        self.pop_size = pop_size
        self.de_mutation_factor = de_mutation_factor
        self.local_search_prob = local_search_prob
        self.diversity_threshold = diversity_threshold
        self.population = np.random.uniform(-5, 5, size=(pop_size, dim))
        self.fitness = np.zeros(pop_size)
        self.best_positions = self.population.copy()
        self.best_fitness = np.full(pop_size, np.inf)
        self.global_best_position = None
        self.global_best_fitness = np.inf
        self.eval_count = 0

    def __call__(self, func):
        self.eval_count = 0
        while self.eval_count < self.budget:
            # Evaluate fitness
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

            # Calculate population diversity
            diversity = np.std(self.population)

            # Update population using ring topology and adaptive strategy
            for i in range(self.pop_size):
                neighbor_left = (i - 1) % self.pop_size
                neighbor_right = (i + 1) % self.pop_size

                # Differential Evolution with neighbors
                r1, r2 = np.random.choice(self.pop_size, 2, replace=False)
                
                de_vector = self.population[i] + self.de_mutation_factor * (self.population[neighbor_left] - self.population[neighbor_right])
                de_vector = np.clip(de_vector, func.bounds.lb, func.bounds.ub)

                # Local Search (only if population is diverse enough)
                if np.random.rand() < self.local_search_prob and diversity > self.diversity_threshold:
                    # Apply a small perturbation to the current individual
                    perturbation = np.random.uniform(-0.1, 0.1, size=self.dim)
                    local_search_vector = self.population[i] + perturbation
                    local_search_vector = np.clip(local_search_vector, func.bounds.lb, func.bounds.ub)
                    
                    # Choose between DE and Local Search based on a random probability
                    if np.random.rand() < 0.5:
                        new_position = de_vector
                    else:
                        new_position = local_search_vector
                else:
                     new_position = de_vector
                
                # Update individual only if it improves fitness
                f_new = func(new_position) if self.eval_count < self.budget else np.inf
                if self.eval_count < self.budget:
                    self.eval_count += 1
                    if f_new < self.fitness[i]:
                        self.population[i] = new_position
                        self.fitness[i] = f_new
                        if f_new < self.best_fitness[i]:
                            self.best_fitness[i] = f_new
                            self.best_positions[i] = self.population[i].copy()
                            if f_new < self.global_best_fitness:
                                self.global_best_fitness = f_new
                                self.global_best_position = self.population[i].copy()

        return self.global_best_fitness, self.global_best_position