import numpy as np

class AdaptivePopulationSearchCMA:
    def __init__(self, budget=10000, dim=10, pop_size=20, exploration_rate=0.5, cma_decay=0.9):
        self.budget = budget
        self.dim = dim
        self.pop_size = pop_size
        self.exploration_rate = exploration_rate
        self.lb = -5.0
        self.ub = 5.0
        self.cma_decay = cma_decay
        self.cma_sigma = 0.5  # Initial CMA step size
        self.cma_C = np.eye(dim)  # Initial covariance matrix

    def __call__(self, func):
        # Initialize population
        population = np.random.uniform(self.lb, self.ub, size=(self.pop_size, self.dim))
        fitness = np.array([func(x) for x in population])
        self.budget -= self.pop_size

        # Find best individual
        best_index = np.argmin(fitness)
        self.f_opt = fitness[best_index]
        self.x_opt = population[best_index]

        # Main loop
        while self.budget > 0:
            # Adaptive adjustment of exploration rate based on fitness variance
            fitness_variance = np.var(fitness)
            if fitness_variance > 1e-3: # If variance is high, explore more
                exploration_rate = min(self.exploration_rate + 0.1, 0.9)
            else: # If variance is low, exploit more
                exploration_rate = max(self.exploration_rate - 0.1, 0.1)

            new_population = np.zeros_like(population)

            for i in range(self.pop_size):
                if np.random.rand() < exploration_rate:
                    # Exploration: Orthogonal Crossover
                    parent1_idx = np.random.randint(0, self.pop_size)
                    parent2_idx = np.random.randint(0, self.pop_size)
                    while parent2_idx == parent1_idx:
                         parent2_idx = np.random.randint(0, self.pop_size)

                    new_population[i] = 0.5 * (population[parent1_idx] + population[parent2_idx]) + np.random.uniform(-1.0, 1.0, size=self.dim) * (self.ub - self.lb) * 0.05

                    new_population[i] = np.clip(new_population[i], self.lb, self.ub)
                else:
                    # Exploitation: CMA-ES sampling
                    z = np.random.normal(0, 1, size=self.dim)
                    new_population[i] = self.x_opt + self.cma_sigma * np.dot(np.linalg.cholesky(self.cma_C), z)
                    new_population[i] = np.clip(new_population[i], self.lb, self.ub)

            new_fitness = np.array([func(x) for x in new_population])
            self.budget -= self.pop_size
            
            # Update population (replace if better)
            for i in range(self.pop_size):
                if new_fitness[i] < fitness[i]:
                    population[i] = new_population[i]
                    fitness[i] = new_fitness[i]
                    
                    if new_fitness[i] < self.f_opt:
                        self.f_opt = new_fitness[i]
                        self.x_opt = new_population[i]

            # CMA-ES covariance matrix adaptation (simplified)
            if self.budget > 0:
                diff = population - self.x_opt
                self.cma_C = self.cma_decay * self.cma_C + (1 - self.cma_decay) * np.cov(diff.T)
                self.cma_sigma *= np.exp(0.1 * (np.mean(new_fitness) - np.mean(fitness)))

        return self.f_opt, self.x_opt