import numpy as np

class BlendedAdaptiveOptimizer:
    def __init__(self, budget=10000, dim=10, pop_size=20, de_mutation_factor=0.5, pso_inertia=0.7, cma_learning_rate=0.1):
        self.budget = budget
        self.dim = dim
        self.pop_size = pop_size
        self.de_mutation_factor = de_mutation_factor
        self.pso_inertia = pso_inertia
        self.cma_learning_rate = cma_learning_rate
        self.population = np.random.uniform(-5, 5, size=(pop_size, dim))
        self.fitness = np.zeros(pop_size)
        self.velocities = np.zeros((pop_size, dim))
        self.best_positions = self.population.copy()
        self.best_fitness = np.full(pop_size, np.inf)
        self.global_best_position = None
        self.global_best_fitness = np.inf
        self.covariance_matrix = np.eye(dim)
        self.adaptation_rate = 0.1
        self.min_sigma = 1e-6
        self.sigma = 0.5
        self.exploration_bias = 0.05

    def __call__(self, func):
        eval_count = 0
        while eval_count < self.budget:
            # Evaluate fitness
            for i in range(self.pop_size):
                if eval_count < self.budget:
                    f = func(self.population[i])
                    eval_count += 1
                    self.fitness[i] = f
                    if f < self.best_fitness[i]:
                        self.best_fitness[i] = f
                        self.best_positions[i] = self.population[i].copy()
                        if f < self.global_best_fitness:
                            self.global_best_fitness = f
                            self.global_best_position = self.population[i].copy()

            # Adaptive Strategy Selection & Parameter Adjustment
            for i in range(self.pop_size):
                # Blend DE, PSO, and CMA-like updates
                r1, r2, r3 = np.random.choice(self.pop_size, 3, replace=False)
                de_vector = self.best_positions[r1] + self.de_mutation_factor * (self.population[r2] - self.population[r3])
                pso_velocity = self.pso_inertia * self.velocities[i] + \
                               2.0 * np.random.rand(self.dim) * (self.best_positions[i] - self.population[i]) + \
                               2.0 * np.random.rand(self.dim) * (self.global_best_position - self.population[i])
                
                # Covariance matrix adaptation-inspired sampling
                z = np.random.multivariate_normal(np.zeros(self.dim), self.covariance_matrix)
                cma_sample = self.population[i] + self.sigma * z

                # Weighted Averaging
                weights = np.random.rand(3)
                weights /= np.sum(weights)
                new_position = weights[0] * de_vector + weights[1] * (self.population[i] + pso_velocity) + weights[2] * cma_sample
                new_position = np.clip(new_position, func.bounds.lb, func.bounds.ub)
                
                self.velocities[i] = pso_velocity
                self.population[i] = new_position
                
                # Adaptive Sigma Control
                if self.fitness[i] > self.best_fitness[i]: # No Improvement
                    self.sigma *= (1 - self.adaptation_rate)
                else:
                    self.sigma *= (1 + self.adaptation_rate)
                self.sigma = max(self.sigma, self.min_sigma)

            # Update Covariance Matrix (Simplified)
            diff = self.population - np.mean(self.population, axis=0)
            self.covariance_matrix = (1 - self.cma_learning_rate) * self.covariance_matrix + self.cma_learning_rate * np.cov(diff.T)
            # Ensure the covariance matrix is positive semi-definite
            try:
                _ = np.linalg.cholesky(self.covariance_matrix)
            except np.linalg.LinAlgError:
                self.covariance_matrix = np.eye(self.dim)  # Reset if not PSD
        return self.global_best_fitness, self.global_best_position