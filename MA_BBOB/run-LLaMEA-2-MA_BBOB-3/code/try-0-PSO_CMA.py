import numpy as np

class PSO_CMA:
    def __init__(self, budget=10000, dim=10, pop_size=20, w=0.7, c1=1.5, c2=1.5, initial_sigma=0.5):
        self.budget = budget
        self.dim = dim
        self.pop_size = pop_size
        self.w = w  # Inertia weight
        self.c1 = c1 # Cognitive coefficient
        self.c2 = c2 # Social coefficient
        self.initial_sigma = initial_sigma
        self.population = np.random.uniform(-5, 5, size=(pop_size, dim))
        self.velocities = np.random.uniform(-1, 1, size=(pop_size, dim))
        self.fitness = np.zeros(pop_size)
        self.personal_best_positions = self.population.copy()
        self.personal_best_fitness = np.full(pop_size, np.inf)
        self.global_best_position = None
        self.global_best_fitness = np.inf
        self.eval_count = 0
        self.mean = np.zeros(dim)
        self.covariance = np.eye(dim) * (self.initial_sigma**2)

    def __call__(self, func):
        self.eval_count = 0
        while self.eval_count < self.budget:
            # Evaluate fitness
            for i in range(self.pop_size):
                if self.eval_count < self.budget:
                    f = func(self.population[i])
                    self.eval_count += 1
                    self.fitness[i] = f
                    if f < self.personal_best_fitness[i]:
                        self.personal_best_fitness[i] = f
                        self.personal_best_positions[i] = self.population[i].copy()
                        if f < self.global_best_fitness:
                            self.global_best_fitness = f
                            self.global_best_position = self.population[i].copy()

            # Update mean and covariance matrix using CMA-ES principles
            self.mean = np.mean(self.population, axis=0)
            self.covariance = np.cov(self.population.T) + np.eye(self.dim) * (self.initial_sigma/10) **2 # Add a small diagonal matrix for numerical stability

            # Update velocities and positions
            for i in range(self.pop_size):
                r1 = np.random.rand(self.dim)
                r2 = np.random.rand(self.dim)

                cognitive_component = self.c1 * r1 * (self.personal_best_positions[i] - self.population[i])
                social_component = self.c2 * r2 * (self.global_best_position - self.population[i])

                # Sample from multivariate normal distribution based on CMA
                innovation = np.random.multivariate_normal(np.zeros(self.dim), self.covariance)
                
                self.velocities[i] = self.w * self.velocities[i] + cognitive_component + social_component + innovation

                self.population[i] = self.population[i] + self.velocities[i]
                self.population[i] = np.clip(self.population[i], func.bounds.lb, func.bounds.ub)

        return self.global_best_fitness, self.global_best_position