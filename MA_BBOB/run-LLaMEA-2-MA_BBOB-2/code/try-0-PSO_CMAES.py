import numpy as np

class PSO_CMAES:
    def __init__(self, budget=10000, dim=10, pop_size=20, pso_inertia=0.7, pso_cognitive=1.4, pso_social=1.4, cmaes_sigma=0.5):
        self.budget = budget
        self.dim = dim
        self.pop_size = pop_size
        self.pso_inertia = pso_inertia
        self.pso_cognitive = pso_cognitive
        self.pso_social = pso_social
        self.cmaes_sigma = cmaes_sigma
        self.population = None
        self.velocities = None
        self.fitness = None
        self.pbest_positions = None
        self.pbest_fitness = None
        self.gbest_position = None
        self.gbest_fitness = np.inf
        self.cmaes_mean = None
        self.cmaes_covariance = None

    def initialize(self, func):
        """Initializes the population, velocities, and other parameters."""
        self.population = np.random.uniform(func.bounds.lb, func.bounds.ub, size=(self.pop_size, self.dim))
        self.velocities = np.random.uniform(-0.1, 0.1, size=(self.pop_size, self.dim))
        self.fitness = np.array([func(x) for x in self.population])
        self.budget -= self.pop_size
        self.pbest_positions = np.copy(self.population)
        self.pbest_fitness = np.copy(self.fitness)
        self.gbest_position = self.population[np.argmin(self.fitness)]
        self.gbest_fitness = np.min(self.fitness)
        self.cmaes_mean = np.copy(self.gbest_position)
        self.cmaes_covariance = np.eye(self.dim)

    def pso_step(self, func):
        """Performs a PSO update step."""
        r1 = np.random.rand(self.pop_size, self.dim)
        r2 = np.random.rand(self.pop_size, self.dim)
        
        cognitive_component = self.pso_cognitive * r1 * (self.pbest_positions - self.population)
        social_component = self.pso_social * r2 * (self.gbest_position - self.population)
        
        self.velocities = self.pso_inertia * self.velocities + cognitive_component + social_component
        self.population += self.velocities
        self.population = np.clip(self.population, func.bounds.lb, func.bounds.ub)
        
        new_fitness = np.array([func(x) for x in self.population])
        self.budget -= self.pop_size

        for i in range(self.pop_size):
            if new_fitness[i] < self.pbest_fitness[i]:
                self.pbest_fitness[i] = new_fitness[i]
                self.pbest_positions[i] = np.copy(self.population[i])

            if new_fitness[i] < self.gbest_fitness:
                self.gbest_fitness = new_fitness[i]
                self.gbest_position = np.copy(self.population[i])

        self.fitness = new_fitness

    def cmaes_step(self, func):
        """Performs a CMA-ES update step."""
        z = np.random.multivariate_normal(np.zeros(self.dim), self.cmaes_covariance, size=self.pop_size)
        new_population = self.cmaes_mean + self.cmaes_sigma * z
        new_population = np.clip(new_population, func.bounds.lb, func.bounds.ub)

        new_fitness = np.array([func(x) for x in new_population])
        self.budget -= self.pop_size

        # Selection (e.g., (mu, lambda) selection)
        combined_population = np.concatenate((self.population, new_population))
        combined_fitness = np.concatenate((self.fitness, new_fitness))
        
        sorted_indices = np.argsort(combined_fitness)
        self.population = combined_population[sorted_indices[:self.pop_size]]
        self.fitness = combined_fitness[sorted_indices[:self.pop_size]]

        # Update CMA-ES parameters (simplified)
        self.cmaes_mean = np.mean(self.population, axis=0)
        self.cmaes_covariance = np.cov(self.population.T) + 1e-8 * np.eye(self.dim) # Adding small value for numerical stability

        best_index = np.argmin(self.fitness)
        if self.fitness[best_index] < self.gbest_fitness:
            self.gbest_fitness = self.fitness[best_index]
            self.gbest_position = self.population[best_index]
        
        self.pbest_fitness = np.minimum(self.pbest_fitness, self.fitness)
        for i in range(self.pop_size):
          if self.fitness[i] < self.pbest_fitness[i]:
            self.pbest_positions[i] = self.population[i]
        

    def __call__(self, func):
        self.initialize(func)
        
        while self.budget > 0:
            if np.random.rand() < 0.5:
                self.pso_step(func)
            else:
                self.cmaes_step(func)

        return self.gbest_fitness, self.gbest_position