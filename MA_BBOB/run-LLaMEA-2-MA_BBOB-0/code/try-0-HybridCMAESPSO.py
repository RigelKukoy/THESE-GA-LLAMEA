import numpy as np

class HybridCMAESPSO:
    def __init__(self, budget=10000, dim=10, pop_size=20, cma_sigma0=0.5, pso_inertia=0.7, pso_cognitive=1.4, pso_social=1.4):
        self.budget = budget
        self.dim = dim
        self.pop_size = pop_size
        self.cma_sigma0 = cma_sigma0
        self.pso_inertia = pso_inertia
        self.pso_cognitive = pso_cognitive
        self.pso_social = pso_social
        self.mean = None
        self.covariance = None
        self.population = None
        self.fitness = None
        self.velocities = None
        self.personal_best_positions = None
        self.personal_best_fitness = None
        self.global_best_position = None
        self.global_best_fitness = np.inf

    def initialize(self, lb, ub):
        self.mean = np.random.uniform(lb, ub, size=self.dim)
        self.covariance = np.eye(self.dim) * self.cma_sigma0**2
        self.population = np.random.multivariate_normal(self.mean, self.covariance, size=self.pop_size)
        self.population = np.clip(self.population, lb, ub)
        self.velocities = np.random.uniform(-0.1*(ub-lb), 0.1*(ub-lb), size=(self.pop_size, self.dim))
        self.personal_best_positions = np.copy(self.population)
        self.personal_best_fitness = np.full(self.pop_size, np.inf)
        
    def sample_population(self):
        self.population = np.random.multivariate_normal(self.mean, self.covariance, size=self.pop_size)
        self.population = np.clip(self.population, func.bounds.lb, func.bounds.ub)

    def update_cma(self):
        # Simple CMA-ES update rules (can be improved)
        weights = np.sort(np.random.rand(self.pop_size))[::-1]
        weights /= np.sum(weights)

        delta_x = self.population - self.mean
        self.mean = np.sum(weights[:, np.newaxis] * self.population, axis=0)

        self.covariance = np.cov(delta_x.T, aweights=weights) + np.eye(self.dim) * self.cma_sigma0**2 #Regularization

    def update_pso(self):
        r1 = np.random.rand(self.pop_size, self.dim)
        r2 = np.random.rand(self.pop_size, self.dim)

        cognitive_component = self.pso_cognitive * r1 * (self.personal_best_positions - self.population)
        social_component = self.pso_social * r2 * (self.global_best_position - self.population)

        self.velocities = (self.pso_inertia * self.velocities + cognitive_component + social_component)
        self.population += self.velocities
        self.population = np.clip(self.population, func.bounds.lb, func.bounds.ub)

    def __call__(self, func):
        lb, ub = func.bounds.lb, func.bounds.ub
        self.initialize(lb, ub)
        
        self.fitness = np.array([func(x) for x in self.population])
        self.budget -= self.pop_size
        
        for i in range(self.pop_size):
            if self.fitness[i] < self.personal_best_fitness[i]:
                self.personal_best_fitness[i] = self.fitness[i]
                self.personal_best_positions[i] = self.population[i]
            if self.fitness[i] < self.global_best_fitness:
                self.global_best_fitness = self.fitness[i]
                self.global_best_position = self.population[i]
        
        while self.budget > 0:
            # Alternate between CMA-ES and PSO steps
            if self.budget % 2 == 0:
                self.update_cma()
                self.sample_population()
            else:
                self.update_pso()

            self.fitness = np.array([func(x) for x in self.population])
            self.budget -= self.pop_size
            
            for i in range(self.pop_size):
                if self.fitness[i] < self.personal_best_fitness[i]:
                    self.personal_best_fitness[i] = self.fitness[i]
                    self.personal_best_positions[i] = self.population[i]
                if self.fitness[i] < self.global_best_fitness:
                    self.global_best_fitness = self.fitness[i]
                    self.global_best_position = self.population[i]

        return self.global_best_fitness, self.global_best_position