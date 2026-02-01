import numpy as np

class AdaptiveGaussianSearch:
    def __init__(self, budget=10000, dim=10, initial_pop_size=20, initial_std=1.0, target_success_rate=0.3, adaptation_rate=0.1):
        self.budget = budget
        self.dim = dim
        self.initial_pop_size = initial_pop_size
        self.pop_size = initial_pop_size
        self.initial_std = initial_std
        self.lb = -5.0
        self.ub = 5.0
        self.target_success_rate = target_success_rate
        self.adaptation_rate = adaptation_rate
        self.eval_count = 0
        self.mean = np.zeros(dim)
        self.C = np.eye(dim) * self.initial_std**2  # Covariance matrix
        self.pc = np.zeros(dim)  # Evolution path for C
        self.ps = np.zeros(dim)  # Evolution path for step size
        self.chiN = np.sqrt(dim) * (1 - (1 / (4 * dim)) + (1 / (21 * dim**2)))  # Expectation of ||N(0,I)||
        self.c_sigma = (self.adaptation_rate * (self.pop_size + 2)) / (dim + 3)
        self.c_c = (4 + dim / 3) * self.adaptation_rate / (dim + 4)
        self.c_mu = self.adaptation_rate * (self.pop_size - 2 + 1 / self.pop_size) / (np.power(dim + 2, 2))
        self.d_sigma = 1 + 2 * max(0, np.sqrt((self.c_sigma * (dim - 1)) / (1 - self.c_sigma)) - 1)

    def __call__(self, func):
        self.f_opt = np.Inf
        self.x_opt = None

        while self.eval_count < self.budget:
            # Generate population
            z = np.random.multivariate_normal(np.zeros(self.dim), self.C, size=self.pop_size)
            population = self.mean + z
            population = np.clip(population, self.lb, self.ub)
            
            # Evaluate population
            fitness = np.array([func(x) for x in population])
            self.eval_count += self.pop_size

            # Find best solution in population
            best_index = np.argmin(fitness)
            if fitness[best_index] < self.f_opt:
                self.f_opt = fitness[best_index]
                self.x_opt = population[best_index]
            
            # Sort population and fitness
            indices = np.argsort(fitness)
            population = population[indices]
            fitness = fitness[indices]

            # Update mean
            old_mean = self.mean.copy()
            self.mean = np.mean(population[:self.pop_size // 2], axis=0)
            
            # Cumulation for covariance matrix
            self.ps = (1 - self.c_sigma) * self.ps + np.sqrt(self.c_sigma * (2 - self.c_sigma)) * np.linalg.solve(self.C, (self.mean - old_mean))
            
            # Cumulation for step size control
            norm_ps = np.linalg.norm(self.ps)
            self.pc = (1 - self.c_c) * self.pc + np.sqrt(self.c_c * (2 - self.c_c)) * (self.mean - old_mean)

            # Update covariance matrix
            delta = population[:self.pop_size // 2] - old_mean
            self.C = (1 - self.c_mu) * self.C + self.c_mu * np.sum([np.outer(delta[i], delta[i]) for i in range(self.pop_size // 2)], axis=0)

            # Ensure C is positive definite (numerical stability)
            try:
                np.linalg.cholesky(self.C)
            except np.linalg.LinAlgError:
                self.C = self.C + np.eye(self.dim) * 1e-6
                
            # Adapt population size
            success_rate = (fitness[0] < self.f_opt)
            if success_rate > self.target_success_rate:
                self.pop_size = min(int(self.pop_size / (1 - self.adaptation_rate)), self.budget // 10)  # Ensure not too large pop_size
            elif success_rate < self.target_success_rate:
                self.pop_size = max(int(self.pop_size * (1 - self.adaptation_rate)), 2)

            if self.eval_count >= self.budget:
                break

        return self.f_opt, self.x_opt