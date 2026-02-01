import numpy as np

class AdaptiveCMAES:
    def __init__(self, budget=10000, dim=10, pop_size=None, initial_sigma=0.2):
        self.budget = budget
        self.dim = dim
        self.pop_size = pop_size if pop_size is not None else 4 + int(3 * np.log(dim))
        self.initial_sigma = initial_sigma
        self.mean = None
        self.covariance = None
        self.sigma = None
        self.pc = None
        self.ps = None
        self.C = None
        self.eigenvalues = None
        self.eigenbasis = None
        self.mu = int(self.pop_size / 2)
        self.weights = np.log(self.mu + 0.5) - np.log(np.arange(1, self.mu + 1))
        self.weights /= np.sum(self.weights)
        self.mueff = np.sum(self.weights)**2 / np.sum(self.weights**2)
        self.cc = (4 + self.mueff / self.dim) / (self.dim + 4 + 2 * self.mueff / self.dim)
        self.cs = (self.mueff + 2) / (self.dim + self.mueff + 5)
        self.c1 = 2 / ((self.dim + 1.3)**2 + self.mueff)
        self.cmu = min(1 - self.c1, 2 * (self.mueff - 2 + 1 / self.mueff) / ((self.dim + 2)**2 + self.mueff))
        self.damps = 1 + 2 * max(0, np.sqrt((self.mueff - 1) / (self.dim + 1)) - 1) + self.cs
        self.x_opt = None
        self.f_opt = np.inf

    def initialize(self, lb, ub):
        self.mean = np.random.uniform(lb, ub, size=self.dim)
        self.sigma = self.initial_sigma
        self.covariance = np.eye(self.dim)
        self.pc = np.zeros(self.dim)
        self.ps = np.zeros(self.dim)
        self.C = self.covariance
        self.eigenvalues, self.eigenbasis = np.linalg.eigh(self.C)

    def sample_population(self):
        z = np.random.normal(size=(self.pop_size, self.dim))
        samples = self.mean + self.sigma * self.eigenbasis @ (np.sqrt(self.eigenvalues) * z.T).T
        return samples

    def update_distribution(self, samples, fitness_values):
        sorted_indices = np.argsort(fitness_values)
        best_samples = samples[sorted_indices[:self.mu]]

        # Calculate weighted mean of best samples
        delta_mean = np.sum((best_samples - self.mean) * self.weights[:, np.newaxis], axis=0)
        
        # Update evolution path
        self.ps = (1 - self.cs) * self.ps + np.sqrt(self.cs * (2 - self.cs) * self.mueff) * (self.eigenbasis @ (delta_mean / self.sigma))
        
        # Length control
        sigma_norm = np.linalg.norm(self.ps) / np.sqrt(1 - (1 - self.cs)**(2 * self.budget / self.pop_size)) / 1.414 / ((np.sqrt(self.dim) * (1 - 1/(4*self.dim) + 1/(21 * self.dim**2))))
        if sigma_norm < 0.2:
            self.sigma *= 0.8
        if sigma_norm > 5:
            self.sigma *= 1.2

        hsig = (np.linalg.norm(self.ps) / np.sqrt(1 - (1 - self.cs)**(2 * self.budget / self.pop_size)) < (1.4 + 2/(self.dim+1)))

        self.pc = (1 - self.cc) * self.pc + hsig * np.sqrt(self.cc * (2 - self.cc) * self.mueff) * delta_mean / self.sigma

        # Adapt covariance matrix
        artmp = (best_samples - self.mean) / self.sigma
        self.C = (1 - self.c1 - self.cmu) * self.C + self.c1 * (self.pc[:, np.newaxis] @ self.pc[np.newaxis, :]) + self.cmu * np.sum(self.weights[:, np.newaxis, np.newaxis] * artmp[:, :, np.newaxis] @ artmp[:, np.newaxis, :], axis=0)

        self.mean += delta_mean
        self.C = np.triu(self.C) + np.triu(self.C, 1).T  # enforce symmetry
        self.eigenvalues, self.eigenbasis = np.linalg.eigh(self.C)
        self.eigenvalues = np.maximum(self.eigenvalues, 1e-10)  # Avoid zero eigenvalues

    def __call__(self, func):
        lb, ub = func.bounds.lb, func.bounds.ub
        self.initialize(lb, ub)
        
        while self.budget > 0:
            samples = self.sample_population()
            samples = np.clip(samples, lb, ub)
            fitness_values = np.array([func(x) for x in samples])
            self.budget -= self.pop_size
            
            best_index = np.argmin(fitness_values)
            if fitness_values[best_index] < self.f_opt:
                self.f_opt = fitness_values[best_index]
                self.x_opt = samples[best_index]

            self.update_distribution(samples, fitness_values)
            
            # Adaptive Sigma Scaling based on Fitness Variance
            fitness_std = np.std(fitness_values)
            if fitness_std < 1e-3:  # Low variance, converge faster
               self.sigma *= 0.95
            elif fitness_std > 1.0: # High variance, explore more
                self.sigma *= 1.05

        return self.f_opt, self.x_opt