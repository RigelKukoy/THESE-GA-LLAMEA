import numpy as np

class BudgetAwareCMAES:
    def __init__(self, budget=10000, dim=10, pop_size=None, sigma_init=0.5, restart_trigger=3):
        self.budget = budget
        self.dim = dim
        self.pop_size = pop_size if pop_size is not None else 4 + int(3 * np.log(dim))
        self.sigma = sigma_init
        self.mean = None
        self.C = None
        self.restart_trigger = restart_trigger
        self.f_opt = np.inf
        self.x_opt = None
        self.success_counter = 0
        self.mu = self.pop_size // 2  # Number of individuals for recombination
        self.weights = np.log(self.mu + 0.5) - np.log(np.arange(1, self.mu + 1))
        self.weights /= np.sum(self.weights)
        self.mueff = np.sum(self.weights)**2 / np.sum(self.weights**2)

        self.cc = (4 + self.mueff/self.dim) / (self.dim + 4 + 2*self.mueff/self.dim)
        self.cs = (self.mueff + 2) / (self.dim + self.mueff + 5)
        self.c1 = 2 / ((self.dim + 1.3)**2 + self.mueff)
        self.cmu = min(1-self.c1, 2 * (self.mueff - 2 + 1/self.mueff) / ((self.dim + 2.0)**2 + self.mueff))
        self.damps = 1 + 2*max(0, np.sqrt((self.mueff-1)/(self.dim+1)) - 1) + self.cs

        self.pc = np.zeros(self.dim)
        self.ps = np.zeros(self.dim)
        self.B = None
        self.D = None
        self.eigen_updated = 0

    def initialize(self, func):
        self.mean = np.random.uniform(func.bounds.lb, func.bounds.ub, size=self.dim)
        self.C = np.eye(self.dim)
        self.B = np.eye(self.dim)
        self.D = np.ones(self.dim)
        self.eigen_updated = 0
        self.pc = np.zeros(self.dim)
        self.ps = np.zeros(self.dim)

    def sample_population(self, func):
        z = np.random.normal(0, 1, size=(self.pop_size, self.dim))
        y = self.B.dot(self.D * z.T).T
        x = self.mean + self.sigma * y
        x = np.clip(x, func.bounds.lb, func.bounds.ub)
        return x

    def update_distribution(self, x, fitness):
        # Sort by fitness
        indices = np.argsort(fitness)
        x_sorted = x[indices]

        # Update mean
        y = (x_sorted[:self.mu] - self.mean) / self.sigma
        self.mean += self.sigma * np.sum(self.weights[:, None] * y, axis=0)

        # Update evolution path
        self.ps = (1 - self.cs) * self.ps + np.sqrt(self.cs * (2 - self.cs) * self.mueff) * (self.B @ self.D) @ np.mean(np.random.normal(0, 1, size=(self.mu, self.dim)), axis=0)
        hsig = np.linalg.norm(self.ps)/np.sqrt(1-(1-self.cs)**(2*self.budget/self.pop_size))/np.sqrt(self.dim+1) < 1.4 + 2/(self.dim+1)
        self.pc = (1 - self.cc) * self.pc + hsig * np.sqrt(self.cc * (2 - self.cc) * self.mueff) * y[0]

        # Update covariance matrix
        self.C = (1 - self.c1 - self.cmu) * self.C + self.c1 * (self.pc[:, None] @ self.pc[None, :]) + self.cmu * np.sum(self.weights[:,None,None] * y[:,:,None] @ y[:,None,:], axis=0)

        self.sigma *= np.exp((self.cs/self.damps)*(np.linalg.norm(self.ps)/np.sqrt(self.dim)/self.chiN - 1))
        self.sigma = np.clip(self.sigma, 1e-10, 5)

        self.eigen_updated += 1
        if self.eigen_updated > self.pop_size / (self.c1 + self.cmu) / self.dim / 10:
            self.eigen_decomposition()
            self.eigen_updated = 0

    def eigen_decomposition(self):
        self.C = np.triu(self.C) + np.triu(self.C, 1).T
        self.D, self.B = np.linalg.eigh(self.C)
        self.D = np.sqrt(np.maximum(self.D, 1e-10))

    def __call__(self, func):
        self.chiN = self.dim**0.5*(1-1/(4*self.dim)+1/(21*self.dim**2))
        self.initialize(func)
        
        iteration = 0
        while self.budget > self.pop_size:
            iteration += 1
            # Sample population
            x = self.sample_population(func)
            
            # Evaluate population
            fitness = np.array([func(xi) for xi in x])
            self.budget -= self.pop_size

            # Update best solution
            best_index_batch = np.argmin(fitness)
            if fitness[best_index_batch] < self.f_opt:
                self.f_opt = fitness[best_index_batch]
                self.x_opt = x[best_index_batch].copy()
                self.success_counter = 0
            else:
                self.success_counter +=1

            # Update distribution parameters
            self.update_distribution(x, fitness)

            if self.success_counter > self.restart_trigger * self.dim: #Restarting criterion based on stagnation
                self.initialize(func)
                self.sigma = 0.5 # Resetting sigma
                self.success_counter = 0 # Resetting the stagnation counter
        
        # Final evaluation (optional, if budget allows)
        if self.budget > 0:
            x = np.random.uniform(func.bounds.lb, func.bounds.ub, size=(min(self.budget,self.pop_size), self.dim))
            fitness = np.array([func(xi) for xi in x])
            best_index_batch = np.argmin(fitness)
            if fitness[best_index_batch] < self.f_opt:
                self.f_opt = fitness[best_index_batch]
                self.x_opt = x[best_index_batch].copy()
            self.budget = 0 # making sure budget is not exceeded.

        return self.f_opt, self.x_opt