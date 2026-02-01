import numpy as np

class DECCMA:
    def __init__(self, budget=10000, dim=10, pop_size=None, restart_trigger=1e-8):
        self.budget = budget
        self.dim = dim
        self.pop_size = pop_size if pop_size is not None else 4 + int(3 * np.log(dim))
        self.restart_trigger = restart_trigger
        self.mu = self.pop_size // 2  # Number of parents
        self.weights = np.log(self.mu + 0.5) - np.log(np.arange(1, self.mu + 1))
        self.weights /= np.sum(self.weights)
        self.mueff = np.sum(self.weights)**2 / np.sum(self.weights**2)
        self.cs = (self.mueff + 2) / (self.dim + self.mueff + 5)
        self.damps = 1 + 2 * max(0, np.sqrt((self.mueff - 1) / (self.dim + 1)) - 1) + self.cs
        self.C = np.eye(self.dim)
        self.pc = np.zeros(self.dim)
        self.ps = np.zeros(self.dim)
        self.chiN = self.dim**0.5 * (1 - 1/(4*self.dim) + 1/(21*self.dim**2))
        self.c1 = 2 / ((self.dim + 1.3)**2 + self.mueff)
        self.cmu = min(1 - self.c1, 2 * (self.mueff - 2 + 1/self.mueff) / ((self.dim + 2)**2 + self.mueff))
        self.sigma = 0.3
        self.mean = None
        self.f_opt = np.Inf
        self.x_opt = None
        self.archive = []
        self.archive_size = 10

    def __call__(self, func):
        self.mean = np.random.uniform(func.bounds.lb, func.bounds.ub, size=self.dim)
        
        while self.budget > 0:
            # Generate population
            z = np.random.randn(self.dim, self.pop_size)
            y = self.sigma * np.dot(np.linalg.cholesky(self.C), z)
            x = self.mean[:, np.newaxis] + y
            x = np.clip(x, func.bounds.lb, func.bounds.ub)
            
            fitness = np.array([func(xi) for xi in x.T])
            self.budget -= self.pop_size

            # Sort by fitness
            idx = np.argsort(fitness)
            fitness = fitness[idx]
            x = x[:, idx]

            # Update optimal solution
            if fitness[0] < self.f_opt:
                self.f_opt = fitness[0]
                self.x_opt = x[:, 0].copy()

            # Update mean
            y_mean = np.sum(self.weights * y[:, :self.mu], axis=1)
            self.mean += self.cs * self.ps
            self.mean = np.clip(self.mean, func.bounds.lb, func.bounds.ub)

            # Update evolution path
            ps = (1 - self.cs) * self.ps + np.sqrt(self.cs * (2 - self.cs) * self.mueff) * y_mean / self.sigma
            hsig = np.sum(ps**2) / (1 - (1 - self.cs)**(2 * self.budget / self.pop_size)) / self.dim < 2 + 4 / (self.dim + 1)
            self.ps = ps
            
            # Update covariance matrix
            pc = (1 - 1) * self.pc + hsig * np.sqrt(1 * (2 - 1) * self.mueff) * (self.mean - self.mean) / self.sigma
            self.pc = pc

            delta = (1 - self.c1 - self.cmu)
            C1 = (self.c1 * (np.outer(ps, ps) + delta * self.C))
            C2 = (self.cmu * np.sum(self.weights * (y[:, :self.mu] * y[:, :self.mu].T), axis=1))
            self.C = delta * self.C + self.c1 * (np.outer(ps, ps) + delta * self.C) + self.cmu * np.sum([w * np.outer(y[:, i], y[:, i]) for i, w in enumerate(self.weights)], axis=0)
            
            # Update step size
            self.sigma *= np.exp((self.cs / self.damps) * (np.linalg.norm(self.ps) / self.chiN - 1))
            self.sigma = np.clip(self.sigma, 1e-10, 10)

            # Restart mechanism
            if np.max(np.diag(self.C)) > 1e10 or self.sigma < self.restart_trigger:
                self.C = np.eye(self.dim)
                self.ps = np.zeros(self.dim)
                self.sigma = 0.3
                self.mean = np.random.uniform(func.bounds.lb, func.bounds.ub, size=self.dim)

        return self.f_opt, self.x_opt