import numpy as np

class CMAES_DE:
    def __init__(self, budget=10000, dim=10, popsize=None, initial_sigma=0.1, local_search_prob=0.1):
        self.budget = budget
        self.dim = dim
        self.popsize = popsize if popsize is not None else 4 + int(3 * np.log(self.dim))
        self.initial_sigma = initial_sigma
        self.local_search_prob = local_search_prob
        self.mean = None  # Initialize in __call__
        self.C = None   # Initialize in __call__
        self.sigma = None # Initialize in __call__
        self.ps = None  # Initialize in __call__
        self.pc = None  # Initialize in __call__
        self.chiN = None # Initialize in __call__

        self.c_sigma = None
        self.d_sigma = None
        self.c_c = None
        self.mu = self.popsize // 2
        self.weights = np.log(self.mu + 1/2) - np.log(np.arange(1, self.mu + 1))
        self.weights = self.weights / np.sum(self.weights)
        self.mueff = np.sum(self.weights)**2 / np.sum(self.weights**2)

        self.c_1 = 2 / ((self.dim + 1.3)**2 + self.mueff)
        self.c_mu = min(1 - self.c_1, 2 * (self.mueff - 2 + 1/self.mueff) / ((self.dim + 2)**2 + self.mueff))
        self.c_sigma = (self.mueff + 2) / (self.dim + self.mueff + 5)
        self.d_sigma = 1 + 2 * max(0, np.sqrt((self.mueff - 1)/(self.dim + 1)) - 1) + self.c_sigma
        self.c_c = (4 + self.mueff/self.dim) / (self.dim + 4 + 2*self.mueff/self.dim)


    def __call__(self, func):
        lb = func.bounds.lb
        ub = func.bounds.ub
        self.mean = np.random.uniform(lb, ub, size=self.dim)
        self.sigma = self.initial_sigma
        self.C = np.eye(self.dim)  # Covariance matrix
        self.ps = np.zeros(self.dim)
        self.pc = np.zeros(self.dim)
        self.chiN = self.dim**0.5 * (1 - 1/(4*self.dim) + 1/(21*self.dim**2))


        self.f_opt = np.Inf
        self.x_opt = None
        self.eval_count = 0

        while self.eval_count < self.budget:
            # Generate population
            z = np.random.randn(self.popsize, self.dim)
            x = self.mean + self.sigma * z @ np.linalg.cholesky(self.C).T
            x = np.clip(x, lb, ub)
            fitness = np.array([func(xi) for xi in x])
            self.eval_count += self.popsize

            # Sort by fitness
            idx_sorted = np.argsort(fitness)
            fitness = fitness[idx_sorted]
            x = x[idx_sorted]
            
            # Local search on best individual
            if np.random.rand() < self.local_search_prob:
                x_local = self.local_search(x[0], func, lb, ub)
                f_local = func(x_local)
                self.eval_count += 1
                if f_local < fitness[0]:
                    fitness[0] = f_local
                    x[0] = x_local

            if fitness[0] < self.f_opt:
                self.f_opt = fitness[0]
                self.x_opt = x[0]

            # Update CMA-ES parameters
            z_sorted = z[idx_sorted]
            y = x - self.mean
            y_w = np.sum(self.weights.reshape(-1, 1) * y[:self.mu], axis=0)

            # Update evolution paths
            self.ps = (1 - self.c_sigma) * self.ps + np.sqrt(self.c_sigma * (2 - self.c_sigma) * self.mueff) * (np.linalg.inv(np.linalg.cholesky(self.C)) @ y_w) / self.sigma
            self.pc = (1 - self.c_c) * self.pc + np.sqrt(self.c_c * (2 - self.c_c) * self.mueff) * y_w / self.sigma

            # Update covariance matrix
            self.C = (1 - self.c_1 - self.c_mu) * self.C + self.c_1 * (np.outer(self.pc, self.pc) + (1 - (np.sum(self.pc**2) / (self.dim))) * self.C) + self.c_mu * np.sum(self.weights.reshape(-1, 1, 1) * np.array([np.outer(y[i], y[i]) for i in range(self.mu)]), axis=0) / (self.sigma**2)

            # Update step size
            self.sigma = self.sigma * np.exp((self.c_sigma / self.d_sigma) * (np.linalg.norm(self.ps) / self.chiN - 1))
            self.mean = np.sum(self.weights.reshape(-1, 1) * x[:self.mu], axis=0)


        return self.f_opt, self.x_opt
    
    def local_search(self, x, func, lb, ub, radius=0.1):
        """Performs a simple local search around x."""
        best_x = x
        best_f = func(x)
        
        num_samples = 20  # Number of samples for local search
        
        for _ in range(num_samples):
            x_new = x + np.random.uniform(-radius, radius, size=self.dim)
            x_new = np.clip(x_new, lb, ub)
            f_new = func(x_new)
            
            if f_new < best_f:
                best_f = f_new
                best_x = x_new
                
        return best_x