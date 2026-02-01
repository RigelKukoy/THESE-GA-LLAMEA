import numpy as np

class BudgetAwareCMAES:
    def __init__(self, budget=10000, dim=10, popsize=None, sigma0=0.5, restart_factor=3.0):
        self.budget = budget
        self.dim = dim
        self.popsize = popsize  # Initial popsize will be set adaptively
        self.sigma = sigma0
        self.mean = None
        self.C = None
        self.pc = None
        self.ps = None
        self.chiN = None
        self.eval_count = 0
        self.f_opt = np.inf
        self.x_opt = None
        self.restart_factor = restart_factor
        self.restart_count = 0
        self.min_popsize = 4 + int(3 * np.log(self.dim))
        self.max_popsize = 4 + int(8 * np.log(self.dim))

    def initialize(self, func):
        self.mean = np.random.uniform(func.bounds.lb, func.bounds.ub, size=self.dim)
        self.popsize = self.min_popsize
        self.C = np.eye(self.dim)
        self.pc = np.zeros(self.dim)
        self.ps = np.zeros(self.dim)
        self.chiN = self.dim**0.5 * (1 - 1/(4*self.dim) + 1/(21*self.dim**2))
        self.f_opt = np.inf
        self.x_opt = None

    def sample_population(self):
        z = np.random.normal(0, 1, size=(self.popsize, self.dim))
        A = np.linalg.cholesky(self.C)
        x = self.mean + self.sigma * z @ A.T
        return x

    def __call__(self, func):
        self.initialize(func)
        
        mu = self.popsize // 2  # Select only the best mu individuals
        c_sigma = (mu / self.dim) / ((self.dim + 4) + (mu / self.dim))
        c_c = (4 + mu / self.dim) / (self.dim + 4)
        c_1 = 2 / ((self.dim + 1.3)**2 + mu)
        c_mu = min(1 - c_1, 2 * (mu - 1 + 1/mu) / ((self.dim + 2)**2 + 2*mu))
        d_sigma = 1 + 2 * max(0, np.sqrt((mu - 1) / (self.dim + 1)) - 1) + c_sigma
        
        while self.eval_count < self.budget:
            # Sample population
            x = self.sample_population()
            
            # Clip individuals to respect boundaries
            lb = func.bounds.lb
            ub = func.bounds.ub
            x = np.clip(x, lb, ub)

            # Evaluate population
            fitness = np.array([func(xi) for xi in x])
            self.eval_count += self.popsize
            
            # Sort by fitness
            idx = np.argsort(fitness)
            fitness = fitness[idx]
            x = x[idx]
            
            # Update optimal solution
            if fitness[0] < self.f_opt:
                self.f_opt = fitness[0]
                self.x_opt = x[0]

            # Update mean
            xmean = np.mean(x[:mu], axis=0)
            self.ps = (1 - c_sigma) * self.ps + np.sqrt(c_sigma * (2 - c_sigma)) * (np.linalg.solve(np.linalg.cholesky(self.C), (xmean - self.mean) / self.sigma))
            
            hsig = (np.linalg.norm(self.ps) / np.sqrt(1 - (1 - c_sigma)**(2 * self.eval_count / self.popsize)) / self.chiN) < (1.4 + 2 / (self.dim + 1))
            self.pc = (1 - c_c) * self.pc + hsig * np.sqrt(c_c * (2 - c_c)) * (xmean - self.mean) / self.sigma
            self.mean = xmean

            # Update covariance matrix
            C_old = self.C.copy()
            self.C = (1 - c_1 - c_mu) * self.C + c_1 * np.outer(self.pc, self.pc) + c_mu * sum(np.outer((x[i] - self.mean) / self.sigma, (x[i] - self.mean) / self.sigma) for i in range(mu))

            # Adapt step size
            self.sigma *= np.exp((c_sigma / d_sigma) * (np.linalg.norm(self.ps) / self.chiN - 1))

            # Handle potential matrix ill-conditioning
            if np.any(np.diag(self.C) <= 0):
                self.C = C_old # Revert to old matrix and increase population size
                self.popsize = min(self.popsize + 2, self.max_popsize)

            # Restart strategy (Budget-aware)
            if self.eval_count > self.restart_factor * self.popsize * self.dim: 
               self.initialize(func)
               self.restart_count += 1
               self.popsize = max(self.min_popsize, int(self.popsize/2)) # Adapt population Size
               self.restart_factor *= 1.2
               

            if self.eval_count >= self.budget:
                break

        return self.f_opt, self.x_opt