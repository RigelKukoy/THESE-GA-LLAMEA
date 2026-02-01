import numpy as np

class HybridCMAES:
    def __init__(self, budget=10000, dim=10, mu_percentage=0.25, sigma0=0.5, cs=0.2, cmu=0.3, c_cov=0.1, de_rate=0.1):
        self.budget = budget
        self.dim = dim
        self.mu = int(mu_percentage * budget)  # Number of parents
        self.sigma = sigma0
        self.mean = None
        self.C = None  # Covariance matrix
        self.ps = None # Evolution path for sigma
        self.pc = None # Evolution path for mean
        self.cs = cs
        self.cmu = cmu
        self.c_cov = c_cov
        self.de_rate = de_rate
        self.restart_criterion = 1e-12
        self.population = None # current population

    def __call__(self, func):
        self.mean = np.random.uniform(func.bounds.lb, func.bounds.ub, size=self.dim)
        self.C = np.eye(self.dim)
        self.ps = np.zeros(self.dim)
        self.pc = np.zeros(self.dim)
        self.f_opt = np.Inf
        self.x_opt = None
        evals = 0
        self.population = np.random.uniform(func.bounds.lb, func.bounds.ub, size=(self.mu, self.dim))
        f_vals = np.array([func(xi) for xi in self.population])
        evals += self.mu
        
        while evals < self.budget:
            # 1. Sample offspring
            z = np.random.multivariate_normal(np.zeros(self.dim), self.C, size=self.mu)
            x = self.mean + self.sigma * z
            
            # Clip to bounds
            x = np.clip(x, func.bounds.lb, func.bounds.ub)
            
            f = np.array([func(xi) for xi in x])
            evals += self.mu

            # Incorporate population diversity using DE
            for i in range(self.mu):
                if np.random.rand() < self.de_rate:
                    idxs = np.random.choice(self.mu, 3, replace=False)
                    x_diff = self.population[idxs[1]] - self.population[idxs[2]]
                    x[i] = self.population[idxs[0]] + 0.5 * x_diff # DE mutation
                    x[i] = np.clip(x[i], func.bounds.lb, func.bounds.ub)
                    f[i] = func(x[i])
                    evals += 1


            # 2. Selection and Recombination
            idx = np.argsort(f)
            x_best = x[idx[:self.mu]]
            z_best = z[idx[:self.mu]]

            if np.min(f) < self.f_opt:
                self.f_opt = np.min(f)
                self.x_opt = x[idx[0]]

            # 3. Update mean
            mean_diff = np.mean(z_best, axis=0)
            self.pc = (1 - self.cs) * self.pc + np.sqrt(self.cs * (2 - self.cs)) * mean_diff
            self.mean = self.mean + self.cmu * self.sigma * self.pc

            # 4. Update covariance matrix
            self.ps = (1 - self.c_cov) * self.ps + np.sqrt(self.c_cov * (2 - self.c_cov)) * mean_diff
            self.C = (1- self.c_cov) * self.C + self.c_cov * (np.outer(self.ps, self.ps) - self.C)

            # 5. Update step size
            self.sigma *= np.exp( (self.cs / 2) * (np.linalg.norm(self.ps)**2 / self.dim - 1) )

            # 6. Update population
            self.population = x_best.copy()
            
            # Check for covariance matrix deterioration
            if np.linalg.det(self.C) < self.restart_criterion or not np.all(np.linalg.eigvals(self.C) > 0):
              self.C = np.eye(self.dim) # restart C
              self.sigma = 0.5 # Reset sigma 
              self.ps = np.zeros(self.dim)
              self.pc = np.zeros(self.dim)
              self.population = np.random.uniform(func.bounds.lb, func.bounds.ub, size=(self.mu, self.dim))


        return self.f_opt, self.x_opt