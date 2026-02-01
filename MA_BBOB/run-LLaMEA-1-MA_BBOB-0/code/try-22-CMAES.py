import numpy as np

class CMAES:
    def __init__(self, budget=10000, dim=10, popsize=None, cs=0.3, damps=None, c_cov_rank_one=None, c_cov_rank_mu=None, mu_factor=0.25, sigma0=0.5, memory_size=100, popsize_factor=2):
        self.budget = budget
        self.dim = dim
        self.popsize = popsize if popsize is not None else 4 + int(3 * np.log(self.dim))
        self.mu = int(self.popsize * mu_factor)
        self.weights = np.log(self.mu + 0.5) - np.log(np.arange(1, self.mu + 1))
        self.weights /= np.sum(self.weights)
        self.m = np.zeros(self.dim)
        self.sigma = sigma0
        self.ps = np.zeros(self.dim)
        self.pc = np.zeros(self.dim)
        self.C = np.eye(self.dim)
        self.cs = cs
        self.damps = damps if damps is not None else 1 + 2 * np.max((0, np.sqrt((self.mu / self.dim) - 1))) + self.cs
        self.c_cov_rank_one = c_cov_rank_one if c_cov_rank_one is not None else 0.15
        self.c_cov_rank_mu = c_cov_rank_mu if c_cov_rank_mu is not None else 0.15
        self.chiN = self.dim**0.5 * (1 - (1 / (4 * self.dim)) + (1 / (21 * self.dim**2)))
        self.f_opt = np.Inf
        self.x_opt = None
        self.archive = []
        self.archive_size = memory_size
        self.popsize_factor = popsize_factor
        self.min_popsize = 4 + int(3 * np.log(self.dim))
        self.max_popsize = int(self.dim * self.popsize_factor)

        self.stagnation_counter = 0
        self.stagnation_threshold = 20 # Number of iterations with minimal improvement before triggering restart
        self.last_f_opt = np.Inf

    def __call__(self, func):
        evals = 0
        while evals < self.budget:
            # Sample population
            z = np.random.normal(0, 1, size=(self.popsize, self.dim))
            C_sqrt = np.linalg.cholesky(self.C)
            x = self.m + self.sigma * z @ C_sqrt.T
            
            # Clip to bounds.
            lb = func.bounds.lb
            ub = func.bounds.ub
            x = np.clip(x, lb, ub)
            
            f = np.array([func(xi) for xi in x])
            evals += self.popsize
            
            # Sort by fitness
            idx = np.argsort(f)
            x = x[idx]
            f = f[idx]
            
            if f[0] < self.f_opt:
                self.f_opt = f[0]
                self.x_opt = x[0]
                self.stagnation_counter = 0 # Reset stagnation counter
            else:
                self.stagnation_counter +=1

            # Dynamic Population Size Adjustment
            if self.stagnation_counter > self.stagnation_threshold and self.popsize < self.max_popsize:
                self.popsize = min(self.popsize * 2, self.max_popsize)
                self.mu = int(self.popsize * 0.25)
                self.weights = np.log(self.mu + 0.5) - np.log(np.arange(1, self.mu + 1))
                self.weights /= np.sum(self.weights)
                self.stagnation_threshold *=1.2 #increase stagnation threshold
                self.stagnation_counter = 0 #reset counter
                
            if self.stagnation_counter > self.stagnation_threshold * 2 and self.popsize > self.min_popsize: # reduce population size if stagnation continues.
                self.popsize = max(self.popsize // 2, self.min_popsize)
                self.mu = int(self.popsize * 0.25)
                self.weights = np.log(self.mu + 0.5) - np.log(np.arange(1, self.mu + 1))
                self.weights /= np.sum(self.weights)
                self.stagnation_threshold = max(10, self.stagnation_threshold/1.2)
                self.stagnation_counter = 0 #reset counter


            # Update mean
            m_old = self.m.copy()
            self.m = np.sum(self.weights[:, None] * x[:self.mu], axis=0)

            # Update evolution paths
            self.ps = (1 - self.cs) * self.ps + (np.sqrt(self.cs * (2 - self.cs) * self.weights @ self.weights)) * (self.m - m_old) @ np.linalg.inv(C_sqrt).T
            self.pc = (1 - self.c_cov_rank_one) * self.pc + np.sqrt(self.c_cov_rank_one * (2 - self.c_cov_rank_one)) * (self.m - m_old) / self.sigma

            # Update covariance matrix
            hsigma = np.linalg.norm(self.ps) / np.sqrt(1 - (1 - self.cs)**(2 * (evals / self.popsize))) < (1.4 + (2 / (self.dim + 1))) * self.chiN
            
            dC_rank_one = self.c_cov_rank_one * (self.pc[:, None] @ self.pc[None, :])
            dC_rank_mu = self.c_cov_rank_mu * np.sum(self.weights[:, None, None] * ((x[:self.mu] - m_old)[:, :, None] @ (x[:self.mu] - m_old)[:, None, :]) / self.sigma**2, axis=0)
                
            self.C = (1 - self.c_cov_rank_one - self.c_cov_rank_mu) * self.C + dC_rank_one + dC_rank_mu
            
            # Ensure positive definiteness
            try:
                np.linalg.cholesky(self.C)
            except np.linalg.LinAlgError:
                self.C = np.eye(self.dim)

            # Update step size
            self.sigma *= np.exp((self.cs / self.damps) * (np.linalg.norm(self.ps) / self.chiN - 1))

            # Simplified archive handling: keep only best from each generation.
            self.archive.append(x[0])
            if len(self.archive) > self.archive_size:
                self.archive = self.archive[-self.archive_size:]

            # Restart mechanism
            if self.stagnation_counter > self.stagnation_threshold * 3: # More conservative stagnation check before restart
                self.m = np.random.uniform(func.bounds.lb, func.bounds.ub, size=self.dim) # Reset to a random point within bounds
                self.sigma = sigma0 # Reset step size
                self.ps = np.zeros(self.dim) # Reset evolution paths
                self.pc = np.zeros(self.dim)
                self.C = np.eye(self.dim)  # Reset covariance matrix
                self.stagnation_counter = 0  # Reset stagnation counter
                self.stagnation_threshold = 20

        
        return self.f_opt, self.x_opt