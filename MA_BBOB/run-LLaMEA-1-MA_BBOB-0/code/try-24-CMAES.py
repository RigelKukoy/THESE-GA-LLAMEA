import numpy as np

class CMAES:
    def __init__(self, budget=10000, dim=10, popsize=None, cs=0.3, damps=None, c_cov=None, mu_factor=0.25, sigma0=0.5, memory_size=100, restart_trigger=1e-12, restart_strategy='archive'):
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
        self.c_cov = c_cov if c_cov is not None else (1 / self.mu) * (2 / ((self.dim + np.sqrt(2))**2))
        self.c_cov_mu = 0.25 + self.c_cov
        self.chiN = self.dim**0.5 * (1 - (1 / (4 * self.dim)) + (1 / (21 * self.dim**2)))
        self.f_opt = np.Inf
        self.x_opt = None
        self.archive = []
        self.archive_size = memory_size
        self.evals = 0
        self.restart_trigger = restart_trigger
        self.restart_strategy = restart_strategy
        self.damps_adapt = 1.0  # adaptive dampening factor

    def __call__(self, func):
        self.evals = 0
        while self.evals < self.budget:
            # Adaptive Population Size
            self.popsize = 4 + int(3 * np.log(self.dim))
            self.mu = int(self.popsize * 0.25)
            self.weights = np.log(self.mu + 0.5) - np.log(np.arange(1, self.mu + 1))
            self.weights /= np.sum(self.weights)
            
            # Sample population
            z = np.random.normal(0, 1, size=(self.popsize, self.dim))
            C_sqrt = np.linalg.cholesky(self.C)
            x = self.m + self.sigma * z @ C_sqrt.T
            
            # Clip to bounds.
            lb = func.bounds.lb
            ub = func.bounds.ub
            x = np.clip(x, lb, ub)
            
            f = np.array([func(xi) for xi in x])
            self.evals += self.popsize
            
            # Sort by fitness
            idx = np.argsort(f)
            x = x[idx]
            f = f[idx]
            
            if f[0] < self.f_opt:
                self.f_opt = f[0]
                self.x_opt = x[0]

            # Update mean
            m_old = self.m.copy()
            self.m = np.sum(self.weights[:, None] * x[:self.mu], axis=0)

            # Update evolution paths
            self.ps = (1 - self.cs) * self.ps + (np.sqrt(self.cs * (2 - self.cs) * self.weights @ self.weights)) * (self.m - m_old) @ np.linalg.inv(C_sqrt).T
            self.pc = (1 - self.c_cov) * self.pc + np.sqrt(self.c_cov * (2 - self.c_cov)) * (self.m - m_old) / self.sigma

            # Update covariance matrix
            hsigma = np.linalg.norm(self.ps) / np.sqrt(1 - (1 - self.cs)**(2 * (self.evals / self.popsize))) < (1.4 + (2 / (self.dim + 1))) * self.chiN
            
            dC = (self.c_cov_mu / self.mu) * (self.pc[:, None] @ self.pc[None, :])
            for i in range(self.mu):
                dC += (self.c_cov / self.mu) * ((x[i] - m_old)[:, None] @ (x[i] - m_old)[None, :]) / self.sigma**2
                
            self.C = (1 - self.c_cov - self.c_cov_mu) * self.C + dC
            
            # Ensure positive definiteness
            try:
                np.linalg.cholesky(self.C)
            except np.linalg.LinAlgError:
                self.C = np.eye(self.dim)

            # Update step size
            self.sigma *= np.exp((self.cs / (self.damps * self.damps_adapt)) * (np.linalg.norm(self.ps) / self.chiN - 1))
            
            # Adaptive Dampening
            if np.linalg.norm(self.ps) / self.chiN > 1:
                self.damps_adapt *= 0.8  # Reduce dampening when step size grows too fast
            else:
                self.damps_adapt = min(1.0, self.damps_adapt * 1.2) # Increase dampening when step size is shrinking


            # Archive for diversity (optional)
            for xi in x:
                self.archive.append(xi)
            if len(self.archive) > self.archive_size:
                self.archive = self.archive[-self.archive_size:]
                
            # Restart mechanism
            if self.sigma < self.restart_trigger:
                if self.restart_strategy == 'archive':
                    if self.archive:
                        # CMA-ES initialization using archive
                        archive_array = np.array(self.archive)
                        self.m = np.mean(archive_array, axis=0)
                        self.C = np.cov(archive_array, rowvar=False)
                        # Ensure covariance matrix is positive semi-definite
                        try:
                            np.linalg.cholesky(self.C)
                        except np.linalg.LinAlgError:
                            self.C = np.eye(self.dim)  # Reset to identity if singular
                    else:
                        self.m = np.zeros(self.dim) # Fallback: Reset to zero
                        self.C = np.eye(self.dim)
                else:
                    self.m = np.zeros(self.dim) # Default reset to zero
                self.sigma = sigma0
                self.ps = np.zeros(self.dim)
                self.pc = np.zeros(self.dim)
                self.damps_adapt = 1.0  # Reset adaptive dampening
        
        return self.f_opt, self.x_opt