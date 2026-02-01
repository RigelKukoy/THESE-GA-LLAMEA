import numpy as np

class CMAES:
    def __init__(self, budget=10000, dim=10, popsize=None, cs=0.3, damps=None, c_cov=None, mu_factor=0.25, sigma0=0.5, memory_size=100, active_update=True):
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
        self.active_update = active_update # Add active update flag
        self.tolXup = 1e9 * np.finfo(float).eps * np.ones(self.dim)

    def orthogonal_sampling(self, popsize):
        z = np.random.normal(0, 1, size=(popsize // 2, self.dim))
        z = np.vstack((z, -z))  # Orthogonal sampling
        return z

    def __call__(self, func):
        evals = 0
        while evals < self.budget:
            # Dynamic population size
            if evals > self.budget * 0.5:
                self.popsize = max(4 + int(2 * np.log(self.dim)), 4)  # Reduce popsize later

            # Sample population
            z = self.orthogonal_sampling(self.popsize) # Orthogonal sampling
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

            # Update mean
            m_old = self.m.copy()
            self.m = np.sum(self.weights[:, None] * x[:self.mu], axis=0)

            # Update evolution paths
            self.ps = (1 - self.cs) * self.ps + (np.sqrt(self.cs * (2 - self.cs) * self.weights @ self.weights)) * (self.m - m_old) @ np.linalg.inv(C_sqrt).T
            self.pc = (1 - self.c_cov) * self.pc + np.sqrt(self.c_cov * (2 - self.c_cov)) * (self.m - m_old) / self.sigma

            # Update covariance matrix
            hsigma = np.linalg.norm(self.ps) / np.sqrt(1 - (1 - self.cs)**(2 * (evals / self.popsize))) < (1.4 + (2 / (self.dim + 1))) * self.chiN
            
            dC = (self.c_cov_mu / self.mu) * (self.pc[:, None] @ self.pc[None, :])
            
            if self.active_update: # Active CMA update
                neg_weights = np.minimum(0, self.weights)
                w_sum = np.sum(self.weights**2)
                w_neg_sum = np.sum(neg_weights**2)
                alpha_mu = 2
                min_alpha = alpha_mu + self.dim / w_sum
                
                for i in range(self.mu):
                    dC += (self.c_cov / self.mu) * ((x[i] - m_old)[:, None] @ (x[i] - m_old)[None, :]) / self.sigma**2
                
                alpha_neg = min(alpha_mu, (self.dim**2) / (w_neg_sum * (alpha_mu - 1)**2 + 2 * self.dim * w_neg_sum))
                for i in range(self.popsize):
                    if i >= self.mu:
                        dC += (alpha_neg * self.c_cov / self.mu) * ((x[i] - m_old)[:, None] @ (x[i] - m_old)[None, :]) / self.sigma**2
            else: 
                for i in range(self.mu):
                    dC += (self.c_cov / self.mu) * ((x[i] - m_old)[:, None] @ (x[i] - m_old)[None, :]) / self.sigma**2
                
            self.C = (1 - self.c_cov - self.c_cov_mu) * self.C + dC
            
            # Ensure positive definiteness
            try:
                np.linalg.cholesky(self.C)
            except np.linalg.LinAlgError:
                self.C = np.eye(self.dim)

            # Update step size
            self.sigma *= np.exp((self.cs / self.damps) * (np.linalg.norm(self.ps) / self.chiN - 1))

            # Archive for diversity (optional)
            for xi in x:
                self.archive.append(xi)
            if len(self.archive) > self.archive_size:
                self.archive = self.archive[-self.archive_size:]
        
        return self.f_opt, self.x_opt