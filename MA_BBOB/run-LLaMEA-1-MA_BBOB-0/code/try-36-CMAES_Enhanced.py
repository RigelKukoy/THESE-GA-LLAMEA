import numpy as np

class CMAES_Enhanced:
    def __init__(self, budget=10000, dim=10, popsize=None, cs=0.3, damps=None, c_cov=None, mu_factor=0.25, sigma0=0.5, memory_size=100, restart_trigger=1e-12, orthogonal_sampling=True, spectral_init=True, moving_average_recombination=True):
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
        self.orthogonal_sampling = orthogonal_sampling
        self.successful_sigma_updates = []
        self.spectral_init = spectral_init
        self.moving_average_recombination = moving_average_recombination

    def __call__(self, func):
        self.evals = 0
        self.successful_sigma_updates = []
        
        # Spectral Initialization
        if self.spectral_init:
            initial_samples = np.random.uniform(func.bounds.lb, func.bounds.ub, size=(self.popsize, self.dim))
            initial_fitness = np.array([func(xi) for xi in initial_samples])
            covariance_matrix = np.cov(initial_samples.T)
            try:
                eigenvalues, eigenvectors = np.linalg.eigh(covariance_matrix)
                self.C = eigenvectors @ np.diag(np.maximum(eigenvalues, 1e-6)) @ eigenvectors.T # Ensure positive definite
                self.sigma = np.std(initial_fitness) # Initialize sigma based on initial fitness variance
            except np.linalg.LinAlgError:
                pass # Keep identity if spectral init fails

        while self.evals < self.budget:
            # Adaptive Population Size
            self.popsize = 4 + int(3 * np.log(self.dim))
            self.mu = int(self.popsize * 0.25)
            self.weights = np.log(self.mu + 0.5) - np.log(np.arange(1, self.mu + 1))
            self.weights /= np.sum(self.weights)
            
            # Sample population
            if self.orthogonal_sampling:
                z = self._pairwise_orthogonal_sampling(self.popsize, self.dim)
            else:
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
            if self.moving_average_recombination:
                 self.m = 0.9 * self.m + 0.1 * np.sum(self.weights[:, None] * x[:self.mu], axis=0)
            else:
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

            # Adaptive step size control
            old_sigma = self.sigma
            self.sigma *= np.exp((self.cs / self.damps) * (np.linalg.norm(self.ps) / self.chiN - 1))
            
            # Store successful sigma updates
            if self.f_opt < np.inf: # Check if a better solution has been found
                self.successful_sigma_updates.append((old_sigma, self.sigma))

            # Weighted average of successful step sizes
            if len(self.successful_sigma_updates) > 5:
                 weights = np.exp(-np.arange(len(self.successful_sigma_updates)) / 2.0)
                 weights /= np.sum(weights)
                 weighted_sigma = np.sum([w * s[1] for w, s in zip(weights[::-1], self.successful_sigma_updates[-len(weights):])])
                 self.sigma = weighted_sigma


            # Archive for diversity (optional)
            for xi in x:
                self.archive.append(xi)
            
            # Dynamic archive size
            self.archive_size = min(int(self.budget / 10), 100 + int(self.evals / self.budget * 200))
            if len(self.archive) > self.archive_size:
                self.archive = self.archive[-self.archive_size:]
                
            # Restart mechanism
            if self.sigma < self.restart_trigger:
                self.m = np.mean(np.array(self.archive), axis=0) if self.archive else np.zeros(self.dim)
                self.sigma = sigma0
                self.ps = np.zeros(self.dim)
                self.pc = np.zeros(self.dim)
                self.C = np.eye(self.dim)
        
        return self.f_opt, self.x_opt

    def _pairwise_orthogonal_sampling(self, popsize, dim):
        z = np.random.normal(0, 1, size=(popsize, dim))
        
        for i in range(0, popsize, 2):
            if i + 1 < popsize:
                # Create two orthogonal vectors
                v1 = z[i]
                v2 = z[i+1] - (np.dot(z[i+1], z[i]) / np.dot(z[i], z[i])) * z[i]
                
                # Normalize
                v1 /= np.linalg.norm(v1)
                v2 /= np.linalg.norm(v2)
                
                z[i] = v1
                z[i+1] = v2
        return z