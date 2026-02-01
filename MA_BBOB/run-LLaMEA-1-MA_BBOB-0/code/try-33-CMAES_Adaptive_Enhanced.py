import numpy as np

class CMAES_Adaptive_Enhanced:
    def __init__(self, budget=10000, dim=10, popsize=None, cs=0.3, damps=None, c_cov_rank_one=None, c_cov_rank_mu=None, mu_factor=0.25, sigma0=0.5, memory_size=100, curvature_adaptation=True, active_adaptation=True, restart_strategy='IPOP', ipop_factor=2, condition_number_threshold=1e14, spectral_regularization_factor=1e-8):
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
        self.curvature_adaptation = curvature_adaptation
        self.active_adaptation = active_adaptation
        self.restart_strategy = restart_strategy
        self.ipop_factor = ipop_factor
        self.condition_number_threshold = condition_number_threshold
        self.spectral_regularization_factor = spectral_regularization_factor
        self.evals = 0
        self.generation = 0
        self.eigenvalues = np.ones(self.dim)  # Initialize eigenvalues
        self.B = np.eye(self.dim) # Initialize rotation matrix
        self.original_popsize = self.popsize
        self.best_history = []

    def update_decomposition(self):
        try:
            self.eigenvalues, self.B = np.linalg.eigh(self.C)  # Eigen decomposition
            self.eigenvalues = np.maximum(self.eigenvalues, self.spectral_regularization_factor)  # Ensure eigenvalues are positive and regularized
        except np.linalg.LinAlgError:
            self.C = np.eye(self.dim)
            self.eigenvalues = np.ones(self.dim)
            self.B = np.eye(self.dim)

    def update_covariance(self, x, m_old):
        # Rank-one update
        self.pc = (1 - self.c_cov_rank_one) * self.pc + np.sqrt(self.c_cov_rank_one * (2 - self.c_cov_rank_one)) * (self.m - m_old) / self.sigma
        dC_rank_one = self.c_cov_rank_one * (self.pc[:, None] @ self.pc[None, :])

        # Rank-mu update
        dC_rank_mu = self.c_cov_rank_mu * np.sum(self.weights[:, None, None] * ((x[:self.mu] - m_old)[:, :, None] @ (x[:self.mu] - m_old)[:, None, :]) / self.sigma**2, axis=0)

        # Active CMA
        if self.active_adaptation:
            weights_neg = np.minimum(0, self.weights)
            dC_active = self.c_cov_rank_mu * np.sum(weights_neg[:, None, None] * ((x[:self.mu] - m_old)[:, :, None] @ (x[:self.mu] - m_old)[:, None, :]) / self.sigma**2, axis=0)
            self.C = (1 - self.c_cov_rank_one - self.c_cov_rank_mu) * self.C + dC_rank_one + dC_rank_mu + dC_active
        else:
            self.C = (1 - self.c_cov_rank_one - self.c_cov_rank_mu) * self.C + dC_rank_one + dC_rank_mu

    def adapt_population_size(self):
        if self.restart_strategy == 'IPOP':
            self.popsize = int(self.original_popsize * (self.ipop_factor ** (len(self.best_history) // 100)))  # Adjust population size every 100 generations
            self.mu = int(self.popsize * 0.25)  # Adjust mu accordingly
            self.weights = np.log(self.mu + 0.5) - np.log(np.arange(1, self.mu + 1))
            self.weights /= np.sum(self.weights)

    def restart(self):
        if self.restart_strategy == 'IPOP':
            self.m = np.zeros(self.dim)  # Reset mean
            self.sigma = 0.5  # Reset step size
            self.ps = np.zeros(self.dim)  # Reset evolution paths
            self.pc = np.zeros(self.dim)
            self.C = np.eye(self.dim)  # Reset covariance matrix
            self.eigenvalues = np.ones(self.dim)
            self.B = np.eye(self.dim)
            self.popsize = self.original_popsize
            self.mu = int(self.popsize * 0.25)
            self.weights = np.log(self.mu + 0.5) - np.log(np.arange(1, self.mu + 1))
            self.weights /= np.sum(self.weights)

    def __call__(self, func):
        while self.evals < self.budget:
            # Sample population
            z = np.random.normal(0, 1, size=(self.popsize, self.dim))

            # Use rotation matrix and eigenvalues directly
            x = self.m + self.sigma * (z @ self.B @ np.diag(self.eigenvalues**0.5))
            
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
                self.best_history.append(self.f_opt)

            # Update mean
            m_old = self.m.copy()
            self.m = np.sum(self.weights[:, None] * x[:self.mu], axis=0)

            # Update evolution paths
            self.ps = (1 - self.cs) * self.ps + (np.sqrt(self.cs * (2 - self.cs) * self.weights @ self.weights)) * (self.m - m_old) @ self.B @ np.diag(self.eigenvalues**-0.5)

            # Update covariance matrix
            self.update_covariance(x, m_old)
            
            # Ensure positive definiteness and update decomposition
            self.update_decomposition()

            # Adaptive step size based on curvature
            if self.curvature_adaptation:
                condition_number = np.max(self.eigenvalues) / np.min(self.eigenvalues)
                self.sigma *= np.exp((self.cs / self.damps) * (np.linalg.norm(self.ps) / self.chiN - 1) + 0.05 * (condition_number - 1)) # Condition number regularization
                if condition_number > self.condition_number_threshold:
                    self.C += np.eye(self.dim) * self.spectral_regularization_factor

            else:
                 self.sigma *= np.exp((self.cs / self.damps) * (np.linalg.norm(self.ps) / self.chiN - 1))

            # Simplified archive handling: keep only best from each generation.
            self.archive.append((x[0], f[0])) # Store fitness as well
            if len(self.archive) > self.archive_size:
                self.archive.sort(key=lambda item: item[1]) # sort by fitness
                self.archive = self.archive[:self.archive_size]  # Keep best

            self.generation += 1

            # Restart mechanism
            if self.restart_strategy == 'IPOP' and self.evals < 0.95 * self.budget and len(self.best_history) > 50: # Don't restart close to budget end
                if np.std(self.best_history[-50:]) < 1e-9:
                    self.adapt_population_size() # adapt population size before restart
                    self.restart()  # Restart if stagnating

        if self.archive:
            best_archived_solution = min(self.archive, key=lambda item: item[1])
            if best_archived_solution[1] < self.f_opt:
                 self.f_opt = best_archived_solution[1]
                 self.x_opt = best_archived_solution[0]
        
        return self.f_opt, self.x_opt