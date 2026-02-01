import numpy as np

class CMAES_Adaptive_Plus:
    def __init__(self, budget=10000, dim=10, popsize=None, cs=0.3, damps=None, c_cov_rank_one=None, c_cov_rank_mu=None, mu_factor=0.25, sigma0=0.5, memory_size=100, curvature_adaptation=True, active_update=True, dynamic_popsize=True):
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
        self.active_update = active_update
        self.dynamic_popsize = dynamic_popsize
        self.eigenvalues = np.ones(self.dim)  # Initialize eigenvalues
        self.B = np.eye(self.dim) # Initialize rotation matrix
        self.age = 0 # Age counter
        self.max_popsize = 2 * self.dim # Dynamic popsize maximum limit

    def update_decomposition(self):
        try:
            self.eigenvalues, self.B = np.linalg.eigh(self.C)  # Eigen decomposition
            self.eigenvalues = np.maximum(self.eigenvalues, 1e-16)  # Ensure eigenvalues are positive
        except np.linalg.LinAlgError:
            self.C = np.eye(self.dim)
            self.eigenvalues = np.ones(self.dim)
            self.B = np.eye(self.dim)

    def __call__(self, func):
        evals = 0
        while evals < self.budget:
            self.age += 1
            # Dynamic population size adjustment
            if self.dynamic_popsize:
                if self.age % 100 == 0:
                    if self.sigma < 0.01:  # Check for convergence
                        self.popsize = min(self.popsize + 1, self.max_popsize)  # Increase if converging
                    elif self.sigma > 1.0:
                        self.popsize = max(4 + int(3 * np.log(self.dim)), self.popsize - 1) # Decrease if diverging
                    self.mu = int(self.popsize * 0.25)
                    self.weights = np.log(self.mu + 0.5) - np.log(np.arange(1, self.mu + 1))
                    self.weights /= np.sum(self.weights)

            # Sample population
            z = np.random.normal(0, 1, size=(self.popsize, self.dim))

            # Use rotation matrix and eigenvalues directly
            x = self.m + self.sigma * (z @ self.B @ np.diag(self.eigenvalues**0.5))
            
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
            self.ps = (1 - self.cs) * self.ps + (np.sqrt(self.cs * (2 - self.cs) * self.weights @ self.weights)) * (self.m - m_old) @ self.B @ np.diag(self.eigenvalues**-0.5)
            self.pc = (1 - self.c_cov_rank_one) * self.pc + np.sqrt(self.c_cov_rank_one * (2 - self.c_cov_rank_one)) * (self.m - m_old) / self.sigma

            # Update covariance matrix
            hsigma = np.linalg.norm(self.ps) / np.sqrt(1 - (1 - self.cs)**(2 * (evals / self.popsize))) < (1.4 + (2 / (self.dim + 1))) * self.chiN
            
            dC_rank_one = self.c_cov_rank_one * (self.pc[:, None] @ self.pc[None, :])
            dC_rank_mu = self.c_cov_rank_mu * np.sum(self.weights[:, None, None] * ((x[:self.mu] - m_old)[:, :, None] @ (x[:self.mu] - m_old)[:, None, :]) / self.sigma**2, axis=0)
            
            self.C = (1 - self.c_cov_rank_one - self.c_cov_rank_mu) * self.C + dC_rank_one + dC_rank_mu

            # Active Update
            if self.active_update:
                w_minus = self.weights * -1
                w_minus = np.minimum(w_minus, 0)
                dC_rank_mu_minus = self.c_cov_rank_mu * np.sum(w_minus[:, None, None] * ((x[self.mu:] - m_old)[:, :, None] @ (x[self.mu:] - m_old)[:, None, :]) / self.sigma**2, axis=0)
                self.C += dC_rank_mu_minus


            # Ensure positive definiteness and update decomposition
            self.update_decomposition()

            # Adaptive step size based on curvature
            if self.curvature_adaptation:
                condition_number = np.max(self.eigenvalues) / np.min(self.eigenvalues)
                self.sigma *= np.exp((self.cs / self.damps) * (np.linalg.norm(self.ps) / self.chiN - 1) + 0.05 * (condition_number - 1)) # Condition number regularization

            else:
                 self.sigma *= np.exp((self.cs / self.damps) * (np.linalg.norm(self.ps) / self.chiN - 1))


            # Simplified archive handling: keep only best from each generation.
            self.archive.append((x[0], f[0], self.age)) # Store fitness as well and age
            if len(self.archive) > self.archive_size:
                # Sort by fitness primarily, then by age (older solutions are penalized)
                self.archive.sort(key=lambda item: (item[1], -item[2]))
                self.archive = self.archive[:self.archive_size]  # Keep best

        if self.archive:
            best_archived_solution = min(self.archive, key=lambda item: item[1])
            if best_archived_solution[1] < self.f_opt:
                 self.f_opt = best_archived_solution[1]
                 self.x_opt = best_archived_solution[0]
        
        return self.f_opt, self.x_opt