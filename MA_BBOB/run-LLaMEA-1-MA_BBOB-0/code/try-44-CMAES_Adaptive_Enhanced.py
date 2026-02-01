import numpy as np

class CMAES_Adaptive_Enhanced:
    def __init__(self, budget=10000, dim=10, popsize=None, cs=0.3, damps=None, c_cov_rank_one=None, c_cov_rank_mu=None, mu_factor=0.25, sigma0=0.5, memory_size=100, curvature_adaptation=True, local_search_probability=0.1, local_search_radius=0.1, local_search_num_points=5, restart_trigger=0.95):
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
        self.eigenvalues = np.ones(self.dim)  # Initialize eigenvalues
        self.B = np.eye(self.dim) # Initialize rotation matrix
        self.evals = 0
        self.local_search_probability = local_search_probability
        self.local_search_radius = local_search_radius
        self.local_search_num_points = local_search_num_points
        self.restart_trigger = restart_trigger
        self.restart_criterion_met = False # Flag to indicate if restart criterion has been met

    def update_decomposition(self):
        try:
            self.eigenvalues, self.B = np.linalg.eigh(self.C)  # Eigen decomposition
            self.eigenvalues = np.maximum(self.eigenvalues, 1e-16)  # Ensure eigenvalues are positive
        except np.linalg.LinAlgError:
            self.C = np.eye(self.dim)
            self.eigenvalues = np.ones(self.dim)
            self.B = np.eye(self.dim)

    def local_search(self, func, x, radius=0.1, num_points=5):
        """Performs a local search around x."""
        best_f = func(x)
        best_x = x
        self.evals += 1

        for _ in range(num_points):
            x_new = x + np.random.uniform(-radius, radius, size=self.dim)
            lb = func.bounds.lb
            ub = func.bounds.ub
            x_new = np.clip(x_new, lb, ub)
            f_new = func(x_new)
            self.evals += 1

            if f_new < best_f:
                best_f = f_new
                best_x = x_new
        return best_f, best_x

    def spectral_preconditioning(self, func):
            """Initialize the covariance matrix using spectral analysis of initial samples."""
            num_samples = min(self.budget // 10, 100)  # Take a fraction of budget or up to 100 samples
            X = np.random.uniform(func.bounds.lb, func.bounds.ub, size=(num_samples, self.dim))
            F = np.array([func(x) for x in X])
            self.evals += num_samples

            # Center the data
            X_centered = X - np.mean(X, axis=0)

            # Calculate covariance matrix
            try:
                covariance_matrix = np.cov(X_centered.T)
                eigenvalues, eigenvectors = np.linalg.eigh(covariance_matrix)

                # Ensure positive definiteness by clipping eigenvalues
                eigenvalues = np.maximum(eigenvalues, 1e-6)

                # Normalize eigenvectors
                eigenvectors = eigenvectors / np.linalg.norm(eigenvectors, axis=0)

                # Update CMA-ES internal parameters
                self.C = covariance_matrix
                self.eigenvalues = eigenvalues
                self.B = eigenvectors
                self.update_decomposition()
            except np.linalg.LinAlgError:
                pass # Keep identity


    def __call__(self, func):
        self.evals = 0
        dynamic_popsize = self.popsize

        # Spectral preconditioning
        self.spectral_preconditioning(func)

        while self.evals < self.budget:
            # Dynamic population size adjustment
            if self.evals > self.budget * self.restart_trigger:
                dynamic_popsize = max(4, int(self.popsize * 0.5))  # Reduce population size later in the search
                self.restart_criterion_met = True
            else:
                dynamic_popsize = self.popsize
            
            self.mu = int(dynamic_popsize * 0.25)  #Adjust mu accordingly.
            self.weights = np.log(self.mu + 0.5) - np.log(np.arange(1, self.mu + 1))
            self.weights /= np.sum(self.weights)

            # Sample population
            z = np.random.normal(0, 1, size=(dynamic_popsize, self.dim))

            # Use rotation matrix and eigenvalues directly
            x = self.m + self.sigma * (z @ self.B @ np.diag(self.eigenvalues**0.5))
            
            # Clip to bounds.
            lb = func.bounds.lb
            ub = func.bounds.ub
            x = np.clip(x, lb, ub)
            
            f = np.array([func(xi) for xi in x])
            self.evals += dynamic_popsize
            
            # Sort by fitness
            idx = np.argsort(f)
            x = x[idx]
            f = f[idx]
            
            if f[0] < self.f_opt:
                self.f_opt = f[0]
                self.x_opt = x[0]

            # Local search refinement
            if np.random.rand() < self.local_search_probability:
                f_local, x_local = self.local_search(func, self.x_opt, radius=self.local_search_radius, num_points=self.local_search_num_points)
                if f_local < self.f_opt:
                    self.f_opt = f_local
                    self.x_opt = x_local


            # Update mean
            m_old = self.m.copy()
            self.m = np.sum(self.weights[:, None] * x[:self.mu], axis=0)

            # Update evolution paths
            self.ps = (1 - self.cs) * self.ps + (np.sqrt(self.cs * (2 - self.cs) * self.weights @ self.weights)) * (self.m - m_old) @ self.B @ np.diag(self.eigenvalues**-0.5)
            self.pc = (1 - self.c_cov_rank_one) * self.pc + np.sqrt(self.c_cov_rank_one * (2 - self.c_cov_rank_one)) * (self.m - m_old) / self.sigma

            # Update covariance matrix
            hsigma = np.linalg.norm(self.ps) / np.sqrt(1 - (1 - self.cs)**(2 * (self.evals / dynamic_popsize))) < (1.4 + (2 / (self.dim + 1))) * self.chiN
            
            dC_rank_one = self.c_cov_rank_one * (self.pc[:, None] @ self.pc[None, :])
            dC_rank_mu = self.c_cov_rank_mu * np.sum(self.weights[:, None, None] * ((x[:self.mu] - m_old)[:, :, None] @ (x[:self.mu] - m_old)[:, None, :]) / self.sigma**2, axis=0)
                
            self.C = (1 - self.c_cov_rank_one - self.c_cov_rank_mu) * self.C + dC_rank_one + dC_rank_mu
            
            # Ensure positive definiteness and update decomposition
            self.update_decomposition()

            # Adaptive step size based on curvature
            if self.curvature_adaptation:
                condition_number = np.max(self.eigenvalues) / np.min(self.eigenvalues)
                self.sigma *= np.exp((self.cs / self.damps) * (np.linalg.norm(self.ps) / self.chiN - 1) + 0.05 * (condition_number - 1)) # Condition number regularization

            else:
                 self.sigma *= np.exp((self.cs / self.damps) * (np.linalg.norm(self.ps) / self.chiN - 1))


            # Enhanced archive handling: keep a diverse set of solutions
            if len(self.archive) < self.archive_size:
                self.archive.append((x[0].copy(), f[0]))
            else:
                # Replace the worst element in the archive with the current best, if it's better
                worst_index = np.argmax([item[1] for item in self.archive])
                if f[0] < self.archive[worst_index][1]:
                    self.archive[worst_index] = (x[0].copy(), f[0])

            # Soft restart mechanism
            if self.restart_criterion_met:
                self.m = np.random.uniform(func.bounds.lb, func.bounds.ub, size=self.dim)
                self.sigma = sigma0  # Resetting sigma might be aggressive; consider a smaller adjustment.
                self.ps = np.zeros(self.dim)
                self.pc = np.zeros(self.dim)
                self.C = np.eye(self.dim)
                self.update_decomposition()
                self.restart_criterion_met = False  # Reset the flag for the next potential restart

        if self.archive:
            best_archived_solution = min(self.archive, key=lambda item: item[1])
            if best_archived_solution[1] < self.f_opt:
                 self.f_opt = best_archived_solution[1]
                 self.x_opt = best_archived_solution[0]
        
        return self.f_opt, self.x_opt