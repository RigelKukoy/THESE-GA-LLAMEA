import numpy as np

class AdaptiveCMAES:
    def __init__(self, budget=10000, dim=10, mu_ratio=0.25, cs=0.3, dsigma=0.2, ccov=0.3, initial_sigma=0.5, active=True, ols_frequency=10):
        self.budget = budget
        self.dim = dim
        self.mu_ratio = mu_ratio
        self.cs = cs
        self.dsigma = dsigma
        self.ccov = ccov
        self.lb = -5.0
        self.ub = 5.0
        self.initial_sigma = initial_sigma
        self.active = active
        self.ols_frequency = ols_frequency
        self.population_factor = 4  # Factor for dynamic population size

    def __call__(self, func):
        # Initialize variables
        xmean = np.random.uniform(self.lb, self.ub, size=self.dim)
        sigma = self.initial_sigma

        lambda_ = int(4 + np.floor(3 * np.log(self.dim)) * self.population_factor)  # Dynamic population size
        mu = int(lambda_ * self.mu_ratio)
        
        weights = np.log(mu + 1/2) - np.log(np.arange(1, mu + 1))
        weights = weights / np.sum(weights)
        
        mueff = np.sum(weights)**2 / np.sum(weights**2)
        
        C = np.eye(self.dim)
        pc = np.zeros(self.dim)
        ps = np.zeros(self.dim)
        
        chiN = np.sqrt(self.dim) * (1 - (1/(4*self.dim)) + 1/(12*self.dim**2))
        
        # Parameters for adaption
        cs = self.cs
        damps = 1 + self.dsigma * max(0, np.sqrt((mueff-1)/(self.dim+1))-1) + cs
        ccov = self.ccov
        c1 = ccov / ((self.dim+1.3)**2 + mueff)
        cmu = min(1-c1, ccov * (mueff-2+1/mueff) / ((self.dim+2.0)**2 + mueff))

        # Active CMA parameters
        c1a = c1
        cmua = cmu
        if self.active:
            c1a = c1 / 10
            cmua = min(1 - c1a, cmu * (mueff - 2 + 1/mueff) / ((self.dim + 2)**2 + mueff))
        
        f_opt = np.Inf
        x_opt = None
        evals = 0

        restart_iter = 0
        max_restarts = 5

        stagnation_counter = 0
        stagnation_threshold = 50

        condition_number_threshold = 1e6

        ols_counter = 0

        while evals < self.budget:
            # Generate and evaluate offspring
            z = np.random.normal(0, 1, size=(self.dim, lambda_))
            y = np.dot(np.linalg.cholesky(C), z)
            x = xmean.reshape(-1, 1) + sigma * y
            x = np.clip(x, self.lb, self.ub)
            
            f = np.array([func(x[:,i]) if evals + i < self.budget else np.inf for i in range(lambda_)])
            evals += lambda_
            
            # Sort by fitness
            idx = np.argsort(f)
            f = f[idx]
            x = x[:, idx]
            
            # Update optimal solution
            if f[0] < f_opt:
                f_opt = f[0]
                x_opt = x[:, 0].copy()
                stagnation_counter = 0
            else:
                stagnation_counter += 1

            # Update distribution parameters
            xmean_new = np.sum(x[:, :mu] * weights, axis=1)
            
            ps = (1-cs) * ps + np.sqrt(cs*(2-cs)*mueff) * np.dot(np.linalg.inv(np.linalg.cholesky(C)), (xmean_new - xmean)) / sigma
            hsig = np.linalg.norm(ps)/np.sqrt(1-(1-cs)**(2*evals/lambda_))/chiN < 1.4 + 2/(self.dim+1)
            pc = (1-ccov) * pc + hsig * np.sqrt(ccov*(2-ccov)*mueff) * (xmean_new - xmean) / sigma
            
            C = (1-c1-cmu) * C + c1 * (np.outer(pc, pc) + (1-hsig) * ccov*(2-ccov) * C)
            
            # More robust covariance update, selective update
            for i in range(mu):
                y = (x[:, i] - xmean) / sigma
                C += cmu * weights[i] * np.outer(y, y)

            # Active covariance update
            if self.active:
                for i in range(mu, min(lambda_, 2*mu)):
                    w = weights[-1] / 10
                    y = (x[:, i] - xmean) / sigma
                    C -= cmua * w * np.outer(y, y)

            sigma = sigma * np.exp((cs/damps) * (np.linalg.norm(ps)/chiN - 1))
            
            xmean = xmean_new

            # Orthogonal Subspace Learning (OLS)
            ols_counter += 1
            if ols_counter >= self.ols_frequency:
                ols_counter = 0
                # 1. Determine the search subspace (e.g., using the best solutions)
                 subspace_dim = min(mu, self.dim // 2)  # Limit the subspace dimension
                eigenvalues, eigenvectors = np.linalg.eigh(C)
                # Sort eigenvalues and eigenvectors
                sorted_indices = np.argsort(eigenvalues)[::-1]
                eigenvalues = eigenvalues[sorted_indices]
                eigenvectors = eigenvectors[:, sorted_indices]
            
                # Select the eigenvectors corresponding to the largest eigenvalues
                search_directions = eigenvectors[:, :subspace_dim]
            
                # 2. Sample new solutions within the subspace
                num_samples = lambda_ // 2  # Reduce the number of samples
                z_ols = np.random.normal(0, 1, size=(subspace_dim, num_samples))
                y_ols = np.dot(search_directions, z_ols)
                x_ols = xmean.reshape(-1, 1) + sigma * y_ols
                x_ols = np.clip(x_ols, self.lb, self.ub)
            
                # 3. Evaluate the new solutions
                f_ols = np.array([func(x_ols[:, i]) if evals + i < self.budget else np.inf for i in range(num_samples)])
                evals += num_samples
            
                # 4. Integrate the best OLS solution into the population
                idx_ols_best = np.argmin(f_ols)
                if f_ols[idx_ols_best] < f[-1]:  # Only replace if better than the worst
                    x[:, -1] = x_ols[:, idx_ols_best]
                    f[-1] = f_ols[idx_ols_best]

            # Repair covariance matrix (ensure positive definiteness)
            C = np.triu(C) + np.transpose(np.triu(C,1))
            try:
                np.linalg.cholesky(C)
            except np.linalg.LinAlgError:
                C = C + np.eye(self.dim) * 1e-6
            
            # Adaptive Restart mechanism
            if stagnation_counter > stagnation_threshold:
                eigenvalues = np.linalg.eigvalsh(C)
                condition_number = np.max(eigenvalues) / np.min(eigenvalues)

                if condition_number > condition_number_threshold:
                    restart_iter += 1
                    
                    # Focused Restart: perturb around the current best
                    xmean = x_opt + np.random.normal(0, 0.1 * (self.ub - self.lb), size=self.dim)
                    xmean = np.clip(xmean, self.lb, self.ub)
                    
                    sigma = self.initial_sigma
                    C = np.eye(self.dim)
                    pc = np.zeros(self.dim)
                    ps = np.zeros(self.dim)
                    stagnation_counter = 0

                    condition_number_threshold *= 0.8

                    if restart_iter > max_restarts:
                        break
                else:
                     stagnation_counter = 0

            if np.any(np.isnan(C)):
                C = np.eye(self.dim)
                sigma = self.initial_sigma
                pc = np.zeros(self.dim)
                ps = np.zeros(self.dim)
                stagnation_counter = 0
                    
        return f_opt, x_opt