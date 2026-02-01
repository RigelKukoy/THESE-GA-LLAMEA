import numpy as np

class CMAES:
    def __init__(self, budget=10000, dim=10, popsize=None, sigma0=0.5):
        self.budget = budget
        self.dim = dim
        self.popsize = popsize if popsize is not None else 4 + int(3 * np.log(self.dim))
        self.sigma0 = sigma0
        self.lb = -5.0
        self.ub = 5.0

    def __call__(self, func):
        # Initialize variables
        mu = np.random.uniform(self.lb, self.ub, size=self.dim)
        sigma = self.sigma0
        C = np.eye(self.dim)  # Covariance matrix
        
        # Parameters for CMA-ES
        c_s = 0.3  # Learning rate for step size
        d_s = 1.0 + 2 * max(0, np.sqrt((self.mu_eff - 1) / (self.dim + 1)) - 1)
        c_mu = 0.3 # Learning rate for mean
        c_1 = 0.3# Learning rate for covariance rank-one update
        
        alpha_mu = 2
        c_mu = min(1-c_1, alpha_mu * (self.mu_eff - 2 + 1/self.mu_eff) / ((self.dim+2)**2 + alpha_mu * self.mu_eff / 2))
        
        c_c = 0.3
        
        mu_weights = np.log(self.popsize+1) - np.log(np.arange(1, self.popsize+1))
        mu_weights = mu_weights / np.sum(mu_weights)
        self.mu_eff = np.sum(mu_weights)**2 / np.sum(mu_weights**2)
        
        c_1 = 2 / ((self.dim+1.3)**2 + self.mu_eff)
        c_mu = min(1-c_1, 2 * (self.mu_eff - 2 + 1/self.mu_eff) / ((self.dim+2)**2 + 2 * self.mu_eff/2))
        c_c = (4 + self.mu_eff/self.dim) / (self.dim + 4 + 2*self.mu_eff/self.dim)

        p_s = np.zeros(self.dim)
        p_c = np.zeros(self.dim)
        
        f_opt = np.Inf
        x_opt = None
        eval_count = 0

        while eval_count < self.budget:
            # Generate population
            z = np.random.normal(0, 1, size=(self.dim, self.popsize))
            
            try:
                C_sqrt = np.linalg.cholesky(C)
            except np.linalg.LinAlgError:
                C = C + 1e-6 * np.eye(self.dim)
                C_sqrt = np.linalg.cholesky(C)
            
            x = mu[:, np.newaxis] + sigma * np.dot(C_sqrt, z)
            
            # Repair individuals outside the bounds
            x = np.clip(x, self.lb, self.ub)

            # Evaluate population
            f = np.array([func(x[:, i]) for i in range(self.popsize)])
            eval_count += self.popsize

            # Sort by fitness
            idx = np.argsort(f)
            x = x[:, idx]
            f = f[idx]

            # Update the mean
            mu_old = mu
            mu = np.dot(x[:, :self.popsize], mu_weights)

            # Update evolution path for step size
            p_s = (1 - c_s) * p_s + np.sqrt(c_s * (2 - c_s) * self.mu_eff) * np.dot(C_sqrt, (mu - mu_old)) / sigma
            
            # Update step size
            sigma = sigma * np.exp((c_s / d_s) * (np.linalg.norm(p_s) / np.sqrt(self.dim) - 1))
            
            # Update evolution path for covariance matrix
            p_c = (1 - c_c) * p_c + np.sqrt(c_c * (2 - c_c) * self.mu_eff) * (mu - mu_old) / sigma
            
            # Update covariance matrix
            C = (1 - c_1 - c_mu) * C + c_1 * np.outer(p_c, p_c)

            for i in range(self.popsize):
                C = C + c_mu * mu_weights[i] * np.outer((x[:,i] - mu_old) / sigma, (x[:,i] - mu_old) / sigma)
            
            # Store best solution
            if f[0] < f_opt:
                f_opt = f[0]
                x_opt = x[:, 0].copy()

        return f_opt, x_opt