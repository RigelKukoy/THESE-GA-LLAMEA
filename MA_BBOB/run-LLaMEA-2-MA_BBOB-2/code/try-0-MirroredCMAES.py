import numpy as np

class MirroredCMAES:
    def __init__(self, budget=10000, dim=10, mu_factor=0.25, cs=0.08, dsigma=0.2, c_cov=2/3, c_cov_mu=None):
        self.budget = budget
        self.dim = dim
        self.mu = int(budget * mu_factor) if int(budget * mu_factor) > 0 else 1
        self.lambda_ = int(4 + 3 * np.log(self.dim))
        self.mu = min(self.mu, self.lambda_)
        self.weights = np.log(self.mu + 1/2) - np.log(np.arange(1, self.mu + 1))
        self.weights = self.weights / np.sum(self.weights)
        self.mueff = np.sum(self.weights)**2 / np.sum(self.weights**2)
        self.cs = cs
        self.dsigma = dsigma
        self.c_cov = c_cov
        self.c_cov_mu = min(1 - self.c_cov, (self.mueff / (self.dim + 13)) * self.c_cov) if c_cov_mu is None else c_cov_mu
        self.chiN = self.dim**0.5 * (1 - (1/(4*self.dim)) + 1/(21*self.dim**2))


    def __call__(self, func):
        self.f_opt = np.Inf
        self.x_opt = None
        
        # Initialize variables
        mean = np.random.uniform(func.bounds.lb, func.bounds.ub, size=self.dim)
        sigma = 0.5  # Initial step size
        C = np.eye(self.dim) # Covariance matrix
        p_sigma = np.zeros(self.dim) # Evolution path for sigma
        p_c = np.zeros(self.dim) # Evolution path for C
        
        used_budget = 0
        
        while used_budget < self.budget:
            # Sample lambda candidate solutions
            z = np.random.normal(0, 1, size=(self.dim, self.lambda_))
            A = np.linalg.cholesky(C)
            x = mean[:, np.newaxis] + sigma * A @ z
            x_mirrored = mean[:, np.newaxis] - sigma * A @ z  # Mirrored samples
            
            x = np.clip(x, func.bounds.lb, func.bounds.ub)
            x_mirrored = np.clip(x_mirrored, func.bounds.lb, func.bounds.ub)
            
            # Evaluate solutions
            f = np.array([func(xi) for xi in x.T])
            f_mirrored = np.array([func(xi) for xi in x_mirrored.T])
            used_budget += 2 * self.lambda_
            
            # Combine original and mirrored samples
            x_combined = np.concatenate((x, x_mirrored), axis=1)
            f_combined = np.concatenate((f, f_mirrored))
            
            # Sort solutions
            idx = np.argsort(f_combined)
            x_sorted = x_combined[:, idx]
            f_sorted = f_combined[idx]
            
            # Update optimal solution
            if f_sorted[0] < self.f_opt:
                self.f_opt = f_sorted[0]
                self.x_opt = x_sorted[:, 0]
                
            # Update mean
            mean_diff = x_sorted[:, :self.mu] - mean[:, np.newaxis]
            mean = np.sum(self.weights[np.newaxis, :] * x_sorted[:, :self.mu], axis=1)

            # Update evolution paths
            B = A @ z[:, :self.mu]
            mean_diff_weighted = np.sum(self.weights[np.newaxis, :] * B, axis=1) / sigma
            p_sigma = (1 - self.cs) * p_sigma + np.sqrt(self.cs * (2 - self.cs) * self.mueff) * (A @ mean_diff_weighted)
            
            hsig = (np.linalg.norm(p_sigma) / np.sqrt(1 - (1 - self.cs)**(2*(used_budget/(2*self.lambda_)))) / self.chiN) < (1.4 + 2/(self.dim + 1))
            p_c = (1 - self.c_cov) * p_c + hsig * np.sqrt(self.c_cov * (2 - self.c_cov) * self.mueff) * (mean_diff / sigma)
            
            # Update covariance matrix
            C = (1 - self.c_cov - self.c_cov_mu) * C + self.c_cov * (p_c[:, np.newaxis] @ p_c[np.newaxis, :]) + self.c_cov_mu * (B @ np.diag(self.weights) @ B.T) / sigma**2
            
            # Update step size
            sigma = sigma * np.exp((self.cs/self.dsigma) * (np.linalg.norm(p_sigma)/self.chiN - 1))

            C = np.triu(C) + np.triu(C, 1).T # enforce symmetry
            C = C / np.linalg.norm(C, ord='fro') * self.dim # normalize
            
            try:
                np.linalg.cholesky(C) # check for positive definiteness
            except np.linalg.LinAlgError:
                C = np.eye(self.dim) # restart covariance matrix

            if used_budget > self.budget:
                used_budget = self.budget

        
        return self.f_opt, self.x_opt