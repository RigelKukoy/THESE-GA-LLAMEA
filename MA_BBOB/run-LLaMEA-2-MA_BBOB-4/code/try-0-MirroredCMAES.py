import numpy as np

class MirroredCMAES:
    def __init__(self, budget=10000, dim=10, pop_size=None, sigma0=0.5, mu_ratio=0.25, mirrored_sampling=True):
        self.budget = budget
        self.dim = dim
        self.lb = -5.0
        self.ub = 5.0
        self.sigma0 = sigma0
        self.mirrored_sampling = mirrored_sampling

        if pop_size is None:
            self.pop_size = 4 + int(3 * np.log(self.dim))
        else:
            self.pop_size = pop_size
        self.mu = int(self.pop_size * mu_ratio)
        self.weights = np.log(self.mu + 0.5) - np.log(np.arange(1, self.mu + 1))
        self.weights /= np.sum(self.weights)
        self.mueff = np.sum(self.weights)**2 / np.sum(self.weights**2)
        self.c_cov = 2 / (self.dim + np.sqrt(2))**2 + self.mueff / self.pop_size  # Simplified c_cov
        self.c_sigma = (self.mueff + 2) / (self.dim + self.mueff + 5)
        self.d_sigma = 1 + 2 * max(0, np.sqrt((self.mueff - 1) / (self.dim + 1)) - 1) + self.c_sigma
        self.chiN = self.dim**0.5 * (1 - 1/(4*self.dim) + 1/(21*self.dim**2))

    def __call__(self, func):
        mean = np.random.uniform(self.lb, self.ub, size=self.dim)
        sigma = self.sigma0
        C = np.eye(self.dim)
        p_sigma = np.zeros(self.dim)
        f_opt = np.Inf
        x_opt = None
        eval_count = 0

        while eval_count < self.budget:
            # Generate population
            Z = np.random.multivariate_normal(np.zeros(self.dim), C, size=self.pop_size // (1 + self.mirrored_sampling)) if self.mirrored_sampling else np.random.multivariate_normal(np.zeros(self.dim), C, size=self.pop_size)
            
            if self.mirrored_sampling:
                Z = np.concatenate([Z, -Z])
            
            X = mean + sigma * Z

            # Repair individuals outside the bounds
            X = np.clip(X, self.lb, self.ub)
            
            F = np.array([func(x) for x in X])
            eval_count += self.pop_size
            if eval_count > self.budget:
                eval_count = self.budget
                F = F[:self.budget - (eval_count - self.pop_size)]
                X = X[:self.budget - (eval_count - self.pop_size)]
                
            if np.min(F) < f_opt:
                f_opt = np.min(F)
                x_opt = X[np.argmin(F)].copy()

            # Selection and recombination
            idx = np.argsort(F)
            X_mu = X[idx[:self.mu]]
            Z_mu = Z[idx[:self.mu]]
            mean_new = np.sum(self.weights[:, None] * X_mu, axis=0)
            z_mean = np.sum(self.weights[:, None] * Z_mu, axis=0)

            # Update evolution paths
            p_sigma = (1 - self.c_sigma) * p_sigma + np.sqrt(self.c_sigma * (2 - self.c_sigma) * self.mueff) * z_mean

            # Simplified Covariance matrix adaptation
            C = (1 - self.c_cov) * C + self.c_cov * (z_mean[:, None] @ z_mean[None, :])  # Simplified update

            # Update step size
            sigma *= np.exp((self.c_sigma / self.d_sigma) * (np.linalg.norm(p_sigma) / self.chiN - 1))
            sigma = np.clip(sigma, 1e-6, 1)
            
            mean = mean_new

        return f_opt, x_opt