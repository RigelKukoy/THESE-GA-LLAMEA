import numpy as np

class OrthogonalCMAES:
    def __init__(self, budget=10000, dim=10, pop_size=None, sigma0=0.5, cs=0.3, c_cov=0.1, mu_ratio=0.25, restart_factor=2.0, success_history=10):
        self.budget = budget
        self.dim = dim
        self.lb = -5.0
        self.ub = 5.0
        self.sigma0 = sigma0
        self.cs = cs
        self.c_cov = c_cov
        self.restart_factor = restart_factor
        if pop_size is None:
            self.pop_size = 4 + int(3 * np.log(self.dim))
        else:
            self.pop_size = pop_size
        self.mu = int(self.pop_size * mu_ratio)
        self.weights = np.log(self.mu + 0.5) - np.log(np.arange(1, self.mu + 1))
        self.weights /= np.sum(self.weights)
        self.mueff = np.sum(self.weights)**2 / np.sum(self.weights**2)
        self.c_sigma = (self.mueff + 2) / (self.dim + self.mueff + 5)
        self.d_sigma = 1 + 2 * max(0, np.sqrt((self.mueff - 1) / (self.dim + 1)) - 1) + self.c_sigma
        self.chiN = self.dim**0.5 * (1 - 1/(4*self.dim) + 1/(21*self.dim**2))
        self.success_history = success_history
        self.success_rate = 0.5
        self.successes = []


    def __call__(self, func):
        mean = np.random.uniform(self.lb, self.ub, size=self.dim)
        sigma = self.sigma0
        C = np.eye(self.dim)
        p_sigma = np.zeros(self.dim)
        p_c = np.zeros(self.dim)
        f_opt = np.Inf
        x_opt = None
        eval_count = 0
        restarts = 0

        while eval_count < self.budget:
            # Generate orthogonal matrix
            H = np.random.normal(size=(self.pop_size, self.dim))
            Q, R = np.linalg.qr(H)

            # Generate population
            Z = Q @ np.random.multivariate_normal(np.zeros(self.dim), C, size=self.dim).T
            Z = Z.T # Back to pop_size x dim
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
            p_c = (1 - self.cs) * p_c + np.sqrt(self.cs * (2 - self.cs) * self.mueff) * (mean_new - mean) / sigma

            # Update covariance matrix
            C = (1 - self.c_cov) * C + self.c_cov * (p_c[:, None] @ p_c[None, :]) + self.c_cov * (1 - self.c_cov) * self.c_sigma * (2 - self.c_sigma) * C

            # Adapt step size based on success rate
            success = np.mean(F < np.median(F))
            self.successes.append(success)
            if len(self.successes) > self.success_history:
                self.successes.pop(0)
            self.success_rate = np.mean(self.successes)

            if self.success_rate > 0.6:
                sigma *= 1.1
            elif self.success_rate < 0.4:
                sigma *= 0.9
            
            sigma = np.clip(sigma, 1e-6, 1)
            
            mean = mean_new

            # Restart mechanism
            if np.max(np.diag(C)) > 1e7 * sigma**2:
                restarts += 1
                mean = np.random.uniform(self.lb, self.ub, size=self.dim)
                sigma = self.sigma0 * self.restart_factor**restarts
                C = np.eye(self.dim)
                p_sigma = np.zeros(self.dim)
                p_c = np.zeros(self.dim)

        return f_opt, x_opt