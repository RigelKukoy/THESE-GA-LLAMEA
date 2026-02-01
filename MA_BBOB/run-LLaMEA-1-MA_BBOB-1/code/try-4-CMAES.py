import numpy as np

class CMAES:
    def __init__(self, budget=10000, dim=10, popsize=None, cs=0.3, damp=None, c_cov=None, mu_factor=0.25):
        self.budget = budget
        self.dim = dim
        self.popsize = popsize if popsize is not None else 4 + int(3 * np.log(self.dim))
        self.mu = int(self.popsize * mu_factor)
        self.weights = np.log(self.mu + 0.5) - np.log(np.arange(1, self.mu + 1))
        self.weights /= np.sum(self.weights)
        self.mueff = np.sum(self.weights)**2 / np.sum(self.weights**2)
        self.cs = cs
        self.damp = damp if damp is not None else 1 + 2 * max(0, np.sqrt((self.mueff - 1) / (self.dim + 1)) - 1) + self.cs
        self.c_cov = c_cov if c_cov is not None else (1 / (self.mueff * self.dim**2)) * (self.mueff - 1 + 2 / self.mueff)
        self.chiN = self.dim**0.5 * (1 - 1 / (4 * self.dim) + 1 / (21 * self.dim**2))

    def __call__(self, func):
        mean = np.zeros(self.dim)
        sigma = 0.5
        C = np.eye(self.dim)
        pc = np.zeros(self.dim)
        ps = np.zeros(self.dim)
        f_opt = np.Inf
        x_opt = None
        used_budget = 0
        lb = func.bounds.lb
        ub = func.bounds.ub

        while used_budget < self.budget:
            z = np.random.normal(0, 1, size=(self.dim, self.popsize))
            A = np.linalg.cholesky(C)
            x = mean[:, np.newaxis] + sigma * np.dot(A, z)
            
            # Clip each dimension of each individual
            x = np.clip(x, lb, ub)

            f = np.array([func(x[:, i]) for i in range(self.popsize)])
            used_budget += self.popsize
            
            if np.min(f) < f_opt:
                f_opt = np.min(f)
                x_opt = x[:, np.argmin(f)].copy()
            
            idx = np.argsort(f)
            x_mu = x[:, idx[:self.mu]]
            z_mu = z[:, idx[:self.mu]]
            mean_new = np.dot(x_mu, self.weights)

            ps = (1 - self.cs) * ps + np.sqrt(self.cs * (2 - self.cs) * self.mueff) * np.dot(np.linalg.inv(A), (mean_new - mean) / sigma)
            
            if np.linalg.norm(ps) / np.sqrt(1 - (1 - self.cs)**(2 * used_budget / self.popsize)) < (1.4 + 2 / (self.dim + 1)) * self.chiN:
                hsig = 1
            else:
                hsig = 0

            mean = mean_new

            sigma *= np.exp((self.cs / self.damp) * (np.linalg.norm(ps) / self.chiN - 1))

            C = (1 - self.c_cov) * C + self.c_cov * (1 / self.mueff) * (np.outer(ps, ps) + (1 - hsig) * self.c_cov * (2 - self.c_cov) * C)
            C += self.c_cov * np.dot(z_mu * self.weights, z_mu.T)

            C = np.triu(C) + np.triu(C, 1).T
            C = (C + C.T) / 2
            
            try:
                np.linalg.cholesky(C)
            except np.linalg.LinAlgError:
                C = np.eye(self.dim)  # Reset covariance matrix if it's not positive definite
        
        return f_opt, x_opt