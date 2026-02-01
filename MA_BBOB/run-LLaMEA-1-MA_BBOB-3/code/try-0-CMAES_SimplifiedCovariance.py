import numpy as np

class CMAES_SimplifiedCovariance:
    def __init__(self, budget=10000, dim=10, sigma0=0.5, initial_mirrored_fraction=0.5, mirrored_decay=0.99, active_adaptation_multiplier=0.1, cs=0.3, cc=0.4):
        self.budget = budget
        self.dim = dim
        self.sigma0 = sigma0
        self.mirrored_fraction = initial_mirrored_fraction
        self.mirrored_decay = mirrored_decay
        self.active_adaptation_multiplier = active_adaptation_multiplier
        self.cs = cs
        self.cc = cc

        self.popsize = 4 + int(np.floor(3 * np.log(self.dim)))
        self.mu = self.popsize // 2
        self.weights = np.log(self.mu + 0.5) - np.log(np.arange(1, self.mu + 1))
        self.weights = self.weights / np.sum(self.weights)
        self.mueff = np.sum(self.weights)**2 / np.sum(self.weights**2)
        self.success_history = []
        self.success_window = 10

    def __call__(self, func):
        # Initialize variables
        mean = np.random.uniform(func.bounds.lb, func.bounds.ub, size=self.dim)
        sigma = self.sigma0
        p_sigma = np.zeros(self.dim)
        C = np.eye(self.dim)

        # Parameters (simplified)
        damps = 1 + self.cs + 2 * max(0, np.sqrt((self.mueff - 1) / (self.dim + 1)) - 1)
        chiN = np.sqrt(self.dim) * (1 - (1 / (4 * self.dim)) + 1 / (21 * self.dim**2))

        f_opt = np.Inf
        x_opt = None
        eval_count = 0
        successes = 0

        while eval_count < self.budget:
            # Sample population
            z = np.random.normal(0, 1, size=(self.popsize, self.dim))
            x = mean + sigma * z

            # Mirrored sampling
            num_mirrored = int(self.popsize * self.mirrored_fraction)
            x_mirrored = mean - sigma * z[:num_mirrored]
            x = np.vstack((x, x_mirrored))
            z_mirrored = -z[:num_mirrored]
            z = np.vstack((z, z_mirrored))

            # Clipping to bounds
            x = np.clip(x, func.bounds.lb, func.bounds.ub)

            # Evaluate the new points
            f = np.array([func(xi) if eval_count + i < self.budget else np.inf for i, xi in enumerate(x)])
            eval_count += len(x)

            # Sort by fitness
            idx = np.argsort(f)
            x = x[idx]
            z = z[idx]
            f = f[idx]

            # Update optimal solution
            if f[0] < f_opt:
                f_opt = f[0]
                x_opt = x[0]
                successes += 1

            # Selection and recombination
            x_mu = x[:self.mu]
            z_mu = z[:self.mu]
            mean_new = np.sum(x_mu * self.weights[:, None], axis=0)
            z_w = np.sum(z_mu * self.weights[:, None], axis=0)

            # Simplified covariance matrix adaptation
            p_sigma = (1 - self.cs) * p_sigma + np.sqrt(self.cs * (2 - self.cs) * self.mueff) * z_w
            C = (1 - self.cc) * C + self.cc * (p_sigma[:, None] @ p_sigma[None, :])

            # Active Covariance Adaptation
            if f[0] > f_opt:
                C -= self.active_adaptation_multiplier * self.cc * (p_sigma[:, None] @ p_sigma[None, :])

            # Update step size
            sigma *= np.exp((self.cs / damps) * (np.linalg.norm(p_sigma) / chiN - 1))
            sigma = max(sigma, 1e-10)

            # Update mean
            mean = mean_new

            # Adaptive Mirrored fraction
            self.mirrored_fraction *= self.mirrored_decay

            # Repair covariance matrix if needed
            if eval_count % (self.popsize * 5) == 0:
                 C = np.triu(C) + np.triu(C, 1).T
                 try:
                     C_evals, C_evecs = np.linalg.eigh(C)
                     C_evals = np.maximum(C_evals, 1e-10)
                     C = C_evecs @ np.diag(C_evals) @ C_evecs.T
                 except np.linalg.LinAlgError:
                     C = np.eye(self.dim)

            if np.any(np.isnan(mean)) or np.any(np.isnan(C)):
                mean = np.random.uniform(func.bounds.lb, func.bounds.ub, size=self.dim)
                C = np.eye(self.dim)
                p_sigma = np.zeros(self.dim)
                sigma = self.sigma0
                successes = 0

        return f_opt, x_opt