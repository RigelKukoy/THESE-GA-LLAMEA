import numpy as np

class SimplifiedCMAES:
    def __init__(self, budget=10000, dim=10, sigma0=0.5, success_rate_history=10):
        self.budget = budget
        self.dim = dim
        self.sigma0 = sigma0
        self.popsize = 4 + int(3 * np.log(self.dim))
        self.mu = self.popsize // 2
        self.weights = np.log(self.mu + 0.5) - np.log(np.arange(1, self.mu + 1))
        self.weights = self.weights / np.sum(self.weights)
        self.mueff = np.sum(self.weights)**2 / np.sum(self.weights**2)
        self.c_sigma = (self.mueff + 2) / (self.dim + self.mueff + 5)
        self.d_sigma = 1 + 2 * max(0, np.sqrt((self.mueff - 1) / (self.dim + 1)) - 1) + self.c_sigma
        self.success_history_length = success_rate_history
        self.success_rate_window = np.array([0.5] * self.success_history_length)
        self.success_rate = 0.5
        self.stagnation_threshold = 50
        self.stagnation_counter = 0

    def __call__(self, func):
        mean = np.random.uniform(func.bounds.lb, func.bounds.ub, size=self.dim)
        sigma = self.sigma0
        C = np.eye(self.dim)
        f_opt = np.inf
        x_opt = None
        eval_count = 0
        ps = np.zeros(self.dim)

        while eval_count < self.budget:
            z = np.random.normal(0, 1, size=(self.popsize, self.dim))
            x = mean + sigma * (np.linalg.cholesky(C) @ z.T).T
            
            f = np.array([func(xi) if eval_count + i < self.budget else np.inf for i, xi in enumerate(x)])
            eval_count += len(x)
            
            idx = np.argsort(f)
            x = x[idx]
            z = z[idx]
            f = f[idx]

            if f[0] < f_opt:
                f_opt = f[0]
                x_opt = x[0]
                self.stagnation_counter = 0
                self.success_history_update(1)
            else:
                self.stagnation_counter += 1
                self.success_history_update(0)

            if self.stagnation_counter > self.stagnation_threshold:
                mean = np.random.uniform(func.bounds.lb, func.bounds.ub, size=self.dim)
                sigma = self.sigma0
                C = np.eye(self.dim)
                ps = np.zeros(self.dim)
                self.stagnation_counter = 0
                self.success_rate = 0.5
                self.success_rate_window = np.array([0.5] * self.success_history_length)

            x_mu = x[:self.mu]
            z_mu = z[:self.mu]
            mean_new = np.sum(x_mu * self.weights[:, None], axis=0)
            z_w = np.sum(z_mu * self.weights[:, None], axis=0)

            ps = (1 - self.c_sigma) * ps + np.sqrt(self.c_sigma * (2 - self.c_sigma)) * z_w
            C = (1 - self.c_sigma) * C + self.c_sigma * np.outer(ps, ps)

            sigma *= np.exp(0.1 * (self.success_rate - 0.5) / self.d_sigma)

            mean = mean_new

            try:
                C = np.triu(C) + np.triu(C, 1).T
                C_evals = np.linalg.eigvalsh(C)
                C_evals = np.maximum(C_evals, 1e-10)
                C = np.linalg.eigh(C)[1] @ np.diag(C_evals) @ np.linalg.eigh(C)[1].T
            except np.linalg.LinAlgError:
                C = np.eye(self.dim)

        return f_opt, x_opt
    
    def success_history_update(self, success):
        self.success_rate_window = np.concatenate((self.success_rate_window[1:], [success]))
        self.success_rate = np.mean(self.success_rate_window)