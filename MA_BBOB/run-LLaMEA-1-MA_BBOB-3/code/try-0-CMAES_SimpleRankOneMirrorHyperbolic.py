import numpy as np

class CMAES_SimpleRankOneMirrorHyperbolic:
    def __init__(self, budget=10000, dim=10, sigma0=0.5, history_length=5, initial_mirrored_fraction=0.5, mirrored_decay=0.99, hyperbolic_decay_factor=0.1, step_size_adaptation_rate=0.2):
        self.budget = budget
        self.dim = dim
        self.sigma0 = sigma0
        self.history_length = history_length
        self.mirrored_fraction = initial_mirrored_fraction
        self.mirrored_decay = mirrored_decay
        self.hyperbolic_decay_factor = hyperbolic_decay_factor
        self.step_size_adaptation_rate = step_size_adaptation_rate

        self.popsize = 4 + int(np.floor(3 * np.log(self.dim)))
        self.mu = self.popsize // 2
        self.weights = np.log(self.mu + 0.5) - np.log(np.arange(1, self.mu + 1))
        self.weights = self.weights / np.sum(self.weights)
        self.mueff = np.sum(self.weights)**2 / np.sum(self.weights**2)

        self.step_history = []

    def __call__(self, func):
        mean = np.random.uniform(func.bounds.lb, func.bounds.ub, size=self.dim)
        sigma = self.sigma0
        C = np.eye(self.dim)
        pc = np.zeros(self.dim)
        
        c_sigma = (self.mueff + 2) / (self.dim + self.mueff + 5)
        c_c = (4 + self.mueff / self.dim) / (self.dim + 4 + 2 * self.mueff / self.dim)
        c_1 = 2 / ((self.dim + 1.3)**2 + self.mueff)
        
        chiN = np.sqrt(self.dim) * (1 - (1 / (4 * self.dim)) + 1 / (21 * self.dim**2))

        f_opt = np.Inf
        x_opt = None
        eval_count = 0

        while eval_count < self.budget:
            z = np.random.normal(0, 1, size=(self.popsize, self.dim))
            x = mean + sigma * (np.linalg.cholesky(C) @ z.T).T

            # Mirrored sampling
            mirrored_fraction = self.mirrored_fraction / (1 + self.hyperbolic_decay_factor * (eval_count / self.budget))
            num_mirrored = int(self.popsize * mirrored_fraction)
            x_mirrored = mean - sigma * (np.linalg.cholesky(C) @ z[:num_mirrored].T).T
            x = np.vstack((x, x_mirrored))
            z_mirrored = -z[:num_mirrored]
            z = np.vstack((z, z_mirrored))
            
            # Enhanced bound handling - clipping and penalty
            x = np.clip(x, func.bounds.lb, func.bounds.ub)
            
            f = np.array([func(xi) + 1e10 * (np.sum((xi < func.bounds.lb) | (xi > func.bounds.ub))) if eval_count + i < self.budget else np.inf for i, xi in enumerate(x)])
            eval_count += len(x)

            idx = np.argsort(f)
            x = x[idx]
            z = z[idx]
            f = f[idx]

            if f[0] < f_opt:
                f_opt = f[0]
                x_opt = x[0]

            x_mu = x[:self.mu]
            mean_new = np.sum(x_mu * self.weights[:,None], axis=0)

            # Simplified Rank-One Update
            pc = (1 - c_c) * pc + np.sqrt(c_c * (2 - c_c) * self.mueff) * (mean_new - mean) / sigma
            C = (1 - c_1) * C + c_1 * np.outer(pc, pc)
            
            # Step size adaptation using moving average
            step = (mean_new - mean) / sigma
            self.step_history.append(np.linalg.norm(step))
            if len(self.step_history) > self.history_length:
                self.step_history.pop(0)

            avg_step_size = np.mean(self.step_history) if self.step_history else 1
            sigma *= np.exp(self.step_size_adaptation_rate * (avg_step_size - 1))

            mean = mean_new
            self.mirrored_fraction *= self.mirrored_decay

            # Ensure C is positive definite
            C = np.triu(C) + np.triu(C, 1).T
            try:
                C_evals = np.linalg.eigvalsh(C)
                C_min_eig = np.min(C_evals)
                if C_min_eig < 1e-10:
                    C += (1e-10 - C_min_eig) * np.eye(self.dim)
            except np.linalg.LinAlgError:
                C = np.eye(self.dim)

            if np.any(np.isnan(mean)) or np.any(np.isnan(C)):
                mean = np.random.uniform(func.bounds.lb, func.bounds.ub, size=self.dim)
                C = np.eye(self.dim)
                pc = np.zeros(self.dim)
                sigma = self.sigma0

        return f_opt, x_opt