import numpy as np

class CoordinateCMAES:
    def __init__(self, budget=10000, dim=10, sigma0=0.5, history_length=5, mirror_ratio=0.5, cs_damp=0.1):
        self.budget = budget
        self.dim = dim
        self.sigma0 = sigma0
        self.history_length = history_length
        self.min_popsize = 4
        self.max_popsize = 50
        self.target_popsize = min(self.max_popsize, self.min_popsize + int(np.floor(3 * np.log(self.dim))))
        self.popsize = self.target_popsize
        self.mu = self.popsize // 2
        self.weights = np.log(self.mu + 0.5) - np.log(np.arange(1, self.mu + 1))
        self.weights = self.weights / np.sum(self.weights)
        self.mueff = np.sum(self.weights)**2 / np.sum(self.weights**2)
        self.step_history = []
        self.last_f_opt = np.inf
        self.stagnation_counter = 0
        self.restart_iterations = 50
        self.mirror_ratio = mirror_ratio
        self.cs_damp = cs_damp  # Damping for coordinate-wise step size

    def __call__(self, func):
        mean = np.random.uniform(func.bounds.lb, func.bounds.ub, size=self.dim)
        sigma = np.ones(self.dim) * self.sigma0  # Coordinate-wise step sizes
        C = np.eye(self.dim)
        pc = np.zeros(self.dim)
        ps = np.zeros(self.dim)
        chiN = np.sqrt(self.dim) * (1 - (1 / (4 * self.dim)) + 1 / (21 * self.dim**2))

        c_sigma = (self.mueff + 2) / (self.dim + self.mueff + 5)
        c_c = (4 + self.mueff / self.dim) / (self.dim + 4 + 2 * self.mueff / self.dim)
        c_1 = 2 / ((self.dim + 1.3)**2 + self.mueff)
        c_mu = min(1 - c_1, 2 * (self.mueff - 1 + 1 / self.mueff) / ((self.dim + 2)**2 + self.mueff))
        d_sigma = 1 + 2 * max(0, np.sqrt((self.mueff - 1) / (self.dim + 1)) - 1) + c_sigma

        C_evals, C_evecs = np.linalg.eigh(C)
        C_sqrt = C_evecs @ np.diag(np.sqrt(C_evals)) @ C_evecs.T
        C_invsqrt = C_evecs @ np.diag(1 / np.sqrt(C_evals)) @ C_evecs.T
        
        f_opt = np.Inf
        x_opt = None
        eval_count = 0

        while eval_count < self.budget:
            z = np.random.normal(0, 1, size=(self.popsize, self.dim))
            x = mean + (sigma * (C_sqrt @ z.T).T)

            # Mirrored Sampling
            num_mirrored = int(self.popsize * self.mirror_ratio)
            z_mirrored = -z[:num_mirrored]
            x_mirrored = mean + (sigma * (C_sqrt @ z_mirrored.T).T)
            x = np.concatenate((x, x_mirrored), axis=0)
            z = np.concatenate((z, z_mirrored), axis=0)

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
            else:
                self.stagnation_counter += 1
                
            if self.stagnation_counter > self.restart_iterations:
                mean = np.random.uniform(func.bounds.lb, func.bounds.ub, size=self.dim)
                sigma = np.ones(self.dim) * self.sigma0
                C = np.eye(self.dim)
                pc = np.zeros(self.dim)
                ps = np.zeros(self.dim)
                C_evals, C_evecs = np.linalg.eigh(C)
                C_sqrt = C_evecs @ np.diag(np.sqrt(C_evals)) @ C_evecs.T
                C_invsqrt = C_evecs @ np.diag(1 / np.sqrt(C_evals)) @ C_evecs.T
                self.stagnation_counter = 0

            x_mu = x[:self.mu]
            z_mu = z[:self.mu]

            mean_new = np.sum(x_mu * self.weights[:, None], axis=0)
            z_w = np.sum(z_mu * self.weights[:, None], axis=0)
            
            ps = (1 - c_sigma) * ps + np.sqrt(c_sigma * (2 - c_sigma) * self.mueff) * (C_invsqrt @ ((mean_new - mean) / sigma))
            hsig = np.linalg.norm(ps) / np.sqrt(1 - (1 - c_sigma)**(2 * eval_count / self.popsize)) < chiN * (1.4 + 2 / (self.dim + 1))
            pc = (1 - c_c) * pc + hsig * np.sqrt(c_c * (2 - c_c) * self.mueff) * ((mean_new - mean) / sigma)

            C = (1 - c_1 - c_mu) * C + c_1 * (pc[:, None] @ pc[None, :])
            for i in range(self.mu):
                C += c_mu * self.weights[i] * (z_mu[i, :, None] @ z_mu[i, None, :])
            
            sigma *= np.exp((c_sigma / d_sigma) * (np.linalg.norm(ps) / chiN - 1))  # Global step size adaptation
            sigma *= np.exp(self.cs_damp * pc)  # Adapt coordinate-wise

            # Simplified Step history
            step = (mean_new - mean) / sigma
            self.step_history.append(step)
            if len(self.step_history) > self.history_length:
                self.step_history.pop(0)

            # Adapt C based on step history, simplified
            for h_step in self.step_history:
                C += 0.1 * c_mu * (h_step[:, None] @ h_step[None, :])

            mean = mean_new
            
            if eval_count % (self.popsize * 5) == 0:
                C = np.triu(C) + np.triu(C, 1).T
                try:
                    C_evals, C_evecs = np.linalg.eigh(C)
                    C_evals = np.maximum(C_evals, 1e-10)
                    C = C_evecs @ np.diag(C_evals) @ C_evecs.T
                    C_sqrt = C_evecs @ np.diag(np.sqrt(C_evals)) @ C_evecs.T
                    C_invsqrt = C_evecs @ np.diag(1 / np.sqrt(C_evals)) @ C_evecs.T
                except np.linalg.LinAlgError:
                    C = np.eye(self.dim)
                    C_evals, C_evecs = np.linalg.eigh(C)
                    C_sqrt = C_evecs @ np.diag(np.sqrt(C_evals)) @ C_evecs.T
                    C_invsqrt = C_evecs @ np.diag(1 / np.sqrt(C_evals)) @ C_evecs.T

                # Regularize Covariance Matrix
                C = C + 1e-8 * np.eye(self.dim)
                
            if np.any(np.isnan(mean)) or np.any(np.isnan(C)):
                mean = np.random.uniform(func.bounds.lb, func.bounds.ub, size=self.dim)
                sigma = np.ones(self.dim) * self.sigma0
                C = np.eye(self.dim)
                pc = np.zeros(self.dim)
                ps = np.zeros(self.dim)
                C_evals, C_evecs = np.linalg.eigh(C)
                C_sqrt = C_evecs @ np.diag(np.sqrt(C_evals)) @ C_evecs.T
                C_invsqrt = C_evecs @ np.diag(1 / np.sqrt(C_evals)) @ C_evecs.T
                sigma = np.ones(self.dim) * self.sigma0

        return f_opt, x_opt