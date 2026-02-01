import numpy as np

class ImprovedCMAES:
    def __init__(self, budget=10000, dim=10, sigma0=0.5, stagnation_threshold=15, adapt_popsize=True, bound_handling='clip'):
        self.budget = budget
        self.dim = dim
        self.sigma0 = sigma0
        self.stagnation_threshold = stagnation_threshold
        self.stagnation_counter = 0
        self.best_fitness = np.inf
        self.adapt_popsize = adapt_popsize
        self.bound_handling = bound_handling
        
        self.popsize = 4 + int(3 * np.log(self.dim)) if not adapt_popsize else int(4 + 3 * np.log(self.dim))
        self.mu = self.popsize // 2
        self.weights = np.log(self.mu + 0.5) - np.log(np.arange(1, self.mu + 1))
        self.weights = self.weights / np.sum(self.weights)
        self.mueff = np.sum(self.weights)**2 / np.sum(self.weights**2)
        self.stepsize_learning_rate = 0.2

    def __call__(self, func):
        # Initialize variables
        mean = np.random.uniform(func.bounds.lb, func.bounds.ub, size=self.dim)
        sigma = self.sigma0
        C = np.eye(self.dim)
        pc = np.zeros(self.dim)
        ps = np.zeros(self.dim)
        chiN = np.sqrt(self.dim) * (1 - (1 / (4 * self.dim)) + 1 / (21 * self.dim**2))

        # Parameters
        c_sigma = (self.mueff + 2) / (self.dim + self.mueff + 5)
        c_c = (4 + self.mueff / self.dim) / (self.dim + 4 + 2 * self.mueff / self.dim)
        c_1 = 2 / ((self.dim + 1.3)**2 + self.mueff)
        c_mu = min(1 - c_1, 2 * (self.mueff - 1 + 1 / self.mueff) / ((self.dim + 2)**2 + self.mueff))
        d_sigma = 1 + 2 * max(0, np.sqrt((self.mueff - 1) / (self.dim + 1)) - 1) + c_sigma
        c_1a = c_1
        c_mua = c_mu

        # Eigen decomposition of C
        C_evals, C_evecs = np.linalg.eigh(C)
        C_sqrt = C_evecs @ np.diag(np.sqrt(C_evals)) @ C_evecs.T
        C_invsqrt = C_evecs @ np.diag(1/np.sqrt(C_evals)) @ C_evecs.T
        
        f_opt = np.Inf
        x_opt = None
        eval_count = 0
        
        stepsize_factor = 1.0

        while eval_count < self.budget:
            # Adjust popsize
            if self.adapt_popsize:
                self.popsize = max(4, int(np.floor(4 + 3 * np.log(self.dim) * (0.8 + 0.2 * (self.budget - eval_count) / self.budget))))
                self.mu = self.popsize // 2
                self.weights = np.log(self.mu + 0.5) - np.log(np.arange(1, self.mu + 1))
                self.weights = self.weights / np.sum(self.weights)
                self.mueff = np.sum(self.weights)**2 / np.sum(self.weights**2)
                c_sigma = (self.mueff + 2) / (self.dim + self.mueff + 5)
                c_c = (4 + self.mueff / self.dim) / (self.dim + 4 + 2 * self.mueff / self.dim)
                c_1 = 2 / ((self.dim + 1.3)**2 + self.mueff)
                c_mu = min(1 - c_1, 2 * (self.mueff - 1 + 1 / self.mueff) / ((self.dim + 2)**2 + self.mueff))
                d_sigma = 1 + 2 * max(0, np.sqrt((self.mueff - 1) / (self.dim + 1)) - 1) + c_sigma

            # Sample population
            z = np.random.normal(0, 1, size=(self.popsize, self.dim))
            x = mean + sigma * (C_sqrt @ z.T).T

            # Boundary Handling
            if self.bound_handling == 'clip':
                x = np.clip(x, func.bounds.lb, func.bounds.ub)
            elif self.bound_handling == 'reflect':
                x = np.where(x < func.bounds.lb, 2 * func.bounds.lb - x, x)
                x = np.where(x > func.bounds.ub, 2 * func.bounds.ub - x, x)
            else: # 'none'
                pass # Do nothing

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
                self.stagnation_counter = 0
            else:
                self.stagnation_counter += 1
                
            # Stagnation check and restart
            if self.stagnation_counter > self.stagnation_threshold:
                # Restart strategy: reset mean and increase sigma
                mean = np.random.uniform(func.bounds.lb, func.bounds.ub, size=self.dim)
                sigma = self.sigma0 * 1.5
                C = np.eye(self.dim)
                pc = np.zeros(self.dim)
                ps = np.zeros(self.dim)
                C_evals, C_evecs = np.linalg.eigh(C)
                C_sqrt = C_evecs @ np.diag(np.sqrt(C_evals)) @ C_evecs.T
                C_invsqrt = C_evecs @ np.diag(1/np.sqrt(C_evals)) @ C_evecs.T
                self.stagnation_counter = 0

            # Selection and recombination
            x_mu = x[:self.mu]
            z_mu = z[:self.mu]

            mean_new = np.sum(x_mu * self.weights[:,None], axis=0)
            z_w = np.sum(z_mu * self.weights[:,None], axis=0)
            
            # Covariance matrix adaptation
            ps = (1 - c_sigma) * ps + np.sqrt(c_sigma * (2 - c_sigma) * self.mueff) * (C_invsqrt @ (mean_new - mean) / sigma)
            hsig = np.linalg.norm(ps) / np.sqrt(1 - (1 - c_sigma)**(2 * eval_count / self.popsize)) < chiN * (1.4 + 2 / (self.dim + 1))
            
            pc = (1 - c_c) * pc + hsig * np.sqrt(c_c * (2 - c_c) * self.mueff) * (mean_new - mean) / sigma

            C = (1 - c_1 - c_mu) * C + c_1 * (pc[:, None] @ pc[None, :])
            for i in range(self.mu):
                C += c_mu * self.weights[i] * (z_mu[i, :, None] @ z_mu[i, None, :])
                
            # Active CMA
            if c_1a > 0 and c_mua > 0:
                negidx = np.where(self.weights < 0)[0]
                znorm = np.zeros((len(negidx), self.dim))
                for i, idx in enumerate(negidx):
                    znorm[i] = z_mu[idx] * np.sqrt(-self.weights[idx])
                C += c_1a * (1/np.linalg.norm(ps)**2 + c_mua) * np.sum(znorm.T @ znorm, axis=1)

            # Update step size
            sigma *= np.exp((c_sigma / d_sigma) * (np.linalg.norm(ps) / chiN - 1))
            
            if f[0] < self.best_fitness:
                self.best_fitness = f[0]
                stepsize_factor *= np.exp(self.stepsize_learning_rate)  # Increase stepsize
            else:
                stepsize_factor *= np.exp(-self.stepsize_learning_rate/5)  # Decrease stepsize slower
            
            sigma *= stepsize_factor

            # Update mean
            mean = mean_new

            # Eigen decomposition of C
            if eval_count % (self.popsize * 5) == 0:
                C = np.triu(C) + np.triu(C, 1).T
                try:
                    C_evals, C_evecs = np.linalg.eigh(C)
                    C_evals = np.maximum(C_evals, 1e-10)
                    C = C_evecs @ np.diag(C_evals) @ C_evecs.T
                    C_sqrt = C_evecs @ np.diag(np.sqrt(C_evals)) @ C_evecs.T
                    C_invsqrt = C_evecs @ np.diag(1/np.sqrt(C_evals)) @ C_evecs.T
                except np.linalg.LinAlgError:
                    C = np.eye(self.dim)
                    C_evals, C_evecs = np.linalg.eigh(C)
                    C_sqrt = C_evecs @ np.diag(np.sqrt(C_evals)) @ C_evecs.T
                    C_invsqrt = C_evecs @ np.diag(1/np.sqrt(C_evals)) @ C_evecs.T
                
            if np.any(np.isnan(mean)) or np.any(np.isnan(C)):
                mean = np.random.uniform(func.bounds.lb, func.bounds.ub, size=self.dim)
                C = np.eye(self.dim)
                pc = np.zeros(self.dim)
                ps = np.zeros(self.dim)
                C_evals, C_evecs = np.linalg.eigh(C)
                C_sqrt = C_evecs @ np.diag(np.sqrt(C_evals)) @ C_evecs.T
                C_invsqrt = C_evecs @ np.diag(1/np.sqrt(C_evals)) @ C_evecs.T
                sigma = self.sigma0
        
        return f_opt, x_opt