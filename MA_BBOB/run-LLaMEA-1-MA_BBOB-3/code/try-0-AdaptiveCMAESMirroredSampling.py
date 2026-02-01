import numpy as np

class AdaptiveCMAESMirroredSampling:
    def __init__(self, budget=10000, dim=10, sigma0=0.5, stagnation_threshold=10, success_history_length=10):
        self.budget = budget
        self.dim = dim
        self.sigma0 = sigma0
        self.stagnation_threshold = stagnation_threshold
        self.stagnation_counter = 0
        self.best_fitness = np.inf
        self.success_history_length = success_history_length
        self.success_history = []

        self.popsize = 4 + int(np.floor(3 * np.log(self.dim)))
        self.mu = self.popsize // 2
        self.weights = np.log(self.mu + 0.5) - np.log(np.arange(1, self.mu + 1))
        self.weights = self.weights / np.sum(self.weights)
        self.mueff = np.sum(self.weights)**2 / np.sum(self.weights**2)

        self.c_sigma = (self.mueff + 2) / (self.dim + self.mueff + 5)
        self.d_sigma = 1 + 2 * max(0, np.sqrt((self.mueff - 1) / (self.dim + 1)) - 1) + self.c_sigma

    def __call__(self, func):
        mean = np.random.uniform(func.bounds.lb, func.bounds.ub, size=self.dim)
        sigma = self.sigma0
        C = np.eye(self.dim)
        eval_count = 0
        f_opt = np.inf
        x_opt = None
        
        while eval_count < self.budget:
            z = np.random.normal(0, 1, size=(self.popsize // 2, self.dim))
            x = mean + sigma * (np.linalg.cholesky(C) @ z.T).T
            x_mirrored = mean - sigma * (np.linalg.cholesky(C) @ z.T).T
            x = np.vstack((x, x_mirrored))
            z = np.vstack((z, -z))
            
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
                
            if self.stagnation_counter > self.stagnation_threshold:
                mean = np.random.uniform(func.bounds.lb, func.bounds.ub, size=self.dim)
                sigma = self.sigma0 * 2
                C = np.eye(self.dim)
                self.stagnation_counter = 0

            x_mu = x[:self.mu]
            z_mu = z[:self.mu]
            mean_new = np.sum(x_mu * self.weights[:,None], axis=0)

            # Simplified Covariance Matrix Update
            C = (1 - self.c_sigma) * C + self.c_sigma * np.cov(z_mu.T)

            #Dynamic Sigma adaptation
            success_rate = (self.best_fitness > f[0])
            self.success_history.append(success_rate)
            if len(self.success_history) > self.success_history_length:
                self.success_history.pop(0)
            
            recent_success_rate = np.mean(self.success_history) if self.success_history else 0.5
            
            if recent_success_rate > 0.7:
                sigma *= 1.1
            elif recent_success_rate < 0.3:
                sigma *= 0.9

            self.best_fitness = min(self.best_fitness, f[0])
            mean = mean_new

            if eval_count % self.popsize == 0:
                try:
                    C = np.triu(C) + np.triu(C, 1).T
                    C = np.linalg.cholesky(C) @ np.linalg.cholesky(C).T
                    eigenvalues = np.linalg.eigvalsh(C)
                    if np.any(eigenvalues < 1e-6):
                        C += np.eye(self.dim) * 1e-5
                except np.linalg.LinAlgError:
                    C = np.eye(self.dim)
                    
        return f_opt, x_opt