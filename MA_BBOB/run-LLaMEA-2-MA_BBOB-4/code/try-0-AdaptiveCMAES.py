import numpy as np

class AdaptiveCMAES:
    def __init__(self, budget=10000, dim=10, initial_pop_size=None, sigma0=0.5, mu_ratio=0.25, ruggedness_detection_window=50):
        self.budget = budget
        self.dim = dim
        self.lb = -5.0
        self.ub = 5.0
        self.sigma0 = sigma0
        self.ruggedness_detection_window = ruggedness_detection_window

        if initial_pop_size is None:
            self.pop_size = 4 + int(3 * np.log(self.dim))
        else:
            self.pop_size = initial_pop_size
        self.min_pop_size = 4  # Minimum population size
        self.max_pop_size = 4 + int(6 * np.log(self.dim)) # Maximum population size
        
        self.mu = int(self.pop_size * mu_ratio)
        self.weights = np.log(self.mu + 0.5) - np.log(np.arange(1, self.mu + 1))
        self.weights /= np.sum(self.weights)
        self.mueff = np.sum(self.weights)**2 / np.sum(self.weights**2)
        self.c_sigma = (self.mueff + 2) / (self.dim + self.mueff + 5)
        self.d_sigma = 1 + 2 * max(0, np.sqrt((self.mueff - 1) / (self.dim + 1)) - 1) + self.c_sigma
        self.chiN = self.dim**0.5 * (1 - 1/(4*self.dim) + 1/(21*self.dim**2))
        self.c_cov = 2 / (self.dim + np.sqrt(2))**2 + self.mueff / self.pop_size #Simplified Covariance matrix adaptation

        self.fitness_history = []
        self.sigma_history = []

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
            Z = np.random.multivariate_normal(np.zeros(self.dim), C, size=self.pop_size)
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
            
            self.fitness_history.append(np.min(F))
            self.sigma_history.append(sigma)

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

            # Ruggedness detection and adaptation
            if len(self.fitness_history) > self.ruggedness_detection_window:
                fitness_window = self.fitness_history[-self.ruggedness_detection_window:]
                sigma_window = self.sigma_history[-self.ruggedness_detection_window:]
                fitness_std = np.std(fitness_window)
                sigma_mean = np.mean(sigma_window)

                if fitness_std < 1e-8 and sigma_mean < 1e-3: #Stagnation detected
                    self.pop_size = max(self.min_pop_size, self.pop_size // 2)
                    sigma = min(self.sigma0, sigma * 1.5) #Increase the step size.
                elif fitness_std > 1e-3: # Rugged landscape
                    self.pop_size = min(self.max_pop_size, self.pop_size * 2) #Increase the population size
                    sigma = max(1e-6, sigma / 1.1) #Decrease the step size.
                else:
                    self.pop_size = min(self.max_pop_size, max(self.min_pop_size, self.pop_size)) #Keep pop size as is
                    
                self.mu = int(self.pop_size * (self.mu / (self.pop_size + 1e-9))) #Adapt mu to the current pop size.

            mean = mean_new

        return f_opt, x_opt