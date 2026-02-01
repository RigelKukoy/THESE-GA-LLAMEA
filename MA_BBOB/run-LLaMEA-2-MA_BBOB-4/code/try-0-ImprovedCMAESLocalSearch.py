import numpy as np

class ImprovedCMAESLocalSearch:
    def __init__(self, budget=10000, dim=10, pop_size=None, sigma0=0.5, mu_ratio=0.25, stagnation_threshold=100):
        self.budget = budget
        self.dim = dim
        self.lb = -5.0
        self.ub = 5.0
        self.sigma0 = sigma0
        if pop_size is None:
            self.pop_size = 4 + int(3 * np.log(self.dim))
        else:
            self.pop_size = pop_size
        self.mu = int(self.pop_size * mu_ratio)
        self.weights = np.log(self.mu + 0.5) - np.log(np.arange(1, self.mu + 1))
        self.weights /= np.sum(self.weights)
        self.mueff = np.sum(self.weights)**2 / np.sum(self.weights**2)
        self.stagnation_threshold = stagnation_threshold

    def local_search(self, func, x_opt, sigma, num_iterations=10):
        """Performs a simple local search around the current best solution."""
        for _ in range(num_iterations):
            x_new = x_opt + np.random.normal(0, sigma, self.dim)
            x_new = np.clip(x_new, self.lb, self.ub)
            f_new = func(x_new)
            if f_new < self.f_opt:
                self.f_opt = f_new
                self.x_opt = x_new.copy()


    def __call__(self, func):
        mean = np.random.uniform(self.lb, self.ub, size=self.dim)
        sigma = self.sigma0
        C = np.eye(self.dim)
        f_opt = np.Inf
        x_opt = None
        eval_count = 0
        stagnation_counter = 0
        previous_f_opt = np.Inf


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
                if f_opt < previous_f_opt:
                    stagnation_counter = 0
                    previous_f_opt = f_opt
            else:
                stagnation_counter += 1
                
            self.f_opt = f_opt
            self.x_opt = x_opt

            # Selection and recombination
            idx = np.argsort(F)
            X_mu = X[idx[:self.mu]]
            mean_new = np.sum(self.weights[:, None] * X_mu, axis=0)
            
            # Simplified covariance matrix adaptation (rank-one update)
            z_mu = (mean_new - mean) / sigma
            C = (1 - 0.1) * C + 0.1 * np.outer(z_mu, z_mu)
            
            # Ensure C is positive definite
            C = np.triu(C)
            C += C.T - np.diag(C.diagonal())
            try:
                np.linalg.cholesky(C)  # Check if positive definite
            except np.linalg.LinAlgError:
                C = np.eye(self.dim)  # Reset if not positive definite

            # Update step size
            sigma *= np.exp(0.2 * (np.linalg.norm(z_mu) - np.sqrt(self.dim)))
            sigma = np.clip(sigma, 1e-6, 1)

            mean = mean_new

            if stagnation_counter > self.stagnation_threshold:
                self.local_search(func, self.x_opt, sigma/2)
                stagnation_counter = 0  # Reset stagnation counter

            f_opt = self.f_opt
            x_opt = self.x_opt
                
        return f_opt, x_opt