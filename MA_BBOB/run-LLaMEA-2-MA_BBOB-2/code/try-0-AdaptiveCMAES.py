import numpy as np

class AdaptiveCMAES:
    def __init__(self, budget=10000, dim=10, initial_popsize = None, min_popsize = 4):
        self.budget = budget
        self.dim = dim
        self.initial_popsize = initial_popsize if initial_popsize is not None else 4 + int(3 * np.log(self.dim))
        self.min_popsize = min_popsize
        self.popsize = self.initial_popsize
        self.mu = self.popsize // 2
        self.weights = np.log(self.mu + 1/2) - np.log(np.arange(1, self.mu + 1))
        self.weights = self.weights / np.sum(self.weights)
        self.m = np.zeros(self.dim)
        self.sigma = 0.5
        self.C = np.eye(self.dim)
        self.p_sigma = np.zeros(self.dim)
        self.p_c = np.zeros(self.dim)
        self.chiN = np.sqrt(self.dim) * (1 - 1/(4*self.dim) + 1/(21*self.dim**2))
        self.c_sigma = (self.mu + 2) / (self.dim + self.mu + 5)
        self.c_c = (4 + self.mu/self.dim) / (self.dim + 4 + 2*self.mu/self.dim)
        self.c_1 = 2 / ((self.dim + 1.3)**2 + self.mu)
        self.c_mu = min(1 - self.c_1, 2 * (self.mu - 2 + 1/self.mu) / ((self.dim + 2)**2 + self.mu))
        self.d_sigma = 1 + 2*max(0, np.sqrt((self.mu-1)/(self.dim+1)) - 1) + self.c_sigma
        self.D = None
        self.B = None
        self.function_evals = 0
        self.success_history = []
        self.learning_rate_scaling = 1.0


    def __call__(self, func):
        self.f_opt = np.Inf
        self.x_opt = None

        while self.function_evals < self.budget:
            # Sample population
            z = np.random.normal(0, 1, size=(self.popsize, self.dim))
            if self.D is None or self.B is None:
                 self.D, self.B = np.linalg.eigh(self.C)
                 self.D = np.sqrt(np.maximum(self.D, 1e-16))
            y = self.B @ np.diag(self.D) @ z.T
            x = self.m + self.sigma * y.T

            # Evaluate population, ensuring bounds are respected and budget is not exceeded.
            f = np.zeros(self.popsize)
            for i in range(self.popsize):
                x[i] = np.clip(x[i], func.bounds.lb, func.bounds.ub)
                if self.function_evals < self.budget:
                    f[i] = func(x[i])
                    self.function_evals += 1
                    if f[i] < self.f_opt:
                        self.f_opt = f[i]
                        self.x_opt = x[i]
                else:
                    f[i] = np.inf

            # Sort by fitness
            idx = np.argsort(f)
            x = x[idx]
            z = z[idx]
            f = f[idx]

            # Update mean
            m_old = self.m.copy()
            self.m = np.sum(self.weights[:, None] * x[:self.mu], axis=0)

            # Update evolution paths
            y_mean = np.mean(z[:self.mu], axis=0)
            self.p_sigma = (1 - self.c_sigma) * self.p_sigma + np.sqrt(self.c_sigma * (2 - self.c_sigma)) * (self.B @ y_mean)
            hsig = np.linalg.norm(self.p_sigma) / np.sqrt(1 - (1 - self.c_sigma)**(2*self.function_evals/self.popsize))/self.chiN < 1.4 + 2/(self.dim+1)
            self.p_c = (1 - self.c_c) * self.p_c + hsig * np.sqrt(self.c_c * (2 - self.c_c)) * (self.m - m_old) / self.sigma

            # Update covariance matrix
            self.C = (1 - self.c_1 - self.c_mu) * self.C + self.c_1 * (self.p_c[:, None] @ self.p_c[None, :]) + self.c_mu * np.sum(self.weights[:, None, None] * (z[:self.mu, :, None] @ z[:self.mu, None, :]), axis=0)

            # Update step size
            self.sigma *= np.exp((self.c_sigma / self.d_sigma) * (np.linalg.norm(self.p_sigma)/self.chiN - 1)) * self.learning_rate_scaling

            self.D = None # invalidate cached B and D

            # Adaptive Pop Size and Learning Rate
            if len(self.success_history) > 10:
                success_rate = np.mean(self.success_history[-10:])
                if success_rate > 0.8 and self.popsize > self.min_popsize:
                    self.popsize = max(self.min_popsize, self.popsize // 2)
                    self.mu = self.popsize // 2
                    self.weights = np.log(self.mu + 1/2) - np.log(np.arange(1, self.mu + 1))
                    self.weights = self.weights / np.sum(self.weights)
                    self.c_sigma = (self.mu + 2) / (self.dim + self.mu + 5)
                    self.c_c = (4 + self.mu/self.dim) / (self.dim + 4 + 2*self.mu/self.dim)
                    self.c_1 = 2 / ((self.dim + 1.3)**2 + self.mu)
                    self.c_mu = min(1 - self.c_1, 2 * (self.mu - 2 + 1/self.mu) / ((self.dim + 2)**2 + self.mu))
                    self.d_sigma = 1 + 2*max(0, np.sqrt((self.mu-1)/(self.dim+1)) - 1) + self.c_sigma
                    self.learning_rate_scaling *= 0.9  # Reduce learning rate when shrinking popsize
                    
                elif success_rate < 0.2 and self.popsize < self.initial_popsize:
                    self.popsize = min(self.initial_popsize, self.popsize * 2)
                    self.mu = self.popsize // 2
                    self.weights = np.log(self.mu + 1/2) - np.log(np.arange(1, self.mu + 1))
                    self.weights = self.weights / np.sum(self.weights)
                    self.c_sigma = (self.mu + 2) / (self.dim + self.mu + 5)
                    self.c_c = (4 + self.mu/self.dim) / (self.dim + 4 + 2*self.mu/self.dim)
                    self.c_1 = 2 / ((self.dim + 1.3)**2 + self.mu)
                    self.c_mu = min(1 - self.c_1, 2 * (self.mu - 2 + 1/self.mu) / ((self.dim + 2)**2 + self.mu))
                    self.d_sigma = 1 + 2*max(0, np.sqrt((self.mu-1)/(self.dim+1)) - 1) + self.c_sigma
                    self.learning_rate_scaling *= 1.1  # Increase learning rate when expanding popsize
            
            self.success_history.append(int(f[0] < self.f_opt)) #Record success if the best in pop is better than global best
            
        return self.f_opt, self.x_opt