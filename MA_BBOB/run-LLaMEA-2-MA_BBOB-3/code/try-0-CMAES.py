import numpy as np

class CMAES:
    def __init__(self, budget=10000, dim=10, popsize=None, cs=0.3, damps=None, c_cov=None, sigma0=0.2):
        self.budget = budget
        self.dim = dim
        self.sigma0 = sigma0
        self.popsize = popsize if popsize is not None else 4 + int(3 * np.log(self.dim))

        self.mu = self.popsize // 2

        self.weights = np.log(self.mu + 1/2) - np.log(np.arange(1, self.mu + 1))
        self.weights = self.weights / np.sum(self.weights)

        self.m = None
        self.sigma = None
        self.C = None
        self.pc = None
        self.ps = None
        self.eigenspace = None
        self.eigenvalues = None

        self.cs = cs
        self.damps = damps if damps is not None else 1 + 2 * np.max([0, np.sqrt((self.mu - 1)/(self.dim + 1)) - 1]) + self.cs
        self.c_cov = c_cov if c_cov is not None else (1 / (self.dim * np.sqrt(self.dim))) * 10
        self.c_cov_mu = self.c_cov
        self.c_cov_mu = self.c_cov_mu if self.c_cov_mu <= 1 else 1
        self.c_cov_mu = self.c_cov_mu if self.c_cov_mu > 0 else 0
        self.c_cov = (1 / self.mu) * self.c_cov

        self.f_opt = np.Inf
        self.x_opt = None
        self.func_evals = 0
        self.stagnation_counter = 0
        self.stagnation_threshold = 50  # Number of iterations to wait before restart

    def initialize(self):
        self.m = np.random.uniform(-2, 2, size=self.dim)
        self.sigma = self.sigma0
        self.C = np.eye(self.dim)
        self.pc = np.zeros(self.dim)
        self.ps = np.zeros(self.dim)
        self.eigenspace = np.eye(self.dim)
        self.eigenvalues = np.ones(self.dim)

    def sample(self):
        z = np.random.normal(0, 1, size=(self.dim, self.popsize))
        x = self.m[:, np.newaxis] + self.sigma * (self.eigenspace @ (np.diag(np.sqrt(self.eigenvalues)) @ z))
        return x

    def update(self, x, fitness_values):
        idx = np.argsort(fitness_values)
        x_mu = x[:, idx[:self.mu]]
        z_mu = np.linalg.solve(self.eigenspace @ np.diag(np.sqrt(self.eigenvalues)), (x_mu - self.m[:, np.newaxis]) / self.sigma)

        m_old = self.m.copy()
        self.m = np.sum(x_mu * self.weights[np.newaxis, :], axis=1)
        self.ps = (1 - self.cs) * self.ps + np.sqrt(self.cs * (2 - self.cs) * np.sum(self.weights)) * (self.eigenspace @ z_mu.mean(axis=1))
        self.pc = (1 - self.c_cov) * self.pc + np.sqrt(self.c_cov * (2 - self.c_cov) * np.sum(self.weights)) * ((self.m - m_old) / self.sigma)
        
        C_temp = self.c_cov_mu * (self.pc[:, np.newaxis] @ self.pc[np.newaxis, :]) + \
               (1 - self.c_cov_mu) * (self.C)

        self.sigma *= np.exp((self.cs / self.damps) * (np.linalg.norm(self.ps) / np.sqrt(self.dim) - 1))

        self.C = C_temp

        self.eigenvalues, self.eigenspace = np.linalg.eigh(self.C)
        self.eigenvalues = np.maximum(self.eigenvalues, 1e-12)

    def __call__(self, func):
        self.initialize()

        while self.func_evals < self.budget:
            x = self.sample()

            fitness_values = np.array([func(x[:, i]) for i in range(self.popsize)])
            self.func_evals += self.popsize

            best_index = np.argmin(fitness_values)
            if fitness_values[best_index] < self.f_opt:
                self.f_opt = fitness_values[best_index]
                self.x_opt = x[:, best_index]
                self.stagnation_counter = 0  # Reset counter if improvement

            self.update(x, fitness_values)
            self.stagnation_counter += 1

            if self.stagnation_counter > self.stagnation_threshold:
                self.initialize()  # Restart CMA-ES
                self.stagnation_counter = 0  # Reset counter

        return self.f_opt, self.x_opt