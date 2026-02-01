import numpy as np

class AdaptiveOrthogonalCMAES:
    def __init__(self, budget=10000, dim=10, popsize_factor=4, cs=0.3, damps=None, c_cov_base=None, sigma0=0.2, orthogonal_trials=5):
        self.budget = budget
        self.dim = dim
        self.sigma0 = sigma0
        self.popsize_factor = popsize_factor
        self.popsize = self.calculate_popsize()
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
        self.c_cov_base = c_cov_base if c_cov_base is not None else (1 / (self.dim * np.sqrt(self.dim))) * 10
        self.c_cov = self.c_cov_base
        self.c_cov_mu = self.c_cov
        self.c_cov_mu = self.c_cov_mu if self.c_cov_mu <= 1 else 1
        self.c_cov_mu = self.c_cov_mu if self.c_cov_mu > 0 else 0
        self.c_cov = (1 / self.mu) * self.c_cov

        self.f_opt = np.Inf
        self.x_opt = None
        self.func_evals = 0
        self.success_rate = 0.5  # Initial success rate
        self.success_history = []

        self.orthogonal_trials = orthogonal_trials

    def calculate_popsize(self):
        return int(self.popsize_factor + 3 * np.log(self.dim))

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

    def orthogonal_sampling(self, func):
        # Generate orthogonal directions
        H = np.random.normal(0, 1, size=(self.dim, self.orthogonal_trials))
        Q, _ = np.linalg.qr(H)

        best_f = np.Inf
        best_x = None

        for i in range(self.orthogonal_trials):
            direction = Q[:, i]
            # Sample along this direction
            x_trial = self.m + self.sigma * direction
            f_trial = func(x_trial)
            self.func_evals += 1

            if f_trial < best_f:
                best_f = f_trial
                best_x = x_trial

        return best_f, best_x

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

        self.C = C_temp

        self.eigenvalues, self.eigenspace = np.linalg.eigh(self.C)
        self.eigenvalues = np.maximum(self.eigenvalues, 1e-12)

        # Update success rate and adapt sigma
        improvement = np.min(fitness_values) < self.f_opt
        self.success_history.append(improvement)
        if len(self.success_history) > 10:
            self.success_history.pop(0)
        self.success_rate = np.mean(self.success_history)

        if self.success_rate > 0.6:
            self.sigma *= np.exp(self.cs / self.damps)  # Increase step size
        elif self.success_rate < 0.4:
            self.sigma *= np.exp(-self.cs / self.damps) # Decrease step size
        
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

            # Perform orthogonal sampling
            f_ortho, x_ortho = self.orthogonal_sampling(func)
            if f_ortho < self.f_opt:
                self.f_opt = f_ortho
                self.x_opt = x_ortho

            self.update(x, fitness_values)

        return self.f_opt, self.x_opt