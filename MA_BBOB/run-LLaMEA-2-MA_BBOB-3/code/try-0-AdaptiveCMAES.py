import numpy as np

class AdaptiveCMAES:
    def __init__(self, budget=10000, dim=10, initial_popsize=None, cs=0.3, damps=None, c_cov_base=None, sigma0=0.2, archive_size=10):
        self.budget = budget
        self.dim = dim
        self.sigma0 = sigma0
        self.initial_popsize = initial_popsize if initial_popsize is not None else 4 + int(3 * np.log(dim)) # Default popsize
        self.popsize = self.initial_popsize
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
        self.archive_size = archive_size
        self.archive_x = []
        self.archive_f = []
        self.stagnation_counter = 0
        self.stagnation_threshold = 50
        self.success_rate = 0.0
        self.success_history = []
        self.success_window = 10 # Number of iterations to track success
        self.popsize_adaptation_rate = 0.1 # Rate at which popsize is adjusted

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

    def adapt_popsize(self):
        if len(self.success_history) < self.success_window:
            return
        
        recent_success_rate = np.mean(self.success_history[-self.success_window:])
        
        if recent_success_rate > 0.2:
            self.popsize = max(4, int(self.popsize * (1 - self.popsize_adaptation_rate))) # Reduce popsize if success rate is high, but ensure at least 4
            self.mu = self.popsize // 2

            # Recompute weights
            self.weights = np.log(self.mu + 1/2) - np.log(np.arange(1, self.mu + 1))
            self.weights = self.weights / np.sum(self.weights)
        elif recent_success_rate < 0.05:
            self.popsize = int(self.popsize * (1 + self.popsize_adaptation_rate))  # Increase popsize if success rate is low
            self.mu = self.popsize // 2

            # Recompute weights
            self.weights = np.log(self.mu + 1/2) - np.log(np.arange(1, self.mu + 1))
            self.weights = self.weights / np.sum(self.weights)

    def restart(self):
        if self.archive_x:
            # Select a point from archive to guide the restart
            idx = np.argmin(self.archive_f)
            best_x_from_archive = self.archive_x[idx]
            self.m = best_x_from_archive + np.random.normal(0, 0.1, size=self.dim)
        else:
            # If archive is empty, restart randomly
            self.m = np.random.uniform(-2, 2, size=self.dim)
        
        self.sigma = self.sigma0
        self.C = np.eye(self.dim)
        self.pc = np.zeros(self.dim)
        self.ps = np.zeros(self.dim)
        self.eigenspace = np.eye(self.dim)
        self.eigenvalues = np.ones(self.dim)

    def __call__(self, func):
        self.initialize()
        improved = False # Flag to track if any improvement was made in an iteration.

        while self.func_evals < self.budget:
            x = self.sample()

            fitness_values = np.array([func(x[:, i]) for i in range(self.popsize)])
            self.func_evals += self.popsize

            best_index = np.argmin(fitness_values)
            if fitness_values[best_index] < self.f_opt:
                self.f_opt = fitness_values[best_index]
                self.x_opt = x[:, best_index]
                improved = True

                # Update archive
                if len(self.archive_x) < self.archive_size:
                    self.archive_x.append(self.x_opt)
                    self.archive_f.append(self.f_opt)
                else:
                    worst_arch_idx = np.argmax(self.archive_f)
                    if self.f_opt < self.archive_f[worst_arch_idx]:
                        self.archive_x[worst_arch_idx] = self.x_opt
                        self.archive_f[worst_arch_idx] = self.f_opt
                self.stagnation_counter = 0  # Reset counter if improvement
            else:
                improved = False
            
            self.update(x, fitness_values)
            self.stagnation_counter += 1

            # Popsize adaptation and success rate tracking
            self.success_history.append(1 if improved else 0)
            if len(self.success_history) > 2 * self.success_window:
                 self.success_history.pop(0) # Keep the history at a manageable size
            self.adapt_popsize()
                

            if self.stagnation_counter > self.stagnation_threshold:
                self.restart()
                self.stagnation_counter = 0  # Reset counter

        return self.f_opt, self.x_opt