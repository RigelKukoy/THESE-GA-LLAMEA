import numpy as np

class ActiveCMAES:
    def __init__(self, budget=10000, dim=10, popsize=None, cs=0.3, damp=None, c_cov=None, archive_size=100, stagnation_threshold=100):
        self.budget = budget
        self.dim = dim
        self.popsize = popsize if popsize is not None else 4 + int(3 * np.log(self.dim))
        self.mu = self.popsize // 2
        self.weights = np.log(self.mu + 1/2) - np.log(np.arange(1, self.mu + 1))
        self.weights = self.weights / np.sum(self.weights)
        self.mueff = np.sum(self.weights)**2 / np.sum(self.weights**2)

        self.cs = cs
        self.damps = damp if damp is not None else 1 + 2 * max(0, np.sqrt((self.mueff - 1) / (self.dim + 1)) - 1) + self.cs
        self.ccov1 = c_cov[0] if c_cov is not None else 2 / ((self.dim + 1.3)**2 + self.mueff)
        self.ccovmu = c_cov[1] if c_cov is not None else 2 * (self.mueff - 2 + 1/self.mueff) / ((self.dim + 2.3)**2 + self.mueff)
        self.ccovmu = min(1-self.ccov1, self.ccovmu)

        self.mean = None
        self.sigma = 0.5
        self.C = None
        self.pc = None
        self.ps = None
        self.archive = []
        self.archive_size = archive_size
        self.f_opt = np.inf
        self.x_opt = None
        self.eval_count = 0
        self.stagnation_counter = 0
        self.stagnation_threshold = stagnation_threshold
        self.last_f_opt = np.inf

    def initialize(self):
        self.mean = np.random.uniform(-5, 5, size=self.dim)
        self.C = np.eye(self.dim)
        self.pc = np.zeros(self.dim)
        self.ps = np.zeros(self.dim)

    def sample(self):
        z = np.random.normal(0, 1, size=(self.dim, self.popsize))
        C_sqrt = np.linalg.cholesky(self.C)
        x = self.mean[:, np.newaxis] + self.sigma * C_sqrt @ z
        return x.T

    def update(self, x, f):
        idx = np.argsort(f)
        x = x[idx]
        f = f[idx]

        xmean = np.sum(x[:self.mu] * self.weights[:, np.newaxis], axis=0)

        self.ps = (1 - self.cs) * self.ps + np.sqrt(self.cs * (2 - self.cs) * self.mueff) * (xmean - self.mean) / self.sigma
        hsig = np.linalg.norm(self.ps) / np.sqrt(1 - (1 - self.cs)**(2 * (self.eval_count / self.popsize))) < (1.4 + 2 / (self.dim + 1))
        self.pc = (1 - 1) * self.pc + hsig * np.sqrt(1 * (2 - 1) * self.mueff) * (xmean - self.mean) / self.sigma

        y = (x[:self.mu] - self.mean) / self.sigma

        # Active CMA update
        B = np.linalg.cholesky(self.C)
        z = np.linalg.solve(B, (x[:self.mu] - self.mean).T / self.sigma)
        w_mask = self.weights > 0
        
        C_temp = (1 - self.ccov1 - self.ccovmu + self.ccov1 * (1 - hsig**2)) * self.C \
                 + self.ccov1 * np.outer(self.pc, self.pc) \
                 + self.ccovmu * np.sum(self.weights[:, np.newaxis, np.newaxis] * z[:, w_mask][:, :, np.newaxis] * z[:, w_mask][:, np.newaxis, :], axis=1)

        self.C = np.triu(C_temp) + np.triu(C_temp, 1).T
        self.C = self.C + 1e-8 * np.eye(self.dim)
        
        self.sigma *= np.exp((self.cs / self.damps) * (np.linalg.norm(self.ps) / np.sqrt(self.dim) - 1))
        self.mean = xmean

        for i in range(len(x)):
            if f[i] < self.f_opt:
                self.f_opt = f[i]
                self.x_opt = x[i]

    def archive_management(self, x, f):
        for i in range(len(x)):
            if len(self.archive) < self.archive_size:
                self.archive.append((x[i], f[i]))
            else:
                # Replace worst element in archive if current element is better
                worst_idx = np.argmax([item[1] for item in self.archive])
                if f[i] < self.archive[worst_idx][1]:
                    self.archive[worst_idx] = (x[i], f[i])

    def check_stagnation(self):
        if self.f_opt < self.last_f_opt:
            self.stagnation_counter = 0
            self.last_f_opt = self.f_opt
        else:
            self.stagnation_counter += 1

        if self.stagnation_counter > self.stagnation_threshold:
            return True
        else:
            return False

    def adjust_popsize(self):
        if self.check_stagnation():
            self.popsize = int(self.popsize * 1.5)  # Increase population size
            self.mu = self.popsize // 2
            self.weights = np.log(self.mu + 1/2) - np.log(np.arange(1, self.mu + 1))
            self.weights = self.weights / np.sum(self.weights)
            self.mueff = np.sum(self.weights)**2 / np.sum(self.weights**2)
            self.stagnation_counter = 0  # Reset stagnation counter
        elif self.popsize > 4 + int(3 * np.log(self.dim)) and self.eval_count > self.budget/2 and self.stagnation_counter < self.stagnation_threshold/2:
             self.popsize = int(self.popsize * 0.9) # Decrease popsize when nearing budget end and not stagnating
             self.mu = self.popsize // 2
             self.weights = np.log(self.mu + 1/2) - np.log(np.arange(1, self.mu + 1))
             self.weights = self.weights / np.sum(self.weights)
             self.mueff = np.sum(self.weights)**2 / np.sum(self.weights**2)

    def __call__(self, func):
        self.initialize()

        while self.eval_count < self.budget:
            self.adjust_popsize()
            x = self.sample()
            
            # Ensure bounds
            x = np.clip(x, func.bounds.lb, func.bounds.ub)
            
            f = np.array([func(xi) for xi in x])
            self.eval_count += len(x)

            self.update(x, f)
            self.archive_management(x, f)

        return self.f_opt, self.x_opt