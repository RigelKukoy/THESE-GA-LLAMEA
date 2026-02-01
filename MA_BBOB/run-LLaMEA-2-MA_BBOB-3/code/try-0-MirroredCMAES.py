import numpy as np

class MirroredCMAES:
    def __init__(self, budget=10000, dim=10, popsize_factor=4, sigma0=0.2, c_cov=0.01, c_sigma=0.3, mirrored_ratio=0.5):
        self.budget = budget
        self.dim = dim
        self.sigma0 = sigma0
        self.popsize_factor = popsize_factor
        self.popsize = int(self.popsize_factor * np.log(self.dim)) # Adjust popsize based on dimension
        self.mirrored_ratio = mirrored_ratio #Ratio of mirrored samples

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

        self.c_cov = c_cov
        self.c_sigma = c_sigma

        self.f_opt = np.Inf
        self.x_opt = None
        self.func_evals = 0

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
        
        # Mirrored sampling
        num_mirrored = int(self.popsize * self.mirrored_ratio)
        z_mirrored = -z[:, :num_mirrored]  # Mirror the first few samples
        x_mirrored = self.m[:, np.newaxis] + self.sigma * (self.eigenspace @ (np.diag(np.sqrt(self.eigenvalues)) @ z_mirrored))
        
        x = np.concatenate((x, x_mirrored), axis=1)  # Combine original and mirrored samples
        return x

    def update(self, x, fitness_values):
        # Handle the potentially doubled population size due to mirrored sampling
        popsize = self.popsize * (1 + int(self.mirrored_ratio>0))
        idx = np.argsort(fitness_values)[:self.mu]  # Select only mu best

        x_mu = x[:, idx]
        z_mu = np.linalg.solve(self.eigenspace @ np.diag(np.sqrt(self.eigenvalues)), (x_mu - self.m[:, np.newaxis]) / self.sigma)

        m_old = self.m.copy()
        self.m = np.sum(x_mu * self.weights[np.newaxis, :], axis=1)

        self.ps = (1 - self.c_sigma) * self.ps + np.sqrt(self.c_sigma * (2 - self.c_sigma) * np.sum(self.weights)) * (self.eigenspace @ z_mu.mean(axis=1))
        self.pc = (1 - self.c_cov) * self.pc + np.sqrt(self.c_cov * (2 - self.c_cov) * np.sum(self.weights)) * ((self.m - m_old) / self.sigma)

        C_temp = (1 - self.c_cov) * self.C + self.c_cov * (self.pc[:, np.newaxis] @ self.pc[np.newaxis, :]) + \
                 self.c_cov * np.sum(self.weights * z_mu * z_mu, axis=1).reshape(self.dim, self.dim)

        self.sigma *= np.exp(self.c_sigma / 2 * (np.linalg.norm(self.ps) / np.sqrt(self.dim) - 1))
        self.C = C_temp
        self.eigenvalues, self.eigenspace = np.linalg.eigh(self.C)
        self.eigenvalues = np.maximum(self.eigenvalues, 1e-12)


    def __call__(self, func):
        self.initialize()

        while self.func_evals < self.budget:
            x = self.sample()
            
            #Handle popsize being doubled
            fitness_values = np.array([func(x[:, i]) for i in range(x.shape[1])])
            self.func_evals += x.shape[1]

            best_index = np.argmin(fitness_values)
            if fitness_values[best_index] < self.f_opt:
                self.f_opt = fitness_values[best_index]
                self.x_opt = x[:, best_index]
            
            self.update(x, fitness_values)

        return self.f_opt, self.x_opt