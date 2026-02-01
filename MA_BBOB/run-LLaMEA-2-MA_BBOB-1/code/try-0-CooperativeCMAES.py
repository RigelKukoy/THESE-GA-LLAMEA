import numpy as np

class CooperativeCMAES:
    def __init__(self, budget=10000, dim=10, num_groups=2, initial_sigma=0.5, mu_factor=0.25):
        self.budget = budget
        self.dim = dim
        self.num_groups = num_groups
        self.group_size = dim // num_groups  # Number of dimensions per group
        self.initial_sigma = initial_sigma
        self.mu_factor = mu_factor
        self.groups = [list(range(i * self.group_size, (i + 1) * self.group_size)) for i in range(num_groups)]
        remaining_dims = dim % num_groups
        for i in range(remaining_dims):
            self.groups[i].append(num_groups * self.group_size + i)

        self.cmaes_optimizers = [CMAES(budget=budget, dim=len(group), initial_sigma=initial_sigma, mu_factor=mu_factor) for group in self.groups]
        self.f_opt = np.Inf
        self.x_opt = None
        self.evals = 0

    def __call__(self, func):
        while self.evals < self.budget:
            x = np.zeros(self.dim)
            group_solutions = []
            group_fitnesses = []
            
            # Optimize each group using CMA-ES
            for i, group in enumerate(self.groups):
                cmaes = self.cmaes_optimizers[i]
                f_group, x_group = cmaes(SubFunction(func, group))
                
                group_solutions.append(x_group)
                group_fitnesses.append(f_group)

                x[group] = x_group

            f = func(x)
            self.evals += 1
            if f < self.f_opt:
                self.f_opt = f
                self.x_opt = x
        return self.f_opt, self.x_opt

class SubFunction:
    def __init__(self, func, dims):
        self.func = func
        self.dims = dims
        self.bounds = func.bounds
        self.lb = func.bounds.lb
        self.ub = func.bounds.ub

    def __call__(self, x_sub):
        x = np.zeros(self.func.bounds.ub.shape[0])
        x[:] = np.nan
        for i, idx in enumerate(self.dims):
            x[idx] = x_sub[i]
        
        if np.any(np.isnan(x)):
            raise ValueError("x contains NaN values. This should not happen.")
            
        return self.func(x)

import numpy as np

class CMAES:
    def __init__(self, budget=10000, dim=10, mu_factor=0.25, initial_sigma=0.5):
        self.budget = budget
        self.dim = dim
        self.mu = int(dim * mu_factor)  # Number of parents
        self.weights = np.log(self.mu + 0.5) - np.log(np.arange(1, self.mu + 1))
        self.weights /= np.sum(self.weights)
        self.mueff = np.sum(self.weights)**2 / np.sum(self.weights**2)
        self.cc = (4 + self.mueff / self.dim) / (self.dim + 4 + 2 * self.mueff / self.dim)
        self.cs = (self.mueff + 2) / (self.dim + self.mueff + 5)
        self.c1 = 2 / ((self.dim + 1.3)**2 + self.mueff)
        self.cmu = min(1 - self.c1, 2 * (self.mueff - 2 + 1 / self.mueff) / ((self.dim + 2)**2 + self.mueff))
        self.damps = 1 + 2 * max(0, np.sqrt((self.mueff - 1) / (self.dim + 1)) - 1) + self.cs
        self.mean = np.zeros(dim)
        self.sigma = initial_sigma
        self.pc = np.zeros(dim)
        self.ps = np.zeros(dim)
        self.C = np.eye(dim)
        self.B = np.eye(dim)
        self.D = np.ones(dim)
        self.C_eigen_age = 0
        self.f_opt = np.Inf
        self.x_opt = None
        self.evals = 0

    def sample_population(self, popsize, func):
        z = np.random.normal(0, 1, size=(popsize, self.dim))
        x = self.mean + self.sigma * (self.B @ (self.D * z).T).T

        # Handle boundary constraints using clipping
        x = np.clip(x, func.bounds.lb, func.bounds.ub)
        
        f = np.array([func(xi) for xi in x])
        return x, f, z

    def update_distribution(self, x, f, z, popsize):
        idx = np.argsort(f)
        x = x[idx]
        z = z[idx]
        
        x_mu = x[:self.mu]
        z_mu = z[:self.mu]
        
        self.mean = np.sum(self.weights.reshape(-1, 1) * x_mu, axis=0)

        zmean = np.sum(self.weights.reshape(-1, 1) * z_mu, axis=0)
        self.ps = (1 - self.cs) * self.ps + np.sqrt(self.cs * (2 - self.cs) * self.mueff) * (self.B @ zmean)
        hsig = np.linalg.norm(self.ps) / np.sqrt(1 - (1 - self.cs)**(2 * (self.budget // popsize))) < 1.4 + 2 / (self.dim + 1)
        self.pc = (1 - self.cc) * self.pc + hsig * np.sqrt(self.cc * (2 - self.cc) * self.mueff) * (self.mean - self.mean) / self.sigma 

        artmp = (1 / self.sigma) * (x_mu - self.mean).T
        self.C = (1 - self.c1 - self.cmu) * self.C + self.c1 * (self.pc.reshape(-1, 1) @ self.pc.reshape(1, -1) + (1-hsig) * self.cc * (2-self.cc) * self.C) + self.cmu * artmp @ np.diag(self.weights) @ artmp.T

        self.sigma *= np.exp((self.cs / self.damps) * (np.linalg.norm(self.ps) / np.sqrt(1 - (1 - self.cs)**(2 * (self.budget // popsize))) / np.sqrt(self.dim) - 1))

    def __call__(self, func):
        popsize = 4 + int(3 * np.log(self.dim))

        while self.evals < self.budget:
            x, f, z = self.sample_population(popsize, func)
            self.evals += popsize

            for i in range(popsize):
                if f[i] < self.f_opt:
                    self.f_opt = f[i]
                    self.x_opt = x[i]
                    
            self.update_distribution(x, f, z, popsize)
            self.C_eigen_age += 1
            if self.C_eigen_age > self.budget // (10 * popsize): # Re-compute eigenvalue decomposition after a while
                self.C_eigen_age = 0
                self.C = np.triu(self.C) + np.triu(self.C, 1).T
                self.D, self.B = np.linalg.eigh(self.C)
                self.D = np.sqrt(self.D)
                
                # Avoid tiny values
                self.D[self.D < 1e-10] = 1e-10

        return self.f_opt, self.x_opt