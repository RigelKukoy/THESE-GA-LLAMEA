import numpy as np

class CMA_NM_SA:
    def __init__(self, budget=10000, dim=10, initial_sigma=0.5, mu_factor=0.25, sa_initial_temp=1.0, sa_cooling_rate=0.95):
        self.budget = budget
        self.dim = dim
        self.initial_sigma = initial_sigma
        self.mu = int(dim * mu_factor)
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

        # Nelder-Mead parameters
        self.simplex = None
        self.nm_alpha = 1.0  # Reflection
        self.nm_beta = 0.5   # Contraction
        self.nm_gamma = 2.0   # Expansion

        # Simulated Annealing parameters
        self.sa_temp = sa_initial_temp
        self.sa_cooling_rate = sa_cooling_rate

    def sample_population(self, popsize, func):
        z = np.random.normal(0, 1, size=(popsize, self.dim))
        x = self.mean + self.sigma * (self.B @ (self.D * z).T).T
        x = np.clip(x, func.bounds.lb, func.bounds.ub)
        f = np.array([func(xi) for xi in x])
        return x, f, z

    def update_distribution(self, x, f, z, popsize, func):
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
    
    def initialize_simplex(self, x0, func):
        self.simplex = np.zeros((self.dim + 1, self.dim))
        self.simplex[0] = x0
        for i in range(1, self.dim + 1):
            self.simplex[i] = x0.copy()
            self.simplex[i, i - 1] += 0.1  # Initial step size
            self.simplex[i] = np.clip(self.simplex[i], func.bounds.lb, func.bounds.ub)
        return np.array([func(xi) for xi in self.simplex])


    def nelder_mead_step(self, func):
        f = np.array([func(xi) for xi in self.simplex])
        idx = np.argsort(f)
        self.simplex = self.simplex[idx]
        f = f[idx]
        
        centroid = np.mean(self.simplex[:-1], axis=0)

        # Reflection
        x_r = centroid + self.nm_alpha * (centroid - self.simplex[-1])
        x_r = np.clip(x_r, func.bounds.lb, func.bounds.ub)
        f_r = func(x_r)

        if f_r < f[0]:
            # Expansion
            x_e = centroid + self.nm_gamma * (x_r - centroid)
            x_e = np.clip(x_e, func.bounds.lb, func.bounds.ub)
            f_e = func(x_e)
            if f_e < f_r:
                self.simplex[-1] = x_e
            else:
                self.simplex[-1] = x_r
        elif f_r < f[-2]:
            self.simplex[-1] = x_r
        else:
            # Contraction
            x_c = centroid + self.nm_beta * (self.simplex[-1] - centroid)
            x_c = np.clip(x_c, func.bounds.lb, func.bounds.ub)
            f_c = func(x_c)
            if f_c < f[-1]:
                self.simplex[-1] = x_c
            else:
                # Shrink
                for i in range(1, self.dim + 1):
                    self.simplex[i] = self.simplex[0] + 0.5 * (self.simplex[i] - self.simplex[0])
                    self.simplex[i] = np.clip(self.simplex[i], func.bounds.lb, func.bounds.ub)
        return np.array([func(xi) for xi in self.simplex])

    def simulated_annealing_step(self, x_current, f_current, func):
        x_new = x_current + np.random.normal(0, self.sigma, size=self.dim)  # Use CMA-ES sigma
        x_new = np.clip(x_new, func.bounds.lb, func.bounds.ub)
        f_new = func(x_new)

        if f_new < f_current:
            return x_new, f_new
        else:
            acceptance_probability = np.exp((f_current - f_new) / self.sa_temp)
            if np.random.rand() < acceptance_probability:
                return x_new, f_new
            else:
                return x_current, f_current

    def __call__(self, func):
        popsize = 4 + int(3 * np.log(self.dim))
        evals = 0

        # Initialization with CMA-ES
        self.mean = np.random.uniform(func.bounds.lb, func.bounds.ub, size=self.dim) # Initialize mean

        # Initialize Simplex
        self.simplex = np.zeros((self.dim + 1, self.dim))
        self.simplex[0] = self.mean  # Start Nelder-Mead near CMA-ES mean
        for i in range(1, self.dim + 1):
            self.simplex[i] = self.mean.copy()
            self.simplex[i, i - 1] += 0.1  # Initial step size
            self.simplex[i] = np.clip(self.simplex[i], func.bounds.lb, func.bounds.ub)
        f_simplex = np.array([func(xi) for xi in self.simplex]) # Evaluate simplex
        evals += (self.dim + 1)

        while evals < self.budget:
            # CMA-ES Step
            x, f, z = self.sample_population(popsize, func)
            evals += popsize

            for i in range(popsize):
                if f[i] < self.f_opt:
                    self.f_opt = f[i]
                    self.x_opt = x[i]

            self.update_distribution(x, f, z, popsize, func)
            self.C_eigen_age += 1

            if self.C_eigen_age > self.budget // (10 * popsize):
                self.C_eigen_age = 0
                self.C = np.triu(self.C) + np.triu(self.C, 1).T
                try:
                    self.D, self.B = np.linalg.eigh(self.C)
                    self.D = np.sqrt(self.D)
                    self.D[self.D < 1e-10] = 1e-10
                except np.linalg.LinAlgError:
                    self.C = np.eye(self.dim)
                    self.B = np.eye(self.dim)
                    self.D = np.ones(self.dim)

            # Nelder-Mead Step
            f_simplex = self.nelder_mead_step(func)  # Take a Nelder-Mead step
            evals += (self.dim + 1) #each nm step takes dim+1 evals
           
            #Simulated Annealing
            best_idx = np.argmin(f_simplex)
            x_current = self.simplex[best_idx]
            f_current = f_simplex[best_idx]
            x_new, f_new = self.simulated_annealing_step(x_current, f_current, func)
            evals +=1
            if f_new < f_current:
                self.simplex[best_idx] = x_new
                f_simplex[best_idx] = f_new
            if f_new < self.f_opt:
                self.f_opt = f_new
                self.x_opt = x_new

            self.sa_temp *= self.sa_cooling_rate # Cool SA temperature

        return self.f_opt, self.x_opt