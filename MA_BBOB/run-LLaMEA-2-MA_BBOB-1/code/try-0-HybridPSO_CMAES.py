import numpy as np

class HybridPSO_CMAES:
    def __init__(self, budget=10000, dim=10, pso_fraction=0.5, cmaes_fraction=0.5, initial_inertia=0.7, initial_cognitive=1.5, initial_social=1.5):
        self.budget = budget
        self.dim = dim
        self.pso_fraction = pso_fraction
        self.cmaes_fraction = cmaes_fraction
        self.inertia = initial_inertia
        self.cognitive = initial_cognitive
        self.social = initial_social
        self.particles = np.random.uniform(-5, 5, size=(dim * 5, dim))
        self.velocities = np.random.uniform(-1, 1, size=(dim * 5, dim))
        self.personal_best_positions = self.particles.copy()
        self.personal_best_values = np.full(dim * 5, np.inf)
        self.global_best_position = None
        self.global_best_value = np.inf

        # CMA-ES parameters
        self.mu = int(dim * 0.25)
        self.weights = np.log(self.mu + 0.5) - np.log(np.arange(1, self.mu + 1))
        self.weights /= np.sum(self.weights)
        self.mueff = np.sum(self.weights)**2 / np.sum(self.weights**2)
        self.cc = (4 + self.mueff / self.dim) / (self.dim + 4 + 2 * self.mueff / self.dim)
        self.cs = (self.mueff + 2) / (self.dim + self.mueff + 5)
        self.c1 = 2 / ((self.dim + 1.3)**2 + self.mueff)
        self.cmu = min(1 - self.c1, 2 * (self.mueff - 2 + 1 / self.mueff) / ((self.dim + 2)**2 + self.mueff))
        self.damps = 1 + 2 * max(0, np.sqrt((self.mueff - 1) / (self.dim + 1)) - 1) + self.cs
        self.mean = np.zeros(dim)
        self.sigma = 0.5
        self.pc = np.zeros(dim)
        self.ps = np.zeros(dim)
        self.C = np.eye(dim)
        self.B = np.eye(dim)
        self.D = np.ones(dim)
        self.C_eigen_age = 0
        self.pso_phase = True # Start with PSO

        self.f_opt = np.Inf
        self.x_opt = None

    def pso_step(self, func):
        for i in range(self.particles.shape[0]):
            f = func(self.particles[i])
            if f < self.personal_best_values[i]:
                self.personal_best_values[i] = f
                self.personal_best_positions[i] = self.particles[i].copy()

            if f < self.global_best_value:
                self.global_best_value = f
                self.global_best_position = self.particles[i].copy()
                self.f_opt = f
                self.x_opt = self.particles[i]

        for i in range(self.particles.shape[0]):
            r1 = np.random.rand(self.dim)
            r2 = np.random.rand(self.dim)
            self.velocities[i] = (self.inertia * self.velocities[i] +
                                  self.cognitive * r1 * (self.personal_best_positions[i] - self.particles[i]) +
                                  self.social * r2 * (self.global_best_position - self.particles[i]))
            self.particles[i] = np.clip(self.particles[i] + self.velocities[i], func.bounds.lb, func.bounds.ub)

    def cmaes_sample_population(self, popsize, func):
        z = np.random.normal(0, 1, size=(popsize, self.dim))
        x = self.mean + self.sigma * (self.B @ (self.D * z).T).T
        x = np.clip(x, func.bounds.lb, func.bounds.ub)
        f = np.array([func(xi) for xi in x])
        return x, f, z

    def cmaes_update_distribution(self, x, f, z, popsize, func):
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

    def cmaes_step(self, func):
        popsize = 4 + int(3 * np.log(self.dim))
        x, f, z = self.cmaes_sample_population(popsize, func)

        for i in range(popsize):
            if f[i] < self.f_opt:
                self.f_opt = f[i]
                self.x_opt = x[i]

        self.cmaes_update_distribution(x, f, z, popsize, func)
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

    def __call__(self, func):
        evals = 0
        switch_interval = self.budget // 20 # Switch every 5% of the budget
        last_switch = 0

        while evals < self.budget:
            if self.pso_phase:
                for _ in range(min(switch_interval, self.budget - evals)):
                    self.pso_step(func)
                    evals += self.particles.shape[0]
            else:
                popsize = 4 + int(3 * np.log(self.dim))
                num_cmaes_evals = 0
                while num_cmaes_evals < min(switch_interval, self.budget - evals):
                    self.cmaes_step(func)
                    num_cmaes_evals += popsize
                    evals+=popsize


            if evals - last_switch >= switch_interval:
                 # Simple switching mechanism: alternate between PSO and CMA-ES
                self.pso_phase = not self.pso_phase
                last_switch = evals

        return self.f_opt, self.x_opt