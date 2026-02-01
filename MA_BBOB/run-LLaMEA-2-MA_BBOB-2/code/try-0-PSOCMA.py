import numpy as np

class PSOCMA:
    def __init__(self, budget=10000, dim=10, popsize=None, w=0.7, c1=1.5, c2=1.5):
        self.budget = budget
        self.dim = dim
        self.popsize = popsize if popsize is not None else 4 + int(3 * np.log(self.dim))
        self.w = w  # Inertia weight
        self.c1 = c1 # Cognitive coefficient
        self.c2 = c2 # Social coefficient
        self.particles = np.random.uniform(-5, 5, size=(self.popsize, self.dim))
        self.velocities = np.random.uniform(-1, 1, size=(self.popsize, self.dim))
        self.personal_best_positions = self.particles.copy()
        self.personal_best_fitness = np.full(self.popsize, np.inf)
        self.global_best_position = None
        self.global_best_fitness = np.inf

        # CMA-ES related parameters
        self.m = np.zeros(self.dim)  # Mean
        self.sigma = 0.5  # Step size
        self.C = np.eye(self.dim)  # Covariance matrix
        self.p_sigma = np.zeros(self.dim)  # Evolution path for sigma
        self.p_c = np.zeros(self.dim)  # Evolution path for C
        self.mu = self.popsize // 2
        self.weights = np.log(self.mu + 1/2) - np.log(np.arange(1, self.mu + 1))
        self.weights = self.weights / np.sum(self.weights)
        self.chiN = np.sqrt(self.dim) * (1 - 1/(4*self.dim) + 1/(21*self.dim**2))
        self.c_sigma = (self.mu + 2) / (self.dim + self.mu + 5)
        self.c_c = (4 + self.mu/self.dim) / (self.dim + 4 + 2*self.mu/self.dim)
        self.c_1 = 2 / ((self.dim + 1.3)**2 + self.mu)
        self.c_mu = min(1 - self.c_1, 2 * (self.mu - 2 + 1/self.mu) / ((self.dim + 2)**2 + self.mu))
        self.d_sigma = 1 + 2*max(0, np.sqrt((self.mu-1)/(self.dim+1)) - 1) + self.c_sigma
        self.D = None
        self.B = None
        self.used_budget = 0


    def __call__(self, func):
        self.f_opt = np.Inf
        self.x_opt = None

        while self.used_budget < self.budget:
            # Evaluate particles
            fitness = np.zeros(self.popsize)
            for i in range(self.popsize):
                if self.used_budget < self.budget:
                    fitness[i] = func(self.particles[i])
                    self.used_budget += 1
                else:
                    fitness[i] = np.inf

                if fitness[i] < self.personal_best_fitness[i]:
                    self.personal_best_fitness[i] = fitness[i]
                    self.personal_best_positions[i] = self.particles[i].copy()

                if fitness[i] < self.global_best_fitness:
                    self.global_best_fitness = fitness[i]
                    self.global_best_position = self.particles[i].copy()
                    self.f_opt = self.global_best_fitness
                    self.x_opt = self.global_best_position

            # Update velocities and positions (PSO)
            for i in range(self.popsize):
                r1 = np.random.rand(self.dim)
                r2 = np.random.rand(self.dim)
                self.velocities[i] = (self.w * self.velocities[i]
                                      + self.c1 * r1 * (self.personal_best_positions[i] - self.particles[i])
                                      + self.c2 * r2 * (self.global_best_position - self.particles[i]))
                self.particles[i] = np.clip(self.particles[i] + self.velocities[i], func.bounds.lb, func.bounds.ub)

            # CMA-ES update
            idx = np.argsort(fitness)
            sorted_particles = self.particles[idx]

            m_old = self.m.copy()
            self.m = np.sum(self.weights[:, None] * sorted_particles[:self.mu], axis=0)

            # Sample population using CMA-ES strategy for generating search directions
            z = np.random.normal(0, 1, size=(self.popsize, self.dim))
            if self.D is None or self.B is None:
                self.D, self.B = np.linalg.eigh(self.C)
                self.D = np.sqrt(np.maximum(self.D, 1e-16))
            y = self.B @ np.diag(self.D) @ z.T
            # Update particles using the CMA-ES directions, instead of purely random PSO updates
            self.particles = self.m + self.sigma * y.T
            self.particles = np.clip(self.particles, func.bounds.lb, func.bounds.ub) # ensure bounds

            y_mean = np.mean(z[:self.mu], axis=0)
            self.p_sigma = (1 - self.c_sigma) * self.p_sigma + np.sqrt(self.c_sigma * (2 - self.c_sigma)) * (self.B @ y_mean)
            hsig = np.linalg.norm(self.p_sigma) / np.sqrt(1 - (1 - self.c_sigma)**(2*self.used_budget/self.popsize))/self.chiN < 1.4 + 2/(self.dim+1)
            self.p_c = (1 - self.c_c) * self.p_c + hsig * np.sqrt(self.c_c * (2 - self.c_c)) * (self.m - m_old) / self.sigma
            self.C = (1 - self.c_1 - self.c_mu) * self.C + self.c_1 * (self.p_c[:, None] @ self.p_c[None, :]) + self.c_mu * np.sum(self.weights[:, None, None] * (z[:self.mu, :, None] @ z[:self.mu, None, :]), axis=0)
            self.sigma *= np.exp((self.c_sigma / self.d_sigma) * (np.linalg.norm(self.p_sigma)/self.chiN - 1))
            self.D = None # invalidate cached B and D

        return self.f_opt, self.x_opt