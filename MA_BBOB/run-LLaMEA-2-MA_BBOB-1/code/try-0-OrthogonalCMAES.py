import numpy as np

class OrthogonalCMAES:
    def __init__(self, budget=10000, dim=10, mu_factor=0.25, initial_sigma=0.5, orthogonal_trials=5):
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
        self.orthogonal_trials = orthogonal_trials

    def sample_population(self, popsize, func):
        z = np.random.normal(0, 1, size=(popsize, self.dim))
        x = self.mean + self.sigma * (self.B @ (self.D * z).T).T
        x = np.clip(x, func.bounds.lb, func.bounds.ub)  # Boundary handling
        f = np.array([func(xi) for xi in x])
        return x, f, z

    def generate_orthogonal_sample(self, x, func):
        """Generates an orthogonal sample around a given point x."""
        orthogonal_points = []
        for _ in range(self.orthogonal_trials):
            direction = np.random.normal(0, 1, self.dim)
            direction /= np.linalg.norm(direction)  # Normalize
            step_size = np.random.uniform(-self.sigma, self.sigma)
            x_new = x + step_size * direction
            x_new = np.clip(x_new, func.bounds.lb, func.bounds.ub)
            orthogonal_points.append((x_new, func(x_new)))
        return orthogonal_points

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
        
        # Orthogonal sampling influence on adaptation
        success_rate = 0
        for i in range(self.mu):
            orthogonal_samples = self.generate_orthogonal_sample(x_mu[i], func)
            best_orthogonal_f = min(sample[1] for sample in orthogonal_samples)
            if best_orthogonal_f < f[i]:
                success_rate += 1

        adaptation_rate_factor = 1.0 + 0.5 * (success_rate / self.mu - 0.5)  # Adjust factor

        self.c1 *= adaptation_rate_factor
        self.cmu *= adaptation_rate_factor


    def __call__(self, func):
        popsize = 4 + int(3 * np.log(self.dim))
        evals = 0

        while evals < self.budget:
            x, f, z = self.sample_population(popsize, func)
            evals += popsize

            for i in range(popsize):
                if f[i] < self.f_opt:
                    self.f_opt = f[i]
                    self.x_opt = x[i]

            self.update_distribution(x, f, z, popsize, func)
            self.C_eigen_age += 1

            if self.C_eigen_age > self.budget // (10 * popsize): # Re-compute eigenvalue decomposition after a while
                self.C_eigen_age = 0
                self.C = np.triu(self.C) + np.triu(self.C, 1).T
                try:
                    self.D, self.B = np.linalg.eigh(self.C)
                    self.D = np.sqrt(self.D)
                    self.D[self.D < 1e-10] = 1e-10  # Avoid tiny values
                except np.linalg.LinAlgError:
                    # Handle non-positive definite matrix
                    self.C = np.eye(self.dim)  # Reset covariance matrix
                    self.B = np.eye(self.dim)
                    self.D = np.ones(self.dim)
                    

        return self.f_opt, self.x_opt