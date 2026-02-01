import numpy as np

class CMAES_Restart:
    def __init__(self, budget=10000, dim=10, pop_size=None, initial_sigma=0.5, restart_trigger=1e-12, adaptation_rate = 0.1):
        self.budget = budget
        self.dim = dim
        self.initial_sigma = initial_sigma
        self.restart_trigger = restart_trigger
        self.adaptation_rate = adaptation_rate
        self.pop_size = pop_size if pop_size is not None else 4 + int(3 * np.log(self.dim))  # Default population size

        self.mean = None
        self.sigma = None
        self.C = None
        self.pc = None
        self.ps = None
        self.chiN = None
        self.eigen_updated = False
        self.B = None
        self.D = None

        self.f_opt = np.Inf
        self.x_opt = None
        self.eval_count = 0
        self.generation = 0

    def initialize(self, func):
        self.mean = np.random.uniform(func.bounds.lb, func.bounds.ub, size=self.dim)
        self.sigma = self.initial_sigma
        self.C = np.eye(self.dim)
        self.pc = np.zeros(self.dim)
        self.ps = np.zeros(self.dim)
        self.chiN = np.sqrt(self.dim) * (1 - 1/(4*self.dim) + 1/(21*self.dim**2))
        self.eigen_updated = False

    def sample_population(self):
        z = np.random.normal(0, 1, size=(self.pop_size, self.dim))
        y = np.dot(z, np.transpose(self.B * self.D)) if self.eigen_updated else z * np.sqrt(np.diag(self.C))
        x = self.mean + self.sigma * y
        return x, z

    def update_distribution(self, x, z, fitness):
        idx = np.argsort(fitness)
        x_sorted = x[idx]
        z_sorted = z[idx]

        weights = np.log(self.pop_size + 1) - np.log(np.arange(1, self.pop_size + 1))
        weights = weights / np.sum(weights)

        # Update mean
        mean_old = self.mean.copy()
        self.mean = np.sum(x_sorted[:self.pop_size] * weights[:, np.newaxis], axis=0)

        # Update evolution path for covariance matrix
        y = self.mean - mean_old
        self.ps = (1 - self.adaptation_rate) * self.ps + np.sqrt(self.adaptation_rate * (2 - self.adaptation_rate)) * (np.dot(y, np.linalg.inv(self.C)) / self.sigma if self.eigen_updated else y / self.sigma) #np.linalg.solve
        hsig = np.linalg.norm(self.ps) / np.sqrt(1 - (1 - self.adaptation_rate)**(2*(self.eval_count//self.pop_size))) < (1.4 + 2/(self.dim+1))*self.chiN
        self.pc = (1 - self.adaptation_rate) * self.pc + hsig * np.sqrt(self.adaptation_rate * (2 - self.adaptation_rate)) * y / self.sigma

        # Update covariance matrix
        delta = (1-hsig) * self.adaptation_rate * (2-self.adaptation_rate)
        self.C = (1 - self.adaptation_rate) * self.C + self.adaptation_rate * (self.pc[:, np.newaxis] @ self.pc[np.newaxis, :]) + delta * self.adaptation_rate * (2-self.adaptation_rate) * self.C
        self.C = (1 - self.adaptation_rate) * self.C + self.adaptation_rate * np.sum(weights[:, np.newaxis, np.newaxis] * (y[:, np.newaxis] @ y[np.newaxis, :]), axis=0)

        # Update step size
        self.sigma *= np.exp((self.adaptation_rate/0.8) * (np.linalg.norm(self.ps)/self.chiN - 1))

        # Eigen decomposition update
        if self.eval_count // self.pop_size % (1 + int(30*self.dim/self.pop_size)) == 0: # Adaptive frequency
             try:
                  self.D, self.B = np.linalg.eig(self.C)
                  self.D = np.sqrt(np.abs(self.D))
                  self.eigen_updated = True
             except np.linalg.LinAlgError:
                  self.C = np.eye(self.dim)
                  self.eigen_updated = False

    def detect_stagnation(self):
          return self.sigma < self.restart_trigger

    def __call__(self, func):
        self.initialize(func)
        while self.budget > 0:
            x, z = self.sample_population()
            x = np.clip(x, func.bounds.lb, func.bounds.ub)
            fitness = np.array([func(xi) for xi in x])
            self.budget -= self.pop_size
            self.eval_count += self.pop_size

            if np.min(fitness) < self.f_opt:
                self.f_opt = np.min(fitness)
                self.x_opt = x[np.argmin(fitness)]

            self.update_distribution(x, z, fitness)
            self.generation += 1

            if self.detect_stagnation():
                self.initialize(func) # Restart
                self.pop_size = min(self.pop_size + 10, 200)
                print(f"Restarting CMA-ES, new pop_size = {self.pop_size}")

            if self.budget <= 0:
                  break
        return self.f_opt, self.x_opt