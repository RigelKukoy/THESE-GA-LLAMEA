import numpy as np

class CoordinateAdaptiveCMAES:
    def __init__(self, budget=10000, dim=10, popsize=None, sigma0=0.5):
        self.budget = budget
        self.dim = dim
        self.popsize = popsize if popsize else 4 + int(3 * np.log(self.dim))
        self.sigma = sigma0
        self.mean = None
        self.coordinate_sigmas = None  # Individual learning rates for each dimension
        self.f_opt = np.inf
        self.x_opt = None
        self.eval_count = 0
        self.success_history = None

    def initialize(self, func):
        self.mean = np.random.uniform(func.bounds.lb, func.bounds.ub, size=self.dim)
        self.coordinate_sigmas = np.ones(self.dim) * self.sigma
        self.f_opt = np.inf
        self.x_opt = None
        self.success_history = np.zeros(self.dim)

    def sample_population(self):
        z = np.random.normal(0, 1, size=(self.popsize, self.dim))
        x = self.mean + self.coordinate_sigmas * z
        return x

    def __call__(self, func):
        self.initialize(func)
        mu = self.popsize // 2

        while self.eval_count < self.budget:
            # Sample population
            x = self.sample_population()

            # Clip to boundaries
            lb = func.bounds.lb
            ub = func.bounds.ub
            x = np.clip(x, lb, ub)

            # Evaluate population
            fitness = np.array([func(xi) for xi in x])
            self.eval_count += self.popsize

            # Sort population
            idx = np.argsort(fitness)
            fitness = fitness[idx]
            x = x[idx]

            # Update optimal solution
            if fitness[0] < self.f_opt:
                self.f_opt = fitness[0]
                self.x_opt = x[0]

            # Update mean
            xmean = np.mean(x[:mu], axis=0)
            delta_mean = xmean - self.mean
            self.mean = xmean

            # Update coordinate-wise learning rates based on success
            for i in range(self.dim):
                if np.abs(delta_mean[i]) > 1e-8: # prevent division by zero
                    success_rate = np.mean((x[:mu, i] - self.mean[i]) * delta_mean[i] > 0) # Measure the success of updates along each coordinate.

                    if success_rate > 0.3:
                        self.coordinate_sigmas[i] *= 1.05  # Increase learning rate if successful
                        self.success_history[i] +=1
                    else:
                        self.coordinate_sigmas[i] *= 0.95  # Decrease learning rate if not successful
                        self.success_history[i] = max(0, self.success_history[i] -1) # Reduce the number of past successes
                self.coordinate_sigmas[i] = np.clip(self.coordinate_sigmas[i], self.sigma/10, self.sigma*10) # Keep the coordinate sigmas in a reasonable range

            if self.eval_count >= self.budget:
                break

        return self.f_opt, self.x_opt