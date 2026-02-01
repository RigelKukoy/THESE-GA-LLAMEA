import numpy as np

class SelfAdaptiveGaussianDE:
    def __init__(self, budget=10000, dim=10, pop_size=50, F_init=0.5, CR_init=0.7, learning_rate=0.1):
        self.budget = budget
        self.dim = dim
        self.pop_size = pop_size
        self.F = np.full(self.pop_size, F_init)
        self.CR = np.full(self.pop_size, CR_init)
        self.learning_rate = learning_rate
        self.population = None
        self.fitness = None
        self.best_index = None
        self.f_opt = np.inf
        self.x_opt = None

    def initialize_population(self, func):
        self.population = np.random.uniform(func.bounds.lb, func.bounds.ub, size=(self.pop_size, self.dim))
        self.fitness = np.array([func(x) for x in self.population])
        self.budget -= self.pop_size
        self.best_index = np.argmin(self.fitness)
        self.f_opt = self.fitness[self.best_index]
        self.x_opt = self.population[self.best_index].copy()

    def __call__(self, func):
        self.initialize_population(func)

        while self.budget > 0:
            for i in range(self.pop_size):
                # Mutation
                indices = np.random.choice(self.pop_size, 3, replace=False)
                x_r1, x_r2, x_r3 = self.population[indices]
                v = self.population[i] + self.F[i] * (x_r1 - x_r2)
                v = np.clip(v, func.bounds.lb, func.bounds.ub)

                # Crossover
                j_rand = np.random.randint(self.dim)
                u = self.population[i].copy()
                for j in range(self.dim):
                    if np.random.rand() < self.CR[i] or j == j_rand:
                        u[j] = v[j]
                
                # Gaussian perturbation
                gaussian_noise = np.random.normal(0, 0.01, size=self.dim)  # Adjust scale
                u = np.clip(u + gaussian_noise, func.bounds.lb, func.bounds.ub)

                # Evaluation
                f_u = func(u)
                self.budget -= 1

                # Selection
                if f_u < self.fitness[i]:
                    # Update parameters based on success
                    delta_f = self.fitness[i] - f_u
                    self.F[i] += self.learning_rate * (1 - self.F[i]) * delta_f  # Adaptive F
                    self.CR[i] += self.learning_rate * (1 - self.CR[i]) * delta_f  # Adaptive CR

                    self.fitness[i] = f_u
                    self.population[i] = u
                    if f_u < self.f_opt:
                        self.f_opt = f_u
                        self.x_opt = u.copy()
                else:
                    # Penalize parameters if unsuccessful
                    self.F[i] -= self.learning_rate * self.F[i] * 0.5  # Reduce F if unsuccessful
                    self.CR[i] -= self.learning_rate * self.CR[i] * 0.5  # Reduce CR if unsuccessful
                    

                self.F[i] = np.clip(self.F[i], 0.1, 1.0)
                self.CR[i] = np.clip(self.CR[i], 0.1, 1.0)

        return self.f_opt, self.x_opt