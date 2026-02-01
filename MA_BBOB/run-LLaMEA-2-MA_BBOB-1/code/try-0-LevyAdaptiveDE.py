import numpy as np

class LevyAdaptiveDE:
    def __init__(self, budget=10000, dim=10, pop_size=50, F=0.5, CR=0.7, levy_exponent=1.5, local_search_prob=0.1):
        self.budget = budget
        self.dim = dim
        self.pop_size = pop_size
        self.F = F  # Initial mutation factor
        self.CR = CR  # Initial crossover rate
        self.levy_exponent = levy_exponent # Exponent for Levy flight
        self.local_search_prob = local_search_prob # Probability of performing local search
        self.archive_size = int(self.pop_size * 0.2)  # Archive size
        self.archive = []
        self.success_F = []
        self.success_CR = []
        self.memory_size = 10

    def levy_flight(self, size, exponent):
        num = np.random.randn(size) * np.sqrt(
            np.math.gamma(1 + exponent) * np.sin(np.pi * exponent / 2) / (np.math.gamma((1 + exponent) / 2) * exponent * (2 ** ((exponent - 1) / 2))))
        den = np.abs(np.random.randn(size)) ** (1 / exponent)
        return num / den

    def local_search(self, x, func, bounds):
        # Perform a simple local search around x
        delta = np.random.uniform(-0.1, 0.1, size=self.dim)  # Small random perturbation
        x_new = x + delta
        x_new = np.clip(x_new, bounds.lb, bounds.ub)  # Keep within bounds
        f_new = func(x_new)
        self.budget -= 1
        if f_new < func(x):
            return x_new, f_new
        else:
            return x, func(x)

    def __call__(self, func):
        # Initialization
        population = np.random.uniform(func.bounds.lb, func.bounds.ub, size=(self.pop_size, self.dim))
        fitness = np.array([func(x) for x in population])
        self.budget -= self.pop_size

        best_index = np.argmin(fitness)
        self.f_opt = fitness[best_index]
        self.x_opt = population[best_index]

        # Evolution loop
        while self.budget > 0:
            for i in range(self.pop_size):
                # Mutation
                indices = np.random.choice(self.pop_size, 3, replace=False)
                x_r1, x_r2, x_r3 = population[indices]

                # Adaptive F using success history
                if self.success_F:
                    F_current = np.mean(self.success_F)
                else:
                    F_current = self.F
                F_current = np.clip(F_current, 0.1, 1.0)

                # Levy flight mutation
                levy_steps = self.levy_flight(self.dim, self.levy_exponent)
                v = x_r1 + F_current * (x_r2 - x_r3) + 0.01 * levy_steps * (self.x_opt - population[i])
                v = np.clip(v, func.bounds.lb, func.bounds.ub)

                # Crossover
                j_rand = np.random.randint(self.dim)
                u = np.copy(population[i])
                for j in range(self.dim):
                    if np.random.rand() < self.CR or j == j_rand:
                        u[j] = v[j]

                # Evaluation
                f_u = func(u)
                self.budget -= 1

                # Local search with probability local_search_prob
                if np.random.rand() < self.local_search_prob:
                    u, f_u = self.local_search(u, func, func.bounds)

                if f_u < fitness[i]:
                    # Replacement
                    fitness[i] = f_u
                    population[i] = u

                    # Update archive (if necessary)
                    if len(self.archive) < self.archive_size:
                        self.archive.append(population[i])
                    else:
                        # Replace a random element in the archive
                        replace_index = np.random.randint(self.archive_size)
                        self.archive[replace_index] = population[i]

                    # Store successful F and CR values
                    self.success_F.append(F_current)
                    self.success_CR.append(self.CR)
                    if len(self.success_F) > self.memory_size:
                        self.success_F.pop(0)
                        self.success_CR.pop(0)

                    # Adaptive CR: Adjust CR based on success history
                    if self.success_CR:
                        self.CR = np.mean(self.success_CR)

                    self.CR = self.CR + np.random.normal(0, 0.05)
                    self.CR = np.clip(self.CR, 0.1, 0.9)
                else:
                    # Reduce CR if the trial vector is not better
                    self.CR = self.CR - 0.05  # Decrease CR
                    self.CR = np.clip(self.CR, 0.1, 0.9)

                # Update best solution
                if f_u < self.f_opt:
                    self.f_opt = f_u
                    self.x_opt = u.copy()

        return self.f_opt, self.x_opt