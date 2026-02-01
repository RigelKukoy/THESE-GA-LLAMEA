import numpy as np

class CauchyAdaptiveDE:
    def __init__(self, budget=10000, dim=10, pop_size=50, F=0.5, CR=0.7, cauchy_scale=0.1):
        self.budget = budget
        self.dim = dim
        self.pop_size = pop_size
        self.F = F  # Initial mutation factor
        self.CR = CR  # Initial crossover rate
        self.cauchy_scale = cauchy_scale # Scale parameter for Cauchy distribution
        self.archive_size = int(self.pop_size * 0.2)  # Archive size for storing successful solutions
        self.archive = []
        self.success_F = []
        self.success_CR = []
        self.memory_size = 10

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

                # Adaptive F using Cauchy distribution
                F_current = self.cauchy_scale * np.random.standard_cauchy()
                F_current = np.clip(F_current, 0.1, 1.0)

                v = x_r1 + F_current * (x_r2 - x_r3)
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
                     self.CR = self.CR - 0.05 # Decrease CR
                     self.CR = np.clip(self.CR, 0.1, 0.9)

                # Update best solution
                if f_u < self.f_opt:
                    self.f_opt = f_u
                    self.x_opt = u.copy()

        return self.f_opt, self.x_opt