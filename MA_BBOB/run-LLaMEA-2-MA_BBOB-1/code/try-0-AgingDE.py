import numpy as np

class AgingDE:
    def __init__(self, budget=10000, dim=10, pop_size=50, F=0.5, CR=0.7, age_limit=50, stagnation_threshold=1e-6):
        self.budget = budget
        self.dim = dim
        self.pop_size = pop_size
        self.F = F
        self.CR = CR
        self.age_limit = age_limit
        self.stagnation_threshold = stagnation_threshold
        self.population = None
        self.fitness = None
        self.ages = None
        self.best_index = None
        self.f_opt = np.inf
        self.x_opt = None

    def initialize_population(self, func):
        self.population = np.random.uniform(func.bounds.lb, func.bounds.ub, size=(self.pop_size, self.dim))
        self.fitness = np.array([func(x) for x in self.population])
        self.ages = np.zeros(self.pop_size, dtype=int)
        self.budget -= self.pop_size
        self.best_index = np.argmin(self.fitness)
        self.f_opt = self.fitness[self.best_index]
        self.x_opt = self.population[self.best_index].copy()
        self.lb = func.bounds.lb
        self.ub = func.bounds.ub

    def __call__(self, func):
        self.initialize_population(func)

        generation = 0
        while self.budget > 0:
            generation += 1
            old_best_fitness = self.f_opt

            for i in range(self.pop_size):
                # Mutation
                indices = np.random.choice(self.pop_size, 3, replace=False)
                x_r1, x_r2, x_r3 = self.population[indices]

                # Stagnation check and F adaptation
                if np.abs(self.f_opt - old_best_fitness) < self.stagnation_threshold:
                    self.F = min(self.F + 0.1, 1.0)  # Increase mutation strength if stagnating
                else:
                    self.F = max(self.F - 0.05, 0.1)  # Decrease mutation strength if improving

                v = x_r1 + self.F * (x_r2 - x_r3)
                v = np.clip(v, self.lb, self.ub)

                # Crossover
                j_rand = np.random.randint(self.dim)
                u = self.population[i].copy()
                for j in range(self.dim):
                    if np.random.rand() < self.CR or j == j_rand:
                        u[j] = v[j]

                # Evaluation
                f_u = func(u)
                self.budget -= 1

                # Selection
                if f_u < self.fitness[i]:
                    self.fitness[i] = f_u
                    self.population[i] = u
                    self.ages[i] = 0  # Reset age
                    if f_u < self.f_opt:
                        self.f_opt = f_u
                        self.x_opt = u.copy()
                        self.best_index = i
                else:
                    self.ages[i] += 1  # Increment age

            # Aging: Replace oldest individuals
            for i in range(self.pop_size):
                if self.ages[i] > self.age_limit:
                    self.population[i] = np.random.uniform(self.lb, self.ub, size=self.dim)
                    self.fitness[i] = func(self.population[i])
                    self.budget -=1
                    self.ages[i] = 0
                    if self.fitness[i] < self.f_opt:
                        self.f_opt = self.fitness[i]
                        self.x_opt = self.population[i].copy()
                        self.best_index = i


        return self.f_opt, self.x_opt