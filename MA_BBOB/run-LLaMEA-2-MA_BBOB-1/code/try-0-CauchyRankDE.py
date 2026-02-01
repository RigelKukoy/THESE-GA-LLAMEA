import numpy as np

class CauchyRankDE:
    def __init__(self, budget=10000, dim=10, pop_size=50, F=0.5, CR=0.7, restart_prob=0.05):
        self.budget = budget
        self.dim = dim
        self.pop_size = pop_size
        self.F = F
        self.CR = CR
        self.restart_prob = restart_prob
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

    def cauchy_mutation(self, x_r1, x_r2, F):
        return x_r1 + F * np.random.standard_cauchy(size=x_r1.shape) * (x_r1 - x_r2)


    def __call__(self, func):
        self.initialize_population(func)

        while self.budget > 0:
            for i in range(self.pop_size):
                # Mutation
                indices = np.random.choice(self.pop_size, 3, replace=False)
                x_r1, x_r2, x_r3 = self.population[indices]
                
                # Cauchy mutation
                v = self.cauchy_mutation(x_r1, x_r2, self.F)

                v = np.clip(v, func.bounds.lb, func.bounds.ub)

                # Crossover
                j_rand = np.random.randint(self.dim)
                u = self.population[i].copy()
                for j in range(self.dim):
                    if np.random.rand() < self.CR or j == j_rand:
                        u[j] = v[j]

                # Evaluation
                f_u = func(u)
                self.budget -= 1

                # Rank-based selection
                if f_u < self.fitness[i]:
                    self.fitness[i] = f_u
                    self.population[i] = u
                    if f_u < self.f_opt:
                        self.f_opt = f_u
                        self.x_opt = u.copy()

            # Restart mechanism
            if np.random.rand() < self.restart_prob:
                self.population = np.random.uniform(func.bounds.lb, func.bounds.ub, size=(self.pop_size, self.dim))
                self.fitness = np.array([func(x) for x in self.population])
                self.budget -= self.pop_size
                self.best_index = np.argmin(self.fitness)
                self.f_opt = self.fitness[self.best_index]
                self.x_opt = self.population[self.best_index].copy()

        return self.f_opt, self.x_opt