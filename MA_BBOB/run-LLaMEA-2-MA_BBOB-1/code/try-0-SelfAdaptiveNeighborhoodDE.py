import numpy as np

class SelfAdaptiveNeighborhoodDE:
    def __init__(self, budget=10000, dim=10, pop_size=50, F_init=0.5, CR_init=0.7, neighborhood_size=5):
        self.budget = budget
        self.dim = dim
        self.pop_size = pop_size
        self.F_init = F_init
        self.CR_init = CR_init
        self.neighborhood_size = neighborhood_size
        self.population = None
        self.fitness = None
        self.F = None
        self.CR = None
        self.best_index = None
        self.f_opt = np.inf
        self.x_opt = None

    def initialize_population(self, func):
        self.population = np.random.uniform(func.bounds.lb, func.bounds.ub, size=(self.pop_size, self.dim))
        self.fitness = np.array([func(x) for x in self.population])
        self.F = np.full(self.pop_size, self.F_init)
        self.CR = np.full(self.pop_size, self.CR_init)
        self.budget -= self.pop_size
        self.best_index = np.argmin(self.fitness)
        self.f_opt = self.fitness[self.best_index]
        self.x_opt = self.population[self.best_index].copy()

    def __call__(self, func):
        self.initialize_population(func)

        while self.budget > 0:
            for i in range(self.pop_size):
                # Self-adaptive parameters
                F = self.F[i]
                CR = self.CR[i]

                # Neighborhood-based mutation
                neighbors = np.random.choice(self.pop_size, self.neighborhood_size, replace=False)
                x_r1, x_r2 = self.population[np.random.choice(neighbors, 2, replace=False)]

                # Mutation
                v = self.population[i] + F * (x_r1 - x_r2)
                v = np.clip(v, func.bounds.lb, func.bounds.ub)

                # Crossover
                j_rand = np.random.randint(self.dim)
                u = self.population[i].copy()
                for j in range(self.dim):
                    if np.random.rand() < CR or j == j_rand:
                        u[j] = v[j]

                # Evaluation
                f_u = func(u)
                self.budget -= 1

                # Stochastic Ranking Selection
                p_f = 0.45  # Probability of comparing based on fitness
                if np.random.rand() < p_f or self.fitness[i] < self.f_opt or f_u < self.f_opt:
                    if f_u < self.fitness[i]:
                        # Update successful
                        self.fitness[i] = f_u
                        self.population[i] = u
                        if f_u < self.f_opt:
                            self.f_opt = f_u
                            self.x_opt = u.copy()

                        # Adapt parameters
                        self.F[i] = np.clip(np.random.normal(F, 0.1), 0.1, 1.0)
                        self.CR[i] = np.clip(np.random.normal(CR, 0.1), 0.1, 1.0)
                    else:
                        # Adapt parameters (failure)
                        self.F[i] = np.clip(np.random.normal(F, 0.1), 0.1, 1.0)
                        self.CR[i] = np.clip(np.random.normal(CR, 0.1), 0.1, 1.0)
                else:
                    # Compare based on constraint violation (in this case, assume all constraints are satisfied, so compare randomly)
                    if np.random.rand() < 0.5:  # 50% chance of replacing
                        self.fitness[i] = f_u
                        self.population[i] = u
        return self.f_opt, self.x_opt