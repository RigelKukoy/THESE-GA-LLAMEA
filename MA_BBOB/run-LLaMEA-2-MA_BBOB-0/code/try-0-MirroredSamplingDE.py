import numpy as np

class MirroredSamplingDE:
    def __init__(self, budget=10000, dim=10, pop_size=50, mirror_rate=0.2, F=0.5, CR=0.9):
        self.budget = budget
        self.dim = dim
        self.pop_size = pop_size
        self.mirror_rate = mirror_rate
        self.F = F
        self.CR = CR

    def __call__(self, func):
        self.f_opt = np.Inf
        self.x_opt = None
        self.population = np.random.uniform(func.bounds.lb, func.bounds.ub, size=(self.pop_size, self.dim))
        fitness = np.array([func(x) for x in self.population])
        self.budget -= self.pop_size

        for i in range(self.pop_size):
            if fitness[i] < self.f_opt:
                self.f_opt = fitness[i]
                self.x_opt = self.population[i]

        while self.budget > 0:
            for i in range(self.pop_size):
                # Mutation
                idxs = np.random.choice(self.pop_size, 3, replace=False)
                x_r1, x_r2, x_r3 = self.population[idxs]
                mutant = x_r1 + self.F * (x_r2 - x_r3)
                mutant = np.clip(mutant, func.bounds.lb, func.bounds.ub)

                # Mirrored Sampling
                if np.random.rand() < self.mirror_rate:
                    mirror_point = self.x_opt + np.random.normal(0, 0.1, self.dim)  # Sample around best solution
                    mirror_point = np.clip(mirror_point, func.bounds.lb, func.bounds.ub)
                    mutant = 0.5 * (mutant + mirror_point) # Combine with the mutant

                # Crossover
                crossover = np.random.uniform(size=self.dim) < self.CR
                trial = np.where(crossover, mutant, self.population[i])
                trial = np.clip(trial, func.bounds.lb, func.bounds.ub)

                # Selection
                f_trial = func(trial)
                self.budget -= 1

                if f_trial < self.f_opt:
                    self.f_opt = f_trial
                    self.x_opt = trial

                if f_trial < fitness[i]:
                    fitness[i] = f_trial
                    self.population[i] = trial

        return self.f_opt, self.x_opt