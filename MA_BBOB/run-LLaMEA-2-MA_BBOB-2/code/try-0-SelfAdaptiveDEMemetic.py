import numpy as np

class SelfAdaptiveDEMemetic:
    def __init__(self, budget=10000, dim=10, popsize=None, F_range=(0.1, 0.9), CR_range=(0.1, 0.9), local_search_freq=10, local_search_radius=0.1):
        self.budget = budget
        self.dim = dim
        self.popsize = popsize if popsize is not None else 10 * self.dim
        self.F_range = F_range
        self.CR_range = CR_range
        self.local_search_freq = local_search_freq
        self.local_search_radius = local_search_radius
        self.F = np.random.uniform(F_range[0], F_range[1], size=self.popsize)
        self.CR = np.random.uniform(CR_range[0], CR_range[1], size=self.popsize)
        self.success_F = []
        self.success_CR = []
        self.archive = []
        self.archive_size = int(self.popsize/2)

    def local_search(self, func, x, radius):
        # Simple random local search around x
        best_x = x
        best_f = func(x)
        for _ in range(5):
            new_x = x + np.random.uniform(-radius, radius, size=self.dim)
            new_x = np.clip(new_x, func.bounds.lb, func.bounds.ub)
            new_f = func(new_x)
            if new_f < best_f:
                best_f = new_f
                best_x = new_x
        return best_f, best_x

    def __call__(self, func):
        lb = func.bounds.lb
        ub = func.bounds.ub
        self.population = np.random.uniform(lb, ub, size=(self.popsize, self.dim))
        self.fitness = np.array([func(x) for x in self.population])
        self.eval_count = self.popsize
        self.f_opt = np.min(self.fitness)
        self.x_opt = self.population[np.argmin(self.fitness)]

        generation = 0
        while self.eval_count < self.budget:
            for i in range(self.popsize):
                # Mutation
                idxs = np.random.choice(self.popsize, 3, replace=False)
                x1, x2, x3 = self.population[idxs]
                mutant = x1 + self.F[i] * (x2 - x3)
                mutant = np.clip(mutant, lb, ub)

                # Crossover
                crossover_mask = np.random.rand(self.dim) < self.CR[i]
                trial = np.where(crossover_mask, mutant, self.population[i])

                # Selection
                f_trial = func(trial)
                self.eval_count += 1

                if f_trial < self.fitness[i]:
                    self.success_F.append(self.F[i])
                    self.success_CR.append(self.CR[i])
                    self.archive.append(self.population[i].copy())
                    if len(self.archive) > self.archive_size:
                        self.archive.pop(0)

                    self.population[i] = trial
                    self.fitness[i] = f_trial
                    if self.fitness[i] < self.f_opt:
                        self.f_opt = self.fitness[i]
                        self.x_opt = self.population[i]

            # Update F and CR based on success history
            if self.success_F:
                self.F = np.clip(np.random.normal(np.mean(self.success_F), np.std(self.success_F), size=self.popsize), self.F_range[0], self.F_range[1])
                self.CR = np.clip(np.random.normal(np.mean(self.success_CR), np.std(self.success_CR), size=self.popsize), self.CR_range[0], self.CR_range[1])

            self.success_F = []
            self.success_CR = []

            # Memetic Local Search
            if generation % self.local_search_freq == 0:
                for i in range(self.popsize):
                    f_local, x_local = self.local_search(func, self.population[i], self.local_search_radius)
                    self.eval_count += 5  #Local search makes 5 evaluations
                    if f_local < self.fitness[i]:
                        self.population[i] = x_local
                        self.fitness[i] = f_local
                        if self.fitness[i] < self.f_opt:
                            self.f_opt = self.fitness[i]
                            self.x_opt = self.population[i]

            generation += 1

        return self.f_opt, self.x_opt