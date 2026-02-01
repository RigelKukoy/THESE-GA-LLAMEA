import numpy as np

class NeighborhoodAdaptiveDE:
    def __init__(self, budget=10000, dim=10, popsize=None, F=0.5, CR=0.7,
                 F_adaptive=True, CR_adaptive=True, neighborhood_size=5, age_limit=500):
        self.budget = budget
        self.dim = dim
        self.popsize = popsize if popsize is not None else 10 * self.dim
        self.F = F
        self.CR = CR
        self.F_adaptive = F_adaptive
        self.CR_adaptive = CR_adaptive
        self.neighborhood_size = neighborhood_size
        self.age_limit = age_limit
        self.ages = np.zeros(self.popsize)  # Initialize ages for each individual

    def __call__(self, func):
        lb = func.bounds.lb
        ub = func.bounds.ub
        self.population = np.random.uniform(lb, ub, size=(self.popsize, self.dim))
        self.fitness = np.array([func(x) for x in self.population])
        self.eval_count = self.popsize
        self.f_opt = np.min(self.fitness)
        self.x_opt = self.population[np.argmin(self.fitness)]

        memory_F = np.ones(self.popsize) * self.F
        memory_CR = np.ones(self.popsize) * self.CR
        success_F = []
        success_CR = []

        while self.eval_count < self.budget:
            for i in range(self.popsize):
                # Neighborhood-based mutation
                neighbors = np.random.choice(self.popsize, self.neighborhood_size, replace=False)
                best_neighbor_idx = neighbors[np.argmin(self.fitness[neighbors])]
                idxs = np.random.choice(self.popsize, 2, replace=False)
                x1, x2 = self.population[idxs]

                if self.F_adaptive:
                    F = memory_F[i]
                else:
                    F = self.F

                mutant = self.population[best_neighbor_idx] + F * (x1 - x2)
                mutant = np.clip(mutant, lb, ub)

                if self.CR_adaptive:
                    CR = memory_CR[i]
                else:
                    CR = self.CR

                crossover_mask = np.random.rand(self.dim) < CR
                trial = np.where(crossover_mask, mutant, self.population[i])

                f_trial = func(trial)
                self.eval_count += 1

                if f_trial < self.fitness[i]:
                    success_F.append(F)
                    success_CR.append(CR)
                    self.population[i] = trial
                    self.fitness[i] = f_trial
                    self.ages[i] = 0  # Reset age if improved

                    if self.fitness[i] < self.f_opt:
                        self.f_opt = self.fitness[i]
                        self.x_opt = self.population[i]

                    if self.F_adaptive:
                        memory_F[i] = np.random.normal(self.F, 0.1)
                        memory_F[i] = np.clip(memory_F[i], 0.1, 1.0)
                    if self.CR_adaptive:
                        memory_CR[i] = np.random.normal(self.CR, 0.1)
                        memory_CR[i] = np.clip(memory_CR[i], 0.1, 1.0)
                else:
                    self.ages[i] += 1  # Increment age if not improved
            #Adaptive F and CR using Lehmer mean
            if self.F_adaptive and len(success_F) > 0:
                self.F = np.sum(np.array(success_F)**2) / np.sum(success_F)
                self.F = np.clip(self.F, 0.1, 1.0)
                success_F = []
            if self.CR_adaptive and len(success_CR) > 0:
                self.CR = np.sum(np.array(success_CR)**2) / np.sum(success_CR)
                self.CR = np.clip(self.CR, 0.1, 1.0)
                success_CR = []


            # Aging mechanism: replace old individuals
            for i in range(self.popsize):
                if self.ages[i] > self.age_limit:
                    self.population[i] = np.random.uniform(lb, ub, size=self.dim)
                    self.fitness[i] = func(self.population[i])
                    self.eval_count += 1
                    self.ages[i] = 0
                    if self.fitness[i] < self.f_opt:
                        self.f_opt = self.fitness[i]
                        self.x_opt = self.population[i]

        return self.f_opt, self.x_opt