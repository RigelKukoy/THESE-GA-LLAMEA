import numpy as np

class LocalityAwareSelfAdaptiveDE:
    def __init__(self, budget=10000, dim=10, popsize=None, CR=0.7, F=0.5, locality_ratio=0.1):
        self.budget = budget
        self.dim = dim
        self.popsize = popsize if popsize is not None else 10 * self.dim
        self.CR = CR
        self.F = F
        self.locality_ratio = locality_ratio

    def __call__(self, func):
        lb = func.bounds.lb
        ub = func.bounds.ub
        self.population = np.random.uniform(lb, ub, size=(self.popsize, self.dim))
        self.fitness = np.array([func(x) for x in self.population])
        self.eval_count = self.popsize
        self.f_opt = np.min(self.fitness)
        self.x_opt = self.population[np.argmin(self.fitness)]
        self.F_history = []

        while self.eval_count < self.budget:
            for i in range(self.popsize):
                # Self-adaptive F
                F = np.random.normal(self.F, 0.1)
                F = np.clip(F, 0.1, 1.0)

                # Mutation: Incorporate locality-based exploration
                if np.random.rand() < self.locality_ratio:
                    # Local search: Perturb the current individual
                    mutant = self.population[i] + np.random.normal(0, 0.1, size=self.dim) * (ub - lb)
                else:
                    # Standard DE mutation (rand1)
                    idxs = np.random.choice(self.popsize, 3, replace=False)
                    x1, x2, x3 = self.population[idxs]
                    mutant = self.population[i] + F * (x2 - x3)

                mutant = np.clip(mutant, lb, ub)

                # Crossover
                crossover_mask = np.random.rand(self.dim) < self.CR
                trial = np.where(crossover_mask, mutant, self.population[i])

                # Selection
                f_trial = func(trial)
                self.eval_count += 1

                if f_trial < self.fitness[i]:
                    self.population[i] = trial
                    self.fitness[i] = f_trial
                    if self.fitness[i] < self.f_opt:
                        self.f_opt = self.fitness[i]
                        self.x_opt = self.population[i]
                    self.F_history.append(F)
                
            # Adapt F based on recent success
            if self.F_history:
              self.F = 0.9 * self.F + 0.1 * np.mean(self.F_history)
              self.F_history = []


        return self.f_opt, self.x_opt