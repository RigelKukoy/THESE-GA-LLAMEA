import numpy as np

class AgingRestartDE:
    def __init__(self, budget=10000, dim=10, popsize=None, F=0.5, CR=0.7, age_limit=50):
        self.budget = budget
        self.dim = dim
        self.popsize = popsize if popsize is not None else 10 * self.dim
        self.F = F
        self.CR = CR
        self.age_limit = age_limit

    def __call__(self, func):
        lb = func.bounds.lb
        ub = func.bounds.ub
        self.population = np.random.uniform(lb, ub, size=(self.popsize, self.dim))
        self.fitness = np.array([func(x) for x in self.population])
        self.eval_count = self.popsize
        self.f_opt = np.min(self.fitness)
        self.x_opt = self.population[np.argmin(self.fitness)]
        self.ages = np.zeros(self.popsize, dtype=int)


        while self.eval_count < self.budget:
            for i in range(self.popsize):
                # Mutation
                idxs = np.random.choice(self.popsize, 3, replace=False)
                x1, x2, x3 = self.population[idxs]
                mutant = x1 + self.F * (x2 - x3)
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
                    self.ages[i] = 0 # Reset age
                    if self.fitness[i] < self.f_opt:
                        self.f_opt = self.fitness[i]
                        self.x_opt = self.population[i]
                else:
                    self.ages[i] += 1

            # Aging mechanism: replace old solutions
            for i in range(self.popsize):
                if self.ages[i] > self.age_limit:
                    self.population[i] = np.random.uniform(lb, ub, size=self.dim)
                    self.fitness[i] = func(self.population[i])
                    self.eval_count += 1
                    self.ages[i] = 0
                    if self.fitness[i] < self.f_opt:
                        self.f_opt = self.fitness[i]
                        self.x_opt = self.population[i]


            # Population diversity maintenance (restart)
            if np.std(self.fitness) < 1e-8:  # Stagnation
                self.population = np.random.uniform(lb, ub, size=(self.popsize, self.dim))
                self.fitness = np.array([func(x) for x in self.population])
                self.eval_count += self.popsize
                best_idx = np.argmin(self.fitness)
                self.f_opt = self.fitness[best_idx]
                self.x_opt = self.population[best_idx]
                self.ages = np.zeros(self.popsize, dtype=int)


        return self.f_opt, self.x_opt