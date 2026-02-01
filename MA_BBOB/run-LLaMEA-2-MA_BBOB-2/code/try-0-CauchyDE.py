import numpy as np

class CauchyDE:
    def __init__(self, budget=10000, dim=10, popsize=None, F=0.5, CR=0.7, restart_interval=1000):
        self.budget = budget
        self.dim = dim
        self.popsize = popsize if popsize is not None else 10 * self.dim
        self.F = F
        self.CR = CR
        self.restart_interval = restart_interval
        self.eval_count = 0

    def __call__(self, func):
        lb = func.bounds.lb
        ub = func.bounds.ub
        self.population = np.random.uniform(lb, ub, size=(self.popsize, self.dim))
        self.fitness = np.array([func(x) for x in self.population])
        self.eval_count += self.popsize
        self.f_opt = np.min(self.fitness)
        self.x_opt = self.population[np.argmin(self.fitness)]
        self.generation = 0

        while self.eval_count < self.budget:
            for i in range(self.popsize):
                # Mutation (Cauchy)
                idxs = np.random.choice(self.popsize, 3, replace=False)
                x1, x2, x3 = self.population[idxs]
                
                mutant = x1 + self.F * np.random.standard_cauchy(size=self.dim) * (x2 - x3)
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
            
            self.generation += 1
            if self.generation * self.popsize % self.restart_interval == 0:
                 # Restart the population
                self.population = np.random.uniform(lb, ub, size=(self.popsize, self.dim))
                self.fitness = np.array([func(x) for x in self.population])
                self.eval_count += self.popsize
                best_idx = np.argmin(self.fitness)
                self.f_opt = self.fitness[best_idx]
                self.x_opt = self.population[best_idx]

        return self.f_opt, self.x_opt