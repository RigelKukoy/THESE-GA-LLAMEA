import numpy as np

class RingTopologyRestartDE:
    def __init__(self, budget=10000, dim=10, popsize=None, F_base=0.5, CR_base=0.7, restart_probability=0.05):
        self.budget = budget
        self.dim = dim
        self.popsize = popsize if popsize is not None else 10 * self.dim
        self.F_base = F_base
        self.CR_base = CR_base
        self.restart_probability = restart_probability

    def __call__(self, func):
        lb = func.bounds.lb
        ub = func.bounds.ub
        self.population = np.random.uniform(lb, ub, size=(self.popsize, self.dim))
        self.fitness = np.array([func(x) for x in self.population])
        self.eval_count = self.popsize
        self.f_opt = np.min(self.fitness)
        self.x_opt = self.population[np.argmin(self.fitness)]

        while self.eval_count < self.budget:
            for i in range(self.popsize):
                # Self-adaptive parameters
                F = np.random.normal(self.F_base, 0.1)
                F = np.clip(F, 0.1, 1.0)
                CR = np.random.normal(self.CR_base, 0.1)
                CR = np.clip(CR, 0.1, 1.0)

                # Ring Topology Selection: Select neighbors
                left = (i - 1) % self.popsize
                right = (i + 1) % self.popsize

                # Mutation
                idxs = np.random.choice(self.popsize, 3, replace=False)
                x1, x2, x3 = self.population[idxs]
                mutant = x1 + F * (x2 - x3)
                mutant = np.clip(mutant, lb, ub)

                # Crossover
                crossover_mask = np.random.rand(self.dim) < CR
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

                # Restart mechanism
                if np.random.rand() < self.restart_probability:
                    self.population[i] = np.random.uniform(lb, ub, size=self.dim)
                    self.fitness[i] = func(self.population[i])
                    self.eval_count += 1
                    if self.fitness[i] < self.f_opt:
                        self.f_opt = self.fitness[i]
                        self.x_opt = self.population[i]

                if self.eval_count >= self.budget:
                    break

        return self.f_opt, self.x_opt