import numpy as np

class CauchyMutationDE:
    def __init__(self, budget=10000, dim=10, popsize=None, CR=0.7, F=0.5, F_adapt=True, CR_adapt=True):
        self.budget = budget
        self.dim = dim
        self.popsize = popsize if popsize is not None else 10 * self.dim
        self.CR = CR
        self.F = F
        self.F_adapt = F_adapt
        self.CR_adapt = CR_adapt
        self.F_memory = []
        self.CR_memory = []

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
                # Parameter adaptation
                if self.F_adapt:
                    self.F = self._adapt_parameter(self.F, self.F_memory, 0.1)
                if self.CR_adapt:
                    self.CR = self._adapt_parameter(self.CR, self.CR_memory, 0.1)

                # Mutation using Cauchy distribution
                idxs = np.random.choice(self.popsize, 3, replace=False)
                x1, x2, x3 = self.population[idxs]
                mutant = self.population[i] + self.F * (x2 - x3) + self._cauchy_mutation(self.dim)

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
                        if self.F_adapt:
                            self.F_memory.append(self.F)
                        if self.CR_adapt:
                            self.CR_memory.append(self.CR)

        return self.f_opt, self.x_opt

    def _cauchy_mutation(self, dim, scale=0.1):
        return np.random.standard_cauchy(size=dim) * scale

    def _adapt_parameter(self, param, memory, adaptation_rate):
        if memory:
            change = adaptation_rate * (np.mean(memory) - param)
            param += change
            param = np.clip(param, 0.1, 0.9)  # Ensure F and CR stay within reasonable bounds
        return param