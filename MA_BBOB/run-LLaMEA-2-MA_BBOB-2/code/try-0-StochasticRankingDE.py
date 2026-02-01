import numpy as np

class StochasticRankingDE:
    def __init__(self, budget=10000, dim=10, popsize=None, F=0.5, CR=0.7, p_rank=0.45):
        self.budget = budget
        self.dim = dim
        self.popsize = popsize if popsize is not None else 10 * self.dim
        self.F = F
        self.CR = CR
        self.p_rank = p_rank  # Probability of ranking based on objective function
        self.population = None
        self.fitness = None
        self.eval_count = 0
        self.f_opt = np.Inf
        self.x_opt = None

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
                idxs = np.random.choice(self.popsize, 3, replace=False)
                x1, x2, x3 = self.population[idxs]

                # Cauchy mutation
                mutant = self.population[i] + self.F * (x1 - x2) + np.random.standard_cauchy(size=self.dim) * 0.01
                mutant = np.clip(mutant, lb, ub)

                crossover_mask = np.random.rand(self.dim) < self.CR
                trial = np.where(crossover_mask, mutant, self.population[i])

                f_trial = func(trial)
                self.eval_count += 1

                # Stochastic ranking
                if (self.fitness[i] < 0 and f_trial < 0) or (np.random.rand() < self.p_rank):
                    if f_trial < self.fitness[i]:
                        self.population[i] = trial
                        self.fitness[i] = f_trial
                        if self.fitness[i] < self.f_opt:
                            self.f_opt = self.fitness[i]
                            self.x_opt = self.population[i]
                else:
                    # If both fitness values are positive and the random number is greater than p_rank
                    # Rank based on constraint violation (in this case, the objective value itself)
                    if f_trial < self.fitness[i]:
                        self.population[i] = trial
                        self.fitness[i] = f_trial
                        if self.fitness[i] < self.f_opt:
                            self.f_opt = self.fitness[i]
                            self.x_opt = self.population[i]
        return self.f_opt, self.x_opt