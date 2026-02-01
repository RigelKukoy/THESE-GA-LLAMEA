import numpy as np

class SelfAdaptiveDEMutation:
    def __init__(self, budget=10000, dim=10, popsize=None, CR=0.7, F_init=0.5, F_min=0.1, F_max=1.0, cauchy_scale=0.01):
        self.budget = budget
        self.dim = dim
        self.popsize = popsize if popsize is not None else 10 * self.dim
        self.CR = CR
        self.F_init = F_init
        self.F_min = F_min
        self.F_max = F_max
        self.cauchy_scale = cauchy_scale
        self.F_history = []

    def __call__(self, func):
        lb = func.bounds.lb
        ub = func.bounds.ub
        self.population = np.random.uniform(lb, ub, size=(self.popsize, self.dim))
        self.fitness = np.array([func(x) for x in self.population])
        self.eval_count = self.popsize
        self.f_opt = np.min(self.fitness)
        self.x_opt = self.population[np.argmin(self.fitness)]
        self.F = np.full(self.popsize, self.F_init)

        while self.eval_count < self.budget:
            for i in range(self.popsize):
                # Mutation
                idxs = np.random.choice(self.popsize, 3, replace=False)
                x1, x2, x3 = self.population[idxs]

                # Self-adaptive F
                if len(self.F_history) > 0:
                    successful_F = [f for f, success in self.F_history if success]
                    if successful_F:
                        self.F[i] = np.mean(successful_F)
                    else:
                        self.F[i] = self.F_init  # Revert to initial value if no success
                
                self.F[i] = np.clip(self.F[i] + np.random.normal(0, self.cauchy_scale), self.F_min, self.F_max)
                
                mutant = x1 + self.F[i] * (x2 - x3)
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
                    self.F_history.append((self.F[i], True))  # Mark F as successful
                else:
                    self.F_history.append((self.F[i], False))  # Mark F as unsuccessful


        return self.f_opt, self.x_opt