import numpy as np

class AdaptiveDECrossoverRepair:
    def __init__(self, budget=10000, dim=10, popsize=None, F=0.5, CR=0.7):
        self.budget = budget
        self.dim = dim
        self.popsize = popsize if popsize is not None else 10 * self.dim
        self.F = F
        self.CR = CR

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
                # Mutation using a combination of Gaussian and Cauchy
                idxs = np.random.choice(self.popsize, 3, replace=False)
                x1, x2, x3 = self.population[idxs]

                # Adaptive F
                F = memory_F[i]
                mutant = x1 + F * (np.random.normal(size=self.dim) * (x2 - x3) + 0.1 * np.random.standard_cauchy(size=self.dim) * (x1 - x3))
                

                # Crossover
                CR = memory_CR[i]
                crossover_mask = np.random.rand(self.dim) < CR
                trial = np.where(crossover_mask, mutant, self.population[i])

                # Repair mechanism: Reflect back if out of bounds
                for j in range(self.dim):
                    if trial[j] < lb:
                        trial[j] = lb + (lb - trial[j])  # Reflect
                        if trial[j] > ub: #If after reflection is still out of bounds, clip it.
                           trial[j] = lb #Clip to the boundary if reflection is not sufficient
                    elif trial[j] > ub:
                        trial[j] = ub - (trial[j] - ub)  # Reflect
                        if trial[j] < lb: #If after reflection is still out of bounds, clip it.
                           trial[j] = ub #Clip to the boundary if reflection is not sufficient
                
                trial = np.clip(trial, lb, ub) # Another clipping for safety
                # Selection
                f_trial = func(trial)
                self.eval_count += 1

                if f_trial < self.fitness[i]:
                    success_F.append(F)
                    success_CR.append(CR)
                    self.population[i] = trial
                    self.fitness[i] = f_trial
                    if self.fitness[i] < self.f_opt:
                        self.f_opt = self.fitness[i]
                        self.x_opt = self.population[i]

                    # Update memory for F and CR
                    memory_F[i] = np.random.normal(self.F, 0.1)
                    memory_F[i] = np.clip(memory_F[i], 0.1, 1.0)
                    memory_CR[i] = np.random.normal(self.CR, 0.1)
                    memory_CR[i] = np.clip(memory_CR[i], 0.1, 1.0)
            
            # Adaptive F and CR based on success history
            if len(success_F) > 0:
                self.F = np.mean(success_F)
                self.F = np.clip(self.F, 0.1, 1.0)
                success_F = []
            if len(success_CR) > 0:
                self.CR = np.mean(success_CR)
                self.CR = np.clip(self.CR, 0.1, 1.0)
                success_CR = []
            
            # Population diversity maintenance (optional)
            if np.std(self.fitness) < 1e-8:  # Stagnation
                self.population = np.random.uniform(lb, ub, size=(self.popsize, self.dim))
                self.fitness = np.array([func(x) for x in self.population])
                self.eval_count += self.popsize
                best_idx = np.argmin(self.fitness)
                self.f_opt = self.fitness[best_idx]
                self.x_opt = self.population[best_idx]
                memory_F = np.ones(self.popsize) * self.F
                memory_CR = np.ones(self.popsize) * self.CR

        return self.f_opt, self.x_opt