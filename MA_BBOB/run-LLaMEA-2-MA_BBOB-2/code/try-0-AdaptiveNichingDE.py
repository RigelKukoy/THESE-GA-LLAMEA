import numpy as np

class AdaptiveNichingDE:
    def __init__(self, budget=10000, dim=10, popsize=None, F=0.5, CR=0.7, F_adaptive=True, CR_adaptive=True, cauchy_scale=0.1, gaussian_scale=0.1, niching_radius=0.1):
        self.budget = budget
        self.dim = dim
        self.popsize = popsize if popsize is not None else 10 * self.dim
        self.F = F
        self.CR = CR
        self.F_adaptive = F_adaptive
        self.CR_adaptive = CR_adaptive
        self.cauchy_scale = cauchy_scale
        self.gaussian_scale = gaussian_scale
        self.niching_radius = niching_radius

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
                # Mutation using a combination of Cauchy and Gaussian distributions
                idxs = np.random.choice(self.popsize, 3, replace=False)
                x1, x2, x3 = self.population[idxs]

                if self.F_adaptive:
                     F = memory_F[i]
                else:
                     F = self.F
                
                # Apply Cauchy mutation with a certain probability, otherwise Gaussian
                if np.random.rand() < 0.5:
                    mutant = x1 + F * np.random.standard_cauchy(size=self.dim) * self.cauchy_scale * (x2 - x3)
                else:
                    mutant = x1 + F * np.random.normal(size=self.dim) * self.gaussian_scale * (x2 - x3)
                
                mutant = np.clip(mutant, lb, ub)

                # Crossover
                if self.CR_adaptive:
                    CR = memory_CR[i]
                else:
                    CR = self.CR
                
                crossover_mask = np.random.rand(self.dim) < CR
                trial = np.where(crossover_mask, mutant, self.population[i])

                # Selection and Niching
                f_trial = func(trial)
                self.eval_count += 1

                # Niching: if trial is too close to another individual, penalize its fitness
                for j in range(self.popsize):
                    if i != j and np.linalg.norm(trial - self.population[j]) < self.niching_radius:
                        f_trial += 0.01 * np.abs(self.fitness[j] - f_trial)  # Small penalty

                if f_trial < self.fitness[i]:
                    success_F.append(F)
                    success_CR.append(CR)
                    self.population[i] = trial
                    self.fitness[i] = f_trial
                    if self.fitness[i] < self.f_opt:
                        self.f_opt = self.fitness[i]
                        self.x_opt = self.population[i]

                    if self.F_adaptive:
                        memory_F[i] = np.random.normal(self.F, 0.1)
                        memory_F[i] = np.clip(memory_F[i], 0.1, 1.0)
                    if self.CR_adaptive:
                        memory_CR[i] = np.random.normal(self.CR, 0.1)
                        memory_CR[i] = np.clip(memory_CR[i], 0.1, 1.0)
            
            # Adaptive F and CR based on success history
            if self.F_adaptive and len(success_F) > 0:
                self.F = np.mean(success_F)
                self.F = np.clip(self.F, 0.1, 1.0)
                success_F = []
            if self.CR_adaptive and len(success_CR) > 0:
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