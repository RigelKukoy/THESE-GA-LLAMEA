import numpy as np
from scipy.optimize import minimize

class SelfAdaptiveDE_NM:
    def __init__(self, budget=10000, dim=10, popsize_init=None, F=0.5, CR=0.7, local_search_freq=10):
        self.budget = budget
        self.dim = dim
        self.popsize_init = popsize_init if popsize_init is not None else 10 * self.dim
        self.popsize = self.popsize_init
        self.F = F
        self.CR = CR
        self.local_search_freq = local_search_freq
        self.eval_count = 0

    def __call__(self, func):
        lb = func.bounds.lb
        ub = func.bounds.ub
        self.population = np.random.uniform(lb, ub, size=(self.popsize, self.dim))
        self.fitness = np.array([func(x) for x in self.population])
        self.eval_count += self.popsize
        self.f_opt = np.min(self.fitness)
        self.x_opt = self.population[np.argmin(self.fitness)]
        generation = 0

        while self.eval_count < self.budget:
            generation += 1
            for i in range(self.popsize):
                # Weighted Difference Mutation
                idxs = np.random.choice(self.popsize, 4, replace=False)
                x1, x2, x3, x4 = self.population[idxs]
                weights = np.random.rand(2)
                weights /= np.sum(weights)  # Normalize weights
                mutant = x1 + self.F * (weights[0] * (x2 - x3) + weights[1] * (x4 - x1))  # Weighted difference
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
            
            # Local Search (Nelder-Mead)
            if generation % self.local_search_freq == 0:
                idx = np.argmin(self.fitness)
                x_local = self.population[idx].copy()
                
                def obj_for_nm(x):
                    return func(x)

                res = minimize(obj_for_nm, x_local, method='Nelder-Mead', bounds=func.bounds)
                nfev = res.nfev
                if self.eval_count + nfev <= self.budget:
                    self.eval_count += nfev
                    f_local = res.fun
                    x_local = res.x
                    if f_local < self.f_opt:
                        self.f_opt = f_local
                        self.x_opt = x_local
                else:
                   break  # Stop if local search exceeds budget
            
            # Self-Adaptive Population Size (simplified - can be made more sophisticated)
            if generation % 20 == 0:  # Adjust every 20 generations
                if np.std(self.fitness) < 1e-6:  # Stagnation
                    self.popsize = int(self.popsize * 0.8)
                    if self.popsize < 4 * self.dim:
                        self.popsize = self.popsize_init  # Reset
                    
                    # Resample population (only if budget allows)
                    if self.eval_count + self.popsize * self.dim <= self.budget:
                        new_population = np.random.uniform(lb, ub, size=(self.popsize, self.dim))
                        new_fitness = np.array([func(x) for x in new_population])
                        self.eval_count += self.popsize
                        
                        # Combine old and new populations and select best
                        combined_population = np.concatenate((self.population, new_population))
                        combined_fitness = np.concatenate((self.fitness, new_fitness))
                        
                        indices = np.argsort(combined_fitness)[:self.popsize]
                        self.population = combined_population[indices]
                        self.fitness = combined_fitness[indices]
                        
                        best_idx = np.argmin(self.fitness)
                        self.f_opt = self.fitness[best_idx]
                        self.x_opt = self.population[best_idx]
                    else:
                        break


        return self.f_opt, self.x_opt