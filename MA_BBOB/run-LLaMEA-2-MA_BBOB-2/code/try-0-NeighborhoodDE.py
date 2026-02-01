import numpy as np
from scipy.optimize import minimize

class NeighborhoodDE:
    def __init__(self, budget=10000, dim=10, popsize=None, F=0.5, CR=0.7, neighborhood_size=5,
                 F_adaptive=True, CR_adaptive=True, local_search_freq=500):
        self.budget = budget
        self.dim = dim
        self.popsize = popsize if popsize is not None else 10 * self.dim
        self.F = F
        self.CR = CR
        self.neighborhood_size = neighborhood_size
        self.F_adaptive = F_adaptive
        self.CR_adaptive = CR_adaptive
        self.local_search_freq = local_search_freq
        self.eval_count = 0
        self.f_opt = np.Inf
        self.x_opt = None
        self.population = None
        self.fitness = None

    def initialize_population(self, func):
        lb = func.bounds.lb
        ub = func.bounds.ub
        self.population = np.random.uniform(lb, ub, size=(self.popsize, self.dim))
        self.fitness = np.array([func(x) for x in self.population])
        self.eval_count += self.popsize
        self.f_opt = np.min(self.fitness)
        self.x_opt = self.population[np.argmin(self.fitness)]

    def mutate(self, i):
        # Select neighborhood
        idxs = np.random.choice(self.popsize, self.neighborhood_size, replace=False)
        
        # Ensure the current individual is not in the neighborhood
        while i in idxs:
            idxs = np.random.choice(self.popsize, self.neighborhood_size, replace=False)

        # Choose three distinct individuals from neighborhood
        a, b, c = idxs[:3]
        
        mutant = self.population[a] + self.F * (self.population[b] - self.population[c])
        return mutant

    def crossover(self, mutant, i):
        crossover_mask = np.random.rand(self.dim) < self.CR
        trial = np.where(crossover_mask, mutant, self.population[i])
        return trial

    def selection(self, trial, i, func):
        f_trial = func(trial)
        self.eval_count += 1

        if f_trial < self.fitness[i]:
            self.population[i] = trial
            self.fitness[i] = f_trial
            if self.fitness[i] < self.f_opt:
                self.f_opt = self.fitness[i]
                self.x_opt = self.population[i]
                return True
        return False

    def adapt_parameters(self):
        # Rank-based adaptation: Adjust F and CR based on fitness rank
        ranked_indices = np.argsort(self.fitness)
        best_indices = ranked_indices[:self.popsize // 4]  # Top 25%

        if self.F_adaptive:
            self.F = np.clip(np.random.normal(0.5, 0.1), 0.1, 1.0)  # Example: Adjust F randomly
        if self.CR_adaptive:
            self.CR = np.clip(np.random.normal(0.7, 0.1), 0.1, 1.0) # Example: Adjust CR randomly

    def local_search(self, func):
        # Apply local search (Nelder-Mead) to the best individual
        res = minimize(func, self.x_opt, method='Nelder-Mead',
                       bounds=func.bounds, options={'maxfev': self.local_search_freq // 2})
        if res.fun < self.f_opt:
            self.f_opt = res.fun
            self.x_opt = res.x
        self.eval_count += res.nfev

    def __call__(self, func):
        self.initialize_population(func)

        while self.eval_count < self.budget:
            for i in range(self.popsize):
                if self.eval_count >= self.budget:
                    break

                # Mutation
                mutant = self.mutate(i)
                lb = func.bounds.lb
                ub = func.bounds.ub
                mutant = np.clip(mutant, lb, ub)
                # Crossover
                trial = self.crossover(mutant, i)
                trial = np.clip(trial, lb, ub)

                # Selection
                self.selection(trial, i, func)

                if self.eval_count >= self.budget:
                    break

            # Adapt parameters periodically
            self.adapt_parameters()

            # Local search
            if self.eval_count % self.local_search_freq == 0 and self.eval_count < self.budget:
                self.local_search(func)
        return self.f_opt, self.x_opt