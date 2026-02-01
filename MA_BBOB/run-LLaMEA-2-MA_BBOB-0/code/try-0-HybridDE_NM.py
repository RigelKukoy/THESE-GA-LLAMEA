import numpy as np
from scipy.optimize import minimize

class HybridDE_NM:
    def __init__(self, budget=10000, dim=10, pop_multiplier=5, initial_F=0.5, initial_CR=0.9, nm_restarts=2):
        self.budget = budget
        self.dim = dim
        self.pop_size = dim * pop_multiplier
        self.F = initial_F
        self.CR = initial_CR
        self.lb = -5.0
        self.ub = 5.0
        self.x_opt = None
        self.f_opt = np.inf
        self.stagnation_limit = 50
        self.stagnation_counter = 0
        self.exploration_phase = True
        self.nm_restarts = nm_restarts

    def differential_evolution(self, func):
        population = np.random.uniform(self.lb, self.ub, size=(self.pop_size, self.dim))
        fitness = np.array([func(x) for x in population])
        self.budget -= self.pop_size

        while self.budget > 0:
            for i in range(self.pop_size):
                if self.budget <= 0:
                    break

                # Mutation
                indices = np.random.choice(self.pop_size, 3, replace=False)
                x1, x2, x3 = population[indices]
                mutant = population[i] + self.F * (x2 - x3)
                mutant = np.clip(mutant, self.lb, self.ub)

                # Crossover
                crossover = np.random.rand(self.dim) < self.CR
                trial = np.where(crossover, mutant, population[i])

                # Selection
                f_trial = func(trial)
                self.budget -= 1
                if f_trial < fitness[i]:
                    fitness[i] = f_trial
                    population[i] = trial

            # Update best solution
            best_index = np.argmin(fitness)
            if fitness[best_index] < self.f_opt:
                self.f_opt = fitness[best_index]
                self.x_opt = population[best_index]
                self.stagnation_counter = 0
            else:
                self.stagnation_counter += 1

            if self.stagnation_counter > self.stagnation_limit:
                self.exploration_phase = False
                break  # Switch to Nelder-Mead

        return population, fitness


    def nelder_mead(self, func, x0):
        bounds = ((self.lb, self.ub),) * self.dim
        result = minimize(func, x0, method='Nelder-Mead', bounds=bounds, options={'maxfev': self.budget})
        self.budget -= result.nfev
        if result.fun < self.f_opt:
             self.f_opt = result.fun
             self.x_opt = result.x
        return result.x, result.fun

    def __call__(self, func):
        # Initial Differential Evolution phase
        population, fitness = self.differential_evolution(func)

        # Nelder-Mead phase if DE stagnates
        if not self.exploration_phase:
            for _ in range(self.nm_restarts):
              if self.budget <= 0:
                break
              best_index = np.argmin(fitness)
              x0 = population[best_index].copy() # start NM from DE's best
              _, _ = self.nelder_mead(func, x0)

        # Nelder-Mead refinement from the best point found in DE if there is budget left and DE didn't stagnate
        if self.exploration_phase and self.budget > 0:
            best_index = np.argmin(fitness)
            x0 = population[best_index].copy() # start NM from DE's best
            _, _ = self.nelder_mead(func, x0)
        return self.f_opt, self.x_opt