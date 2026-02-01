import numpy as np
from scipy.optimize import minimize

class AdaptiveDE_NM:
    def __init__(self, budget=10000, dim=10, pop_size=20, F=0.5, CR=0.7, local_search_prob=0.1, local_search_scale=0.1):
        self.budget = budget
        self.dim = dim
        self.pop_size = pop_size
        self.F = F
        self.CR = CR
        self.local_search_prob = local_search_prob
        self.local_search_scale = local_search_scale

    def __call__(self, func):
        # Initialize population
        pop = np.random.uniform(func.bounds.lb, func.bounds.ub, size=(self.pop_size, self.dim))
        fitness = np.array([func(x) for x in pop])
        self.budget -= self.pop_size

        # Find best initial solution
        best_index = np.argmin(fitness)
        best_fitness = fitness[best_index]
        best_solution = pop[best_index].copy()

        generation = 0
        while self.budget > 0:
            generation += 1
            for i in range(self.pop_size):
                # Differential Evolution mutation
                idxs = [idx for idx in range(self.pop_size) if idx != i]
                a, b, c = np.random.choice(idxs, 3, replace=False)
                mutant = pop[a] + self.F * (pop[b] - pop[c])
                mutant = np.clip(mutant, func.bounds.lb, func.bounds.ub)

                # Crossover
                trial = pop[i].copy()
                mask = np.random.rand(self.dim) < self.CR
                trial[mask] = mutant[mask]

                # Evaluate trial vector
                f = func(trial)
                self.budget -= 1
                if f < fitness[i]:
                    fitness[i] = f
                    pop[i] = trial.copy()

                    # Update best solution
                    if f < best_fitness:
                        best_fitness = f
                        best_solution = trial.copy()

            # Adaptive Local Search using Nelder-Mead
            if np.random.rand() < self.local_search_prob:
                index = np.random.randint(self.pop_size)
                initial_x = pop[index].copy()

                def obj_func(x):
                    return func(x)

                result = minimize(obj_func, initial_x, method='Nelder-Mead', bounds=func.bounds)

                if result.fun < fitness[index]:
                    fitness[index] = result.fun
                    pop[index] = result.x.copy()
                    
                    if result.fun < best_fitness:
                        best_fitness = result.fun
                        best_solution = result.x.copy()

                self.budget -= result.nfev  # Account for function evaluations in Nelder-Mead
                if self.budget <= 0:
                    break
            
        return best_fitness, best_solution