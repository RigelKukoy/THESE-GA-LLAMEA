import numpy as np
from scipy.optimize import minimize

class DE_NM:
    def __init__(self, budget=10000, dim=10, pop_size=20, F=0.5, CR=0.7, nm_iterations=5):
        self.budget = budget
        self.dim = dim
        self.pop_size = pop_size
        self.F = F
        self.CR = CR
        self.nm_iterations = nm_iterations

    def __call__(self, func):
        # Initialize population
        pop = np.random.uniform(func.bounds.lb, func.bounds.ub, size=(self.pop_size, self.dim))
        fitness = np.array([func(x) for x in pop])
        self.budget -= self.pop_size

        # Find initial best
        best_index = np.argmin(fitness)
        best_x = pop[best_index].copy()
        best_f = fitness[best_index]

        while self.budget > 0:
            for i in range(self.pop_size):
                # Differential Evolution mutation and crossover
                idxs = [idx for idx in range(self.pop_size) if idx != i]
                a, b, c = np.random.choice(idxs, 3, replace=False)
                mutant = pop[a] + self.F * (pop[b] - pop[c])
                mutant = np.clip(mutant, func.bounds.lb, func.bounds.ub)

                trial = np.copy(pop[i])
                cross_points = np.random.rand(self.dim) < self.CR
                if not np.any(cross_points):
                    cross_points[np.random.randint(0, self.dim)] = True
                trial[cross_points] = mutant[cross_points]

                # Evaluate trial vector
                f = func(trial)
                self.budget -= 1
                if f < fitness[i]:
                    fitness[i] = f
                    pop[i] = trial
                    # Nelder-Mead local search around the improved solution
                    if self.budget > 0 and self.nm_iterations > 0:
                      nm_result = minimize(func, pop[i], method='Nelder-Mead', options={'maxiter': self.nm_iterations})
                      if nm_result.success:
                        f_nm = nm_result.fun
                        x_nm = nm_result.x
                        
                        num_evals = nm_result.nfev
                        self.budget -= num_evals

                        if f_nm < fitness[i]:
                            fitness[i] = f_nm
                            pop[i] = x_nm
                            f = f_nm # Update f to the improved fitness for DE update

                # Update best solution
                if f < best_f:
                    best_f = f
                    best_x = pop[i].copy()

        return best_f, best_x