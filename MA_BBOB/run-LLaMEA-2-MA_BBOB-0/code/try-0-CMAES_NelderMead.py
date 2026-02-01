import numpy as np
from scipy.optimize import minimize

class CMAES_NelderMead:
    def __init__(self, budget=10000, dim=10, cmaes_popsize=None, nelder_mead_iters=5):
        self.budget = budget
        self.dim = dim
        self.cmaes_popsize = cmaes_popsize if cmaes_popsize is not None else 4 + int(3 * np.log(dim))
        self.nelder_mead_iters = nelder_mead_iters
        self.x_opt = None
        self.f_opt = np.inf

    def __call__(self, func):
        # Initial CMA-ES population
        mean = np.random.uniform(func.bounds.lb, func.bounds.ub, size=self.dim)
        sigma = 0.5  # Initial step size
        population = np.random.normal(mean, sigma, size=(self.cmaes_popsize, self.dim))
        population = np.clip(population, func.bounds.lb, func.bounds.ub)

        fitness = np.array([func(x) for x in population])
        self.budget -= self.cmaes_popsize

        best_index = np.argmin(fitness)
        self.x_opt = population[best_index]
        self.f_opt = fitness[best_index]

        # CMA-ES loop
        while self.budget > 0:
            # Sample new population
            population = np.random.normal(mean, sigma, size=(self.cmaes_popsize, self.dim))
            population = np.clip(population, func.bounds.lb, func.bounds.ub)

            fitness = np.array([func(x) for x in population])
            self.budget -= self.cmaes_popsize

            # Update best solution
            best_index = np.argmin(fitness)
            if fitness[best_index] < self.f_opt:
                self.f_opt = fitness[best_index]
                self.x_opt = population[best_index]

            # Update mean and sigma (simplified CMA-ES update)
            mean = np.mean(population, axis=0)
            sigma *= np.exp(0.1 * (np.std(fitness) / np.mean(fitness) - 1))

            # Local search with Nelder-Mead every few iterations (e.g., every 10 CMA-ES steps)
            if self.budget > 0 and self.budget % (10 * self.cmaes_popsize) <= self.cmaes_popsize and self.nelder_mead_iters > 0:
                # Use Nelder-Mead starting from the best CMA-ES point
                result = minimize(func, self.x_opt, method='Nelder-Mead', 
                                  options={'maxiter': self.nelder_mead_iters})  # Limit iterations to save budget

                if result.fun < self.f_opt:
                    self.f_opt = result.fun
                    self.x_opt = result.x
                self.budget -= result.nit # Number of iterations that Nelder-Mead performed.  
                if self.budget < 0:
                   break;       
        return self.f_opt, self.x_opt