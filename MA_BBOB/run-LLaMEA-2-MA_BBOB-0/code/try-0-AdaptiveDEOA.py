import numpy as np
from scipy.optimize import minimize

class AdaptiveDEOA:
    def __init__(self, budget=10000, dim=10, pop_multiplier=5, initial_F=0.5, initial_CR=0.9, oa_design_points=10):
        self.budget = budget
        self.dim = dim
        self.pop_size = dim * pop_multiplier
        self.F = initial_F  # Initial mutation factor
        self.CR = initial_CR # Crossover rate
        self.lb = -5.0
        self.ub = 5.0
        self.x_opt = None
        self.f_opt = np.inf
        self.oa_design_points = oa_design_points # Number of design points for Orthogonal Array
        self.local_search_frequency = 10 # Frequency of local search
        self.local_search_radius = 0.1 # Radius for local search

    def generate_orthogonal_array(self, n, k, levels):
        """Generates an orthogonal array using a simple method."""
        oa = np.zeros((n, k), dtype=int)
        for i in range(n):
            for j in range(k):
                oa[i, j] = i % levels
        return oa

    def __call__(self, func):
        self.population = np.random.uniform(self.lb, self.ub, size=(self.pop_size, self.dim))
        self.fitness = np.array([func(x) for x in self.population])
        self.budget -= self.pop_size
        iteration = 0

        while self.budget > 0:
            # Sort population by fitness
            sorted_indices = np.argsort(self.fitness)
            self.population = self.population[sorted_indices]
            self.fitness = self.fitness[sorted_indices]

            if self.fitness[0] < self.f_opt:
                self.f_opt = self.fitness[0]
                self.x_opt = self.population[0]

            for i in range(self.pop_size):
                # Mutation using Orthogonal Array
                oa = self.generate_orthogonal_array(self.oa_design_points, self.dim, 3)
                mutant = np.copy(self.population[i])
                for j in range(self.dim):
                    if oa[i % self.oa_design_points, j] == 1:
                        mutant[j] = self.population[np.random.randint(self.pop_size)][j]  # Random individual
                    elif oa[i % self.oa_design_points, j] == 2:
                        mutant[j] = self.population[np.random.randint(self.pop_size)][j] + self.F * (self.population[np.random.randint(self.pop_size)][j] - self.population[i][j]) # DE Mutation

                mutant = np.clip(mutant, self.lb, self.ub)

                # Crossover
                crossover = np.random.rand(self.dim) < self.CR
                trial = np.where(crossover, mutant, self.population[i])

                # Selection
                f_trial = func(trial)
                self.budget -= 1
                if f_trial < self.fitness[i]:
                    self.fitness[i] = f_trial
                    self.population[i] = trial

                if self.budget <= 0:
                    break
            
            iteration += 1
            if iteration % self.local_search_frequency == 0:
                # Local search around the best solution
                best_index = np.argmin(self.fitness)
                x_best = self.population[best_index].copy()
                
                # Define bounds for local search
                bounds = [(max(self.lb, x_best[i] - self.local_search_radius), min(self.ub, x_best[i] + self.local_search_radius)) for i in range(self.dim)]
                
                # Perform local search
                res = minimize(func, x_best, method='L-BFGS-B', bounds=bounds, options={'maxfun': min(self.budget, self.pop_size)})
                
                if res.fun < self.f_opt:
                        self.f_opt = res.fun
                        self.x_opt = res.x

                self.budget -= min(self.budget, self.pop_size)

        return self.f_opt, self.x_opt