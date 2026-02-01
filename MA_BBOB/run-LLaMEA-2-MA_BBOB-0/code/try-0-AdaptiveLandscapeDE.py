import numpy as np

class AdaptiveLandscapeDE:
    def __init__(self, budget=10000, dim=10, pop_size=50, local_search_iterations=5):
        self.budget = budget
        self.dim = dim
        self.pop_size = pop_size
        self.local_search_iterations = local_search_iterations
        self.F = 0.5
        self.CR = 0.9
        self.smoothness = 0.5  # Initial guess; updated dynamically
        self.min_F = 0.1
        self.max_F = 0.9
        self.min_CR = 0.1
        self.max_CR = 0.9

    def __call__(self, func):
        self.f_opt = np.Inf
        self.x_opt = None
        self.population = np.random.uniform(func.bounds.lb, func.bounds.ub, size=(self.pop_size, self.dim))
        fitness = np.array([func(x) for x in self.population])
        self.budget -= self.pop_size
        
        for i in range(self.pop_size):
            if fitness[i] < self.f_opt:
                self.f_opt = fitness[i]
                self.x_opt = self.population[i]

        while self.budget > 0:
            # Landscape Analysis (Simple estimation of ruggedness)
            fitness_std = np.std(fitness)
            # Update smoothness parameter based on the fitness landscape. Higher std means more rugged.
            self.smoothness = 1.0 / (1.0 + fitness_std) # smoothness is between 0 and 1. close to 1 is smooth

            # Adaptive Parameter Control
            self.F = self.min_F + (self.max_F - self.min_F) * self.smoothness
            self.CR = self.min_CR + (self.max_CR - self.min_CR) * self.smoothness


            for i in range(self.pop_size):
                # Mutation
                idxs = np.random.choice(self.pop_size, 3, replace=False)
                x_r1, x_r2, x_r3 = self.population[idxs]
                mutant = x_r1 + self.F * (x_r2 - x_r3)
                mutant = np.clip(mutant, func.bounds.lb, func.bounds.ub)

                # Crossover
                crossover = np.random.uniform(size=self.dim) < self.CR
                trial = np.where(crossover, mutant, self.population[i])

                # Selection
                f_trial = func(trial)
                self.budget -= 1
                
                if f_trial < self.f_opt:
                    self.f_opt = f_trial
                    self.x_opt = trial
                
                # Local Search
                if f_trial < fitness[i]:
                    # Perform local search around the new solution
                    x_local = self.local_search(trial, func, self.local_search_iterations)
                    f_local = func(x_local)
                    self.budget -= 1  # Local search counts as function evaluations

                    if f_local < self.f_opt:
                        self.f_opt = f_local
                        self.x_opt = x_local

                    if f_local < f_trial:
                         f_trial = f_local
                         trial = x_local #replace trial

                    if f_trial < fitness[i]:
                        fitness[i] = f_trial
                        self.population[i] = trial
        return self.f_opt, self.x_opt

    def local_search(self, x, func, iterations):
        x_best = x.copy()
        f_best = func(x)

        for _ in range(iterations):
            # Generate a neighbor by adding a small random perturbation
            x_neighbor = x_best + np.random.normal(0, 0.1, size=self.dim)  # Adjust step size (0.1) as needed
            x_neighbor = np.clip(x_neighbor, func.bounds.lb, func.bounds.ub)
            f_neighbor = func(x_neighbor)
            
            if f_neighbor < f_best:
                f_best = f_neighbor
                x_best = x_neighbor.copy()
        return x_best