import numpy as np

class SelfAdaptiveDE_GLS:
    def __init__(self, budget=10000, dim=10, pop_size=50, initial_F=0.5, initial_CR=0.9, local_search_prob=0.1, local_search_sigma=0.1):
        self.budget = budget
        self.dim = dim
        self.pop_size = pop_size
        self.F = initial_F * np.ones(pop_size)
        self.CR = initial_CR * np.ones(pop_size)
        self.local_search_prob = local_search_prob
        self.local_search_sigma = local_search_sigma
        self.lb = None
        self.ub = None

    def __call__(self, func):
        self.f_opt = np.Inf
        self.x_opt = None
        self.lb = func.bounds.lb
        self.ub = func.bounds.ub
        self.population = np.random.uniform(self.lb, self.ub, size=(self.pop_size, self.dim))
        fitness = np.array([func(x) for x in self.population])
        self.budget -= self.pop_size
        
        for i in range(self.pop_size):
            if fitness[i] < self.f_opt:
                self.f_opt = fitness[i]
                self.x_opt = self.population[i]

        while self.budget > 0:
            new_population = np.copy(self.population)
            new_fitness = np.copy(fitness)

            for i in range(self.pop_size):
                # Self-adaptive parameters
                self.F[i] = np.clip(self.F[i] + 0.1 * np.random.randn(), 0.1, 0.9)
                self.CR[i] = np.clip(self.CR[i] + 0.1 * np.random.randn(), 0.1, 0.9)

                # Differential Evolution
                idxs = np.random.choice(self.pop_size, 3, replace=False)
                x_r1, x_r2, x_r3 = self.population[idxs]
                mutant = x_r1 + self.F[i] * (x_r2 - x_r3)
                mutant = np.clip(mutant, self.lb, self.ub)
                
                crossover = np.random.uniform(size=self.dim) < self.CR[i]
                trial = np.where(crossover, mutant, self.population[i])
                
                # Local Search
                if np.random.rand() < self.local_search_prob:
                    trial = trial + self.local_search_sigma * np.random.randn(self.dim)
                    trial = np.clip(trial, self.lb, self.ub)
                
                # Selection
                f_trial = func(trial)
                self.budget -= 1
                
                if f_trial < self.f_opt:
                    self.f_opt = f_trial
                    self.x_opt = trial
                    
                if f_trial < fitness[i]:
                    new_fitness[i] = f_trial
                    new_population[i] = trial

            self.population = new_population
            fitness = new_fitness

        return self.f_opt, self.x_opt