import numpy as np

class RingTopologyAdaptiveDE:
    def __init__(self, budget=10000, dim=10, pop_size=50, ring_neighbors=3, lr_F=0.1, lr_CR=0.1):
        self.budget = budget
        self.dim = dim
        self.pop_size = pop_size
        self.ring_neighbors = ring_neighbors
        self.lr_F = lr_F
        self.lr_CR = lr_CR
        self.F = np.full(pop_size, 0.5)
        self.CR = np.full(pop_size, 0.9)

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
            for i in range(self.pop_size):
                # Ring Topology Selection
                neighbors = [(i + j) % self.pop_size for j in range(1, self.ring_neighbors + 1)]
                
                # Select best neighbor for mutation
                best_neighbor = i
                for neighbor in neighbors:
                    if fitness[neighbor] < fitness[best_neighbor]:
                        best_neighbor = neighbor
                
                # Differential Evolution
                idxs = np.random.choice(self.pop_size, 2, replace=False)
                x_r1, x_r2 = self.population[idxs]
                mutant = self.population[i] + self.F[i] * (self.population[best_neighbor] - self.population[i]) + self.F[i] * (x_r1 - x_r2)
                mutant = np.clip(mutant, func.bounds.lb, func.bounds.ub)
                
                crossover = np.random.uniform(size=self.dim) < self.CR[i]
                trial = np.where(crossover, mutant, self.population[i])
                
                # Selection
                f_trial = func(trial)
                self.budget -= 1
                
                if f_trial < self.f_opt:
                    self.f_opt = f_trial
                    self.x_opt = trial

                if f_trial < fitness[i]:
                    # Adaptive Parameter Control
                    delta_f = fitness[i] - f_trial
                    self.F[i] = max(0, min(1, self.F[i] + self.lr_F * delta_f, 1.0))
                    self.CR[i] = max(0, min(1, self.CR[i] + self.lr_CR * delta_f, 1.0))
                    
                    fitness[i] = f_trial
                    self.population[i] = trial

        return self.f_opt, self.x_opt