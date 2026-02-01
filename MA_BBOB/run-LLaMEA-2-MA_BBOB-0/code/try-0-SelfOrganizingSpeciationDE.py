import numpy as np

class SelfOrganizingSpeciationDE:
    def __init__(self, budget=10000, dim=10, pop_size=50, num_species=5, lr_F=0.1, lr_CR=0.1, local_search_prob=0.1):
        self.budget = budget
        self.dim = dim
        self.pop_size = pop_size
        self.num_species = num_species
        self.lr_F = lr_F
        self.lr_CR = lr_CR
        self.local_search_prob = local_search_prob
        self.F = np.full(num_species, 0.5)
        self.CR = np.full(num_species, 0.9)
        self.species = None
        self.centroids = None

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
        
        # Initialize Species
        self.centroids = np.random.uniform(func.bounds.lb, func.bounds.ub, size=(self.num_species, self.dim))
        self.assign_species()

        while self.budget > 0:
            for i in range(self.pop_size):
                species_id = self.species[i]

                # Differential Evolution
                idxs = np.random.choice(self.pop_size, 3, replace=False)
                x_r1, x_r2, x_r3 = self.population[idxs]
                mutant = x_r1 + self.F[species_id] * (x_r2 - x_r3)
                mutant = np.clip(mutant, func.bounds.lb, func.bounds.ub)
                
                crossover = np.random.uniform(size=self.dim) < self.CR[species_id]
                trial = np.where(crossover, mutant, self.population[i])
                
                # Local Search
                if np.random.rand() < self.local_search_prob:
                    trial = self.local_search(trial, func.bounds.lb, func.bounds.ub)
                
                # Selection
                f_trial = func(trial)
                self.budget -= 1
                
                if f_trial < self.f_opt:
                    self.f_opt = f_trial
                    self.x_opt = trial

                if f_trial < fitness[i]:
                    # Adaptive Parameter Control
                    delta_f = fitness[i] - f_trial
                    self.F[species_id] = max(0, min(1, self.F[species_id] + self.lr_F * delta_f))
                    self.CR[species_id] = max(0, min(1, self.CR[species_id] + self.lr_CR * delta_f))
                    
                    fitness[i] = f_trial
                    self.population[i] = trial
            
            # Re-assign Species periodically
            if self.budget % (self.pop_size) == 0:
                self.assign_species()

        return self.f_opt, self.x_opt

    def assign_species(self):
        self.species = np.zeros(self.pop_size, dtype=int)
        for i in range(self.pop_size):
            distances = np.linalg.norm(self.population[i] - self.centroids, axis=1)
            self.species[i] = np.argmin(distances)

    def local_search(self, x, lb, ub, step_size=0.1):
        # Simple local search around x
        x_new = x + np.random.uniform(-step_size, step_size, size=self.dim)
        return np.clip(x_new, lb, ub)