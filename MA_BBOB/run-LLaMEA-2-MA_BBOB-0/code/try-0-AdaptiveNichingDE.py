import numpy as np

class AdaptiveNichingDE:
    def __init__(self, budget=10000, dim=10, pop_multiplier=5, initial_F=0.5, initial_CR=0.9, niche_radius=0.1):
        self.budget = budget
        self.dim = dim
        self.pop_size = dim * pop_multiplier
        self.F = initial_F  # Initial mutation factor
        self.CR = initial_CR # Crossover rate
        self.niche_radius = niche_radius # Radius of the niche
        self.lb = -5.0
        self.ub = 5.0
        self.x_opt = None
        self.f_opt = np.inf
        self.F_history = []
        self.CR_history = []


    def __call__(self, func):
        self.population = np.random.uniform(self.lb, self.ub, size=(self.pop_size, self.dim))
        self.fitness = np.array([func(x) for x in self.population])
        self.budget -= self.pop_size

        while self.budget > 0:
            for i in range(self.pop_size):
                # Adaptive F and CR
                if len(self.F_history) > 0:
                    self.F = np.mean(self.F_history)
                if len(self.CR_history) > 0:
                    self.CR = np.mean(self.CR_history)
                
                # Mutation
                idxs = np.random.choice(self.pop_size, 3, replace=False)
                x_r1, x_r2, x_r3 = self.population[idxs]
                mutant = self.population[i] + self.F * (x_r2 - x_r3)
                mutant = np.clip(mutant, self.lb, self.ub)
                
                # Crossover
                crossover = np.random.rand(self.dim) < self.CR
                trial = np.where(crossover, mutant, self.population[i])

                # Selection
                f_trial = func(trial)
                self.budget -= 1

                #Niche Comparison: only accept if it is better AND far from existing solutions within the niche
                distance_to_others = np.linalg.norm(self.population - trial, axis=1)
                nearby_indices = np.where(distance_to_others < self.niche_radius)[0]
                
                
                if f_trial < self.fitness[i]:
                    is_better_than_nearby = True
                    for idx in nearby_indices:
                         if f_trial >= self.fitness[idx]:
                            is_better_than_nearby = False
                            break #Only is_better_than_nearby when strictly better
                    
                    if is_better_than_nearby:
                        
                        if self.budget > 0:
                            self.F_history.append(self.F)
                            self.CR_history.append(self.CR)
                        self.fitness[i] = f_trial
                        self.population[i] = trial

                        if f_trial < self.f_opt:
                            self.f_opt = f_trial
                            self.x_opt = trial
                

                if self.budget <= 0:
                    break

        return self.f_opt, self.x_opt