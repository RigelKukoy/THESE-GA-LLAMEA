import numpy as np

class AdaptiveDEAR:
    def __init__(self, budget=10000, dim=10, pop_size=40, F=0.5, Cr=0.9, archive_size=10, restart_trigger=0.001):
        self.budget = budget
        self.dim = dim
        self.pop_size = pop_size
        self.F = F
        self.Cr = Cr
        self.archive_size = archive_size
        self.restart_trigger = restart_trigger

    def __call__(self, func):
        # Initialization
        population = np.random.uniform(func.bounds.lb, func.bounds.ub, size=(self.pop_size, self.dim))
        fitness = np.array([func(x) for x in population])
        self.budget -= self.pop_size
        
        archive = []
        
        self.f_opt = np.min(fitness)
        self.x_opt = population[np.argmin(fitness)]
        
        iteration = 0
        
        while self.budget > self.pop_size:
            iteration += 1
            
            # Mutation and Crossover
            new_population = np.copy(population)
            for i in range(self.pop_size):
                # Mutation
                indices = [j for j in range(self.pop_size) if j != i]
                if len(archive) > 0 and np.random.rand() < 0.1: # Use archive sometimes
                    use_archive = True
                    a_idx = np.random.randint(len(archive))
                    x_r1 = archive[a_idx]
                    idxs = np.random.choice(indices, size=2, replace=False)
                    x_r2, x_r3 = population[idxs[0]], population[idxs[1]]
                    
                else:
                    use_archive = False
                    idxs = np.random.choice(indices, size=3, replace=False)
                    x_r1, x_r2, x_r3 = population[idxs[0]], population[idxs[1]], population[idxs[2]]

                mutant = population[i] + self.F * (x_r1 - x_r2) + self.F * (x_r3 - population[i]) if not use_archive else population[i] + self.F * (x_r1 - x_r2)
                
                # Crossover
                for j in range(self.dim):
                    if np.random.rand() < self.Cr:
                        new_population[i, j] = mutant[j]
                    else:
                        new_population[i, j] = population[i, j]
                
                new_population[i] = np.clip(new_population[i], func.bounds.lb, func.bounds.ub)
            
            # Evaluation
            new_fitness = np.array([func(x) for x in new_population])
            self.budget -= self.pop_size
            
            # Selection
            for i in range(self.pop_size):
                if new_fitness[i] < fitness[i]:
                    
                    # Update archive
                    if len(archive) < self.archive_size:
                        archive.append(population[i].copy())
                    else:
                        idx_to_replace = np.random.randint(self.archive_size)
                        archive[idx_to_replace] = population[i].copy()
                    
                    population[i] = new_population[i]
                    fitness[i] = new_fitness[i]
                    
                    if new_fitness[i] < self.f_opt:
                        self.f_opt = new_fitness[i]
                        self.x_opt = new_population[i]
                        
            # Restart Mechanism
            if np.std(fitness) < self.restart_trigger:
                population = np.random.uniform(func.bounds.lb, func.bounds.ub, size=(self.pop_size, self.dim))
                fitness = np.array([func(x) for x in population])
                self.budget -= self.pop_size

                if np.min(fitness) < self.f_opt:
                    self.f_opt = np.min(fitness)
                    self.x_opt = population[np.argmin(fitness)]
        
        return self.f_opt, self.x_opt