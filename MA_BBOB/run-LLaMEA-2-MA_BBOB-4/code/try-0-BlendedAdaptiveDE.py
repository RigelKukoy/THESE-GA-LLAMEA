import numpy as np

class BlendedAdaptiveDE:
    def __init__(self, budget=10000, dim=10, pop_size=40, archive_size=10, crowding_threshold=0.01):
        self.budget = budget
        self.dim = dim
        self.pop_size = pop_size
        self.archive_size = archive_size
        self.crowding_threshold = crowding_threshold
        self.F = 0.5
        self.Cr = 0.9
        self.F_adapt_rate = 0.1
        self.Cr_adapt_rate = 0.1


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
            new_fitness = np.zeros(self.pop_size)
            for i in range(self.pop_size):
                # Mutation
                indices = [j for j in range(self.pop_size) if j != i]
                idxs = np.random.choice(indices, size=3, replace=False)
                x_r1, x_r2, x_r3 = population[idxs[0]], population[idxs[1]], population[idxs[2]]

                mutant = population[i] + self.F * (x_r1 - x_r2) + self.F * (x_r3 - population[i])
                mutant = np.clip(mutant, func.bounds.lb, func.bounds.ub)
                
                # Blended Crossover
                for j in range(self.dim):
                    if np.random.rand() < self.Cr:
                        alpha = np.random.rand()
                        new_population[i, j] = alpha * mutant[j] + (1 - alpha) * population[i, j]
                    else:
                        new_population[i, j] = population[i, j]

                new_population[i] = np.clip(new_population[i], func.bounds.lb, func.bounds.ub)
                new_fitness[i] = func(new_population[i])
                self.budget -= 1
                
            # Selection and Adaptation
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
                        
                    # Adaptive F and Cr
                    self.F = self.F * (1 - self.F_adapt_rate) + np.random.rand() * self.F_adapt_rate
                    self.Cr = self.Cr * (1 - self.Cr_adapt_rate) + np.random.rand() * self.Cr_adapt_rate

            # Diversity Maintenance (Crowding Distance)
            distances = np.zeros(self.pop_size)
            for i in range(self.pop_size):
                for j in range(self.pop_size):
                    if i != j:
                        distances[i] += np.linalg.norm(population[i] - population[j])

            # Remove individuals with low crowding distance if necessary
            if np.std(fitness) < self.crowding_threshold:
                worst_index = np.argmin(distances)
                population[worst_index] = np.random.uniform(func.bounds.lb, func.bounds.ub)
                fitness[worst_index] = func(population[worst_index])
                self.budget -= 1
                
        return self.f_opt, self.x_opt