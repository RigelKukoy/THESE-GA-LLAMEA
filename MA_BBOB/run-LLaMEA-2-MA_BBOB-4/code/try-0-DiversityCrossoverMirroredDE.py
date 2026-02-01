import numpy as np

class DiversityCrossoverMirroredDE:
    def __init__(self, budget=10000, dim=10, pop_size=40, F=0.5, diversity_threshold=0.1, mirrored_sampling_prob=0.1):
        self.budget = budget
        self.dim = dim
        self.pop_size = pop_size
        self.F = F
        self.diversity_threshold = diversity_threshold
        self.mirrored_sampling_prob = mirrored_sampling_prob
        self.best_fitness_history = []

    def __call__(self, func):
        # Initialization
        population = np.random.uniform(func.bounds.lb, func.bounds.ub, size=(self.pop_size, self.dim))
        fitness = np.array([func(x) for x in population])
        self.budget -= self.pop_size
        
        self.f_opt = np.min(fitness)
        self.x_opt = population[np.argmin(fitness)]
        self.best_fitness_history.append(self.f_opt)
        
        while self.budget > self.pop_size:
            # Calculate population diversity
            diversity = np.std(population)
            
            # Adjust crossover rate based on diversity
            if diversity > self.diversity_threshold:
                Cr = 0.9  # High diversity, high crossover
            else:
                Cr = 0.3  # Low diversity, low crossover
                
            new_population = np.copy(population)
            for i in range(self.pop_size):
                # Mutation
                idxs = np.random.choice(self.pop_size, 3, replace=False)
                x_r1, x_r2, x_r3 = population[idxs]
                mutant = population[i] + self.F * (x_r2 - x_r3)
                mutant = np.clip(mutant, func.bounds.lb, func.bounds.ub)
                
                # Crossover
                for j in range(self.dim):
                    if np.random.rand() < Cr:
                        new_population[i, j] = mutant[j]
                    else:
                        new_population[i, j] = population[i, j]
                        
                # Mirrored Sampling
                if np.random.rand() < self.mirrored_sampling_prob:
                    for j in range(self.dim):
                        if new_population[i,j] < func.bounds.lb[j]:
                            new_population[i,j] = func.bounds.lb[j] + (func.bounds.lb[j] - new_population[i,j])
                        elif new_population[i,j] > func.bounds.ub[j]:
                            new_population[i,j] = func.bounds.ub[j] - (new_population[i,j] - func.bounds.ub[j])
                    new_population[i] = np.clip(new_population[i], func.bounds.lb, func.bounds.ub)

            # Evaluation
            new_fitness = np.array([func(x) for x in new_population])
            self.budget -= self.pop_size

            # Selection
            for i in range(self.pop_size):
                if new_fitness[i] < fitness[i]:
                    population[i] = new_population[i]
                    fitness[i] = new_fitness[i]
                    if new_fitness[i] < self.f_opt:
                        self.f_opt = new_fitness[i]
                        self.x_opt = new_population[i]
            self.best_fitness_history.append(self.f_opt)
        return self.f_opt, self.x_opt