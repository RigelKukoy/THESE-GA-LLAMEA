import numpy as np

class CooperativeDE:
    def __init__(self, budget=10000, dim=10, pop_size_min=10, pop_size_max=80, F=0.5, Cr=0.9, good_fitness_share=0.25, stagnation_threshold=10):
        self.budget = budget
        self.dim = dim
        self.pop_size_min = pop_size_min
        self.pop_size_max = pop_size_max
        self.pop_size = pop_size_max # Initial population size
        self.F = F
        self.Cr = Cr
        self.good_fitness_share = good_fitness_share
        self.stagnation_threshold = stagnation_threshold
        self.stagnation_counter = 0
        self.previous_best_fitness = np.inf

    def __call__(self, func):
        # Initialization
        population = np.random.uniform(func.bounds.lb, func.bounds.ub, size=(self.pop_size, self.dim))
        fitness = np.array([func(x) for x in population])
        self.budget -= self.pop_size
        
        self.f_opt = np.min(fitness)
        self.x_opt = population[np.argmin(fitness)]
        
        while self.budget > self.pop_size_min:
            # Mutation and Crossover
            new_population = np.copy(population)
            for i in range(self.pop_size):
                # Mutation
                indices = [j for j in range(self.pop_size) if j != i]
                idxs = np.random.choice(indices, size=3, replace=False)
                x_r1, x_r2, x_r3 = population[idxs[0]], population[idxs[1]], population[idxs[2]]
                
                mutant = population[i] + self.F * (x_r1 - x_r2)
                mutant = np.clip(mutant, func.bounds.lb, func.bounds.ub)
                
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
            
            # Stochastic Ranking
            combined_fitness = np.concatenate((fitness, new_fitness))
            combined_population = np.vstack((population, new_population))
            
            ranked_indices = np.argsort(combined_fitness)
            
            # Adaptive Population Sizing
            good_count = int(self.good_fitness_share * self.pop_size)
            
            population = combined_population[ranked_indices[:self.pop_size]]
            fitness = combined_fitness[ranked_indices[:self.pop_size]]
        
            # Best solution update
            if np.min(fitness) < self.f_opt:
                self.f_opt = np.min(fitness)
                self.x_opt = population[np.argmin(fitness)]
                self.stagnation_counter = 0
            else:
                self.stagnation_counter += 1

            # Adjust population size based on stagnation
            if self.stagnation_counter > self.stagnation_threshold:
                self.pop_size = max(self.pop_size_min, int(self.pop_size * 0.8)) # Reduce population size
                self.stagnation_counter = 0
            elif self.pop_size < self.pop_size_max and np.random.rand() < 0.1:  # Increase population size occasionally
                self.pop_size = min(self.pop_size_max, int(self.pop_size * 1.2))

            # Re-initialize population if stagnation is too high and budget allows
            if self.stagnation_counter > 3 * self.stagnation_threshold and self.budget > self.pop_size:
                population = np.random.uniform(func.bounds.lb, func.bounds.ub, size=(self.pop_size, self.dim))
                fitness = np.array([func(x) for x in population])
                self.budget -= self.pop_size
                self.stagnation_counter = 0
        
        return self.f_opt, self.x_opt