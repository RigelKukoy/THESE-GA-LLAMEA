import numpy as np

class AdaptiveDERestart:
    def __init__(self, budget=10000, dim=10, pop_size=40, F=0.5, Cr=0.9, restart_prob=0.05, stagnation_threshold=100, pop_size_reduction_factor=0.9):
        self.budget = budget
        self.dim = dim
        self.pop_size = pop_size
        self.F = F
        self.Cr = Cr
        self.restart_prob = restart_prob
        self.stagnation_threshold = stagnation_threshold
        self.pop_size_reduction_factor = pop_size_reduction_factor
        self.best_fitness_history = []
        self.last_improvement = 0

    def __call__(self, func):
        # Initialization
        population = np.random.uniform(func.bounds.lb, func.bounds.ub, size=(self.pop_size, self.dim))
        fitness = np.array([func(x) for x in population])
        self.budget -= self.pop_size
        
        self.f_opt = np.min(fitness)
        self.x_opt = population[np.argmin(fitness)]
        self.best_fitness_history.append(self.f_opt)
        
        generation = 0

        while self.budget > self.pop_size:
            # Mutation and Crossover
            new_population = np.copy(population)
            for i in range(self.pop_size):
                # Mutation
                indices = [j for j in range(self.pop_size) if j != i]
                idxs = np.random.choice(indices, size=3, replace=False)
                x_r1, x_r2, x_r3 = population[idxs[0]], population[idxs[1]], population[idxs[2]]
                
                mutant = population[i] + self.F * (x_r1 - x_r2) + self.F * (x_r3 - population[i])
                
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
                    population[i] = new_population[i]
                    fitness[i] = new_fitness[i]
                    
                    if new_fitness[i] < self.f_opt:
                        self.f_opt = new_fitness[i]
                        self.x_opt = new_population[i]
                        self.last_improvement = generation
                        
            self.best_fitness_history.append(self.f_opt)
            
            # Stagnation check and restart
            if (generation - self.last_improvement) > self.stagnation_threshold:
                if np.random.rand() < self.restart_prob:
                    # Restart: Re-initialize a portion of the population
                    num_restart = int(self.pop_size * 0.2)
                    restart_indices = np.random.choice(self.pop_size, size=num_restart, replace=False)
                    population[restart_indices] = np.random.uniform(func.bounds.lb, func.bounds.ub, size=(num_restart, self.dim))
                    fitness[restart_indices] = np.array([func(x) for x in population[restart_indices]])
                    self.budget -= num_restart # Adjust budget for new evaluations

                    current_best_idx = np.argmin(fitness)
                    if fitness[current_best_idx] < self.f_opt:
                        self.f_opt = fitness[current_best_idx]
                        self.x_opt = population[current_best_idx]
                    
                    self.last_improvement = generation

                #Dynamic Population Size Reduction:
                else:
                    self.pop_size = max(int(self.pop_size * self.pop_size_reduction_factor), 10) #Ensure minimum population of 10
                    population = population[:self.pop_size]
                    fitness = fitness[:self.pop_size]

            generation += 1
        
        return self.f_opt, self.x_opt