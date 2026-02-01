import numpy as np

class DynamicPopSizeAdaptiveDE:
    def __init__(self, budget=10000, dim=10, pop_size_initial=40, pop_size_min=10, pop_size_max=100, Cr_initial=0.9, F_initial=0.5, stagnation_threshold=100, pop_size_adjust_interval=50, diversity_threshold=0.1):
        self.budget = budget
        self.dim = dim
        self.pop_size = pop_initial
        self.pop_size_initial = pop_size_initial
        self.pop_size_min = pop_size_min
        self.pop_size_max = pop_size_max
        self.Cr = Cr_initial
        self.F = F_initial
        self.stagnation_threshold = stagnation_threshold
        self.pop_size_adjust_interval = pop_size_adjust_interval
        self.diversity_threshold = diversity_threshold
        self.best_fitness_history = []
        self.last_improvement = 0
        self.generation = 0

    def __call__(self, func):
        # Initialization
        population = np.random.uniform(func.bounds.lb, func.bounds.ub, size=(self.pop_size, self.dim))
        fitness = np.array([func(x) for x in population])
        self.budget -= self.pop_size
        
        self.f_opt = np.min(fitness)
        self.x_opt = population[np.argmin(fitness)]
        self.best_fitness_history.append(self.f_opt)
        self.last_improvement = 0
        self.generation = 0

        while self.budget > self.pop_size_min:
            # Mutation and Crossover
            new_population = np.copy(population)
            for i in range(self.pop_size):
                # DE mutation
                idxs = np.random.choice(self.pop_size, 3, replace=False)
                x_r1, x_r2, x_r3 = population[idxs]
                
                mutant = population[i] + self.F * (x_r1 - x_r2)

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
                        self.last_improvement = self.generation
                        
            self.best_fitness_history.append(self.f_opt)
            
            # Stagnation check
            if (self.generation - self.last_improvement) > self.stagnation_threshold:
                # Adaptive F and Cr: Reduce mutation strength and increase crossover rate upon stagnation
                self.F *= 0.8
                self.Cr = min(self.Cr * 1.2, 1.0)
                self.F = max(self.F, 0.1)

            # Population size adjustment
            if self.generation % self.pop_size_adjust_interval == 0:
                # Calculate fitness diversity
                fitness_range = np.max(fitness) - np.min(fitness)
                if fitness_range < self.diversity_threshold:
                    # Low diversity, increase population size to explore more
                    self.pop_size = min(self.pop_size + 10, self.pop_size_max)
                else:
                    # High diversity, decrease population size to exploit more
                    self.pop_size = max(self.pop_size - 5, self.pop_size_min)

                # Resize population
                if self.pop_size != population.shape[0]:
                    if self.pop_size > population.shape[0]:
                        # Add new random individuals
                        new_individuals = np.random.uniform(func.bounds.lb, func.bounds.ub, size=(self.pop_size - population.shape[0], self.dim))
                        population = np.vstack((population, new_individuals))
                        new_fitness = np.array([func(x) for x in new_individuals])
                        fitness = np.concatenate((fitness, new_fitness))
                        self.budget -= (self.pop_size - population.shape[0])

                    else:
                        # Remove worst individuals
                        ranked_indices = np.argsort(fitness)
                        population = population[ranked_indices[:self.pop_size]]
                        fitness = fitness[ranked_indices[:self.pop_size]]
            
            self.generation += 1
        
        return self.f_opt, self.x_opt