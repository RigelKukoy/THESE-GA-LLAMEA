import numpy as np

class ProbabilisticRingOrthoDE:
    def __init__(self, budget=10000, dim=10, pop_size=40, Cr=0.9, F_initial=0.5, stagnation_threshold=100, restart_prob=0.1, ortho_prob=0.1, migration_interval=50, cauchy_prob=0.5):
        self.budget = budget
        self.dim = dim
        self.pop_size = pop_size
        self.Cr = Cr
        self.F = F_initial
        self.stagnation_threshold = stagnation_threshold
        self.restart_prob = restart_prob
        self.ortho_prob = ortho_prob
        self.migration_interval = migration_interval
        self.cauchy_prob = cauchy_prob
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

        while self.budget > self.pop_size:
            # Mutation and Crossover
            new_population = np.copy(population)
            for i in range(self.pop_size):
                # Probabilistic Ring Topology
                neighbors = []
                if np.random.rand() < 0.7:  # 70% chance to include left neighbor
                    neighbors.append((i - 1) % self.pop_size)
                if np.random.rand() < 0.7:  # 70% chance to include right neighbor
                    neighbors.append((i + 1) % self.pop_size)
                if not neighbors:
                    neighbors.append(np.random.randint(0, self.pop_size)) # Ensure at least one neighbor

                # Combined Cauchy-Gaussian Mutation
                if np.random.rand() < self.cauchy_prob:
                    mutation_noise = self.F * np.random.standard_cauchy(size=self.dim)  # Cauchy
                else:
                    mutation_noise = self.F * np.random.normal(0, 1, size=self.dim)  # Gaussian

                mutant = population[i] + mutation_noise

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
            
            # Stagnation check and restart
            if (self.generation - self.last_improvement) > self.stagnation_threshold:
                if np.random.rand() < self.restart_prob:
                    # Restart the population
                    population = np.random.uniform(func.bounds.lb, func.bounds.ub, size=(self.pop_size, self.dim))
                    fitness = np.array([func(x) for x in population])
                    self.budget -= self.pop_size
                    self.f_opt = np.min(fitness)
                    self.x_opt = population[np.argmin(fitness)]
                    self.last_improvement = self.generation
                    self.F = 0.5 #reset F
                else:
                    # Adaptive F: Reduce mutation strength upon stagnation
                    self.F *= 0.9  # Reduce F, but prevent it from becoming zero.
                    self.F = max(self.F, 0.1)
            
            # Migration strategy
            if self.generation % self.migration_interval == 0:
                ranked_indices = np.argsort(fitness)
                # Replace worst individuals with slightly perturbed best individuals
                for i in range(self.pop_size // 4): #Migrate 25% of population
                    worst_idx = ranked_indices[self.pop_size - 1 - i]
                    best_idx = ranked_indices[0]
                    population[worst_idx] = population[best_idx] + 0.05 * np.random.normal(0, 1, self.dim)
                    population[worst_idx] = np.clip(population[worst_idx], func.bounds.lb, func.bounds.ub)
                    fitness[worst_idx] = func(population[worst_idx])
                    self.budget -= 1

                    if fitness[worst_idx] < self.f_opt:
                        self.f_opt = fitness[worst_idx]
                        self.x_opt = population[worst_idx]
                        self.last_improvement = self.generation
            
            # Orthogonal Learning
            if np.random.rand() < self.ortho_prob:
                idx = np.random.randint(0, self.pop_size)
                
                # Generate orthogonal array (simplified - random sampling)
                levels = 3  # Number of levels for each factor
                factors = self.dim  # Number of factors (dimensions)
                orthogonal_array = np.random.randint(0, levels, size=(levels**2, factors)) # L9 array

                # Evaluate all points in the orthogonal array around the selected individual
                best_fitness_oa = fitness[idx]
                best_oa_point = population[idx]

                for oa_point in orthogonal_array:
                    # Map the levels to a perturbation around the current individual
                    perturbation = (oa_point - (levels - 1) / 2) * 0.05 # small perturbation
                    new_point = population[idx] + perturbation
                    new_point = np.clip(new_point, func.bounds.lb, func.bounds.ub)

                    new_fitness_oa = func(new_point)
                    self.budget -= 1

                    if new_fitness_oa < best_fitness_oa:
                        best_fitness_oa = new_fitness_oa
                        best_oa_point = new_point

                        if new_fitness_oa < self.f_opt:
                            self.f_opt = new_fitness_oa
                            self.x_opt = new_point
                            self.last_improvement = self.generation
                
                # Replace the individual with the best point found in the orthogonal array
                population[idx] = best_oa_point
                fitness[idx] = best_fitness_oa
            
            self.generation += 1
        
        return self.f_opt, self.x_opt