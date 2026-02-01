import numpy as np

class RingCauchyAdaptiveDE:
    def __init__(self, budget=10000, dim=10, pop_size=40, Cr=0.9, F_initial=0.5, stagnation_threshold=100, restart_prob=0.1, local_search_prob=0.1, migration_interval=50):
        self.budget = budget
        self.dim = dim
        self.pop_size = pop_size
        self.Cr = Cr
        self.F = F_initial
        self.stagnation_threshold = stagnation_threshold
        self.restart_prob = restart_prob
        self.local_search_prob = local_search_prob
        self.migration_interval = migration_interval
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
                # Ring Topology based Mutation (Cauchy Mutation)
                idx_prev = (i - 1) % self.pop_size
                idx_next = (i + 1) % self.pop_size
                
                # Cauchy mutation
                cauchy_noise = self.F * np.random.standard_cauchy(size=self.dim)
                mutant = population[i] + cauchy_noise

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

            # Local Search
            if np.random.rand() < self.local_search_prob:
                idx = np.random.randint(0, self.pop_size)
                # Apply small perturbation to the selected individual
                population[idx] = population[idx] + 0.01 * np.random.normal(0, 1, self.dim)
                population[idx] = np.clip(population[idx], func.bounds.lb, func.bounds.ub)
                fitness[idx] = func(population[idx])
                self.budget -= 1

                if fitness[idx] < self.f_opt:
                    self.f_opt = fitness[idx]
                    self.x_opt = population[idx]
                    self.last_improvement = self.generation
                    
            self.generation += 1
        
        return self.f_opt, self.x_opt