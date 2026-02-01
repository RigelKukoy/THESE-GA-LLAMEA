import numpy as np

class EnhancedDE:
    def __init__(self, budget=10000, dim=10, pop_size=40, F_min=0.1, F_max=0.9, Cr=0.9, stagnation_threshold=100, local_search_prob=0.1, local_search_radius=0.1):
        self.budget = budget
        self.dim = dim
        self.pop_size = pop_size
        self.F_min = F_min
        self.F_max = F_max
        self.Cr = Cr
        self.stagnation_threshold = stagnation_threshold
        self.local_search_prob = local_search_prob
        self.local_search_radius = local_search_radius
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
                # Self-adaptive Mutation Factor
                F = np.random.uniform(self.F_min, self.F_max)

                # Mutation
                indices = [j for j in range(self.pop_size) if j != i]
                idxs = np.random.choice(indices, size=3, replace=False)
                x_r1, x_r2, x_r3 = population[idxs[0]], population[idxs[1]], population[idxs[2]]
                
                mutant = population[i] + F * (x_r1 - x_r2) + F * (x_r3 - population[i])
                
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
            
            # Stagnation check and local search
            if (generation - self.last_improvement) > self.stagnation_threshold:
                if np.random.rand() < self.local_search_prob:
                    # Apply local search to the best individual
                    x_local_search = self.x_opt + np.random.uniform(-self.local_search_radius, self.local_search_radius, size=self.dim)
                    x_local_search = np.clip(x_local_search, func.bounds.lb, func.bounds.ub)
                    f_local_search = func(x_local_search)
                    self.budget -= 1
                    
                    if f_local_search < self.f_opt:
                        self.f_opt = f_local_search
                        self.x_opt = x_local_search
                        self.last_improvement = generation
                        
            generation += 1
        
        return self.f_opt, self.x_opt