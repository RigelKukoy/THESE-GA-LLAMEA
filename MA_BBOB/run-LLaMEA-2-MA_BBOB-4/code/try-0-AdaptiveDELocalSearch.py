import numpy as np

class AdaptiveDELocalSearch:
    def __init__(self, budget=10000, dim=10, pop_size=40, F=0.5, Cr=0.9, exploration_decay=0.995, local_search_prob=0.1, local_search_radius=0.1):
        self.budget = budget
        self.dim = dim
        self.pop_size = pop_size
        self.F = F
        self.Cr = Cr
        self.exploration_decay = exploration_decay
        self.local_search_prob = local_search_prob
        self.local_search_radius = local_search_radius

    def __call__(self, func):
        # Initialization
        population = np.random.uniform(func.bounds.lb, func.bounds.ub, size=(self.pop_size, self.dim))
        fitness = np.array([func(x) for x in population])
        self.budget -= self.pop_size
        
        self.f_opt = np.min(fitness)
        self.x_opt = population[np.argmin(fitness)]
        
        exploration_rate = 1.0 
        
        while self.budget > self.pop_size:
            # Mutation and Crossover
            new_population = np.copy(population)
            for i in range(self.pop_size):
                # Mutation
                indices = [j for j in range(self.pop_size) if j != i]
                idxs = np.random.choice(indices, size=3, replace=False)
                x_r1, x_r2, x_r3 = population[idxs[0]], population[idxs[1]], population[idxs[2]]
                
                mutant = population[i] + self.F * (x_r1 - x_r2) + self.F * exploration_rate * (x_r3 - population[i])
                
                # Crossover
                for j in range(self.dim):
                    if np.random.rand() < self.Cr:
                        new_population[i, j] = mutant[j]
                    else:
                        new_population[i, j] = population[i, j]
                
                new_population[i] = np.clip(new_population[i], func.bounds.lb, func.bounds.ub)
                
                # Local Search
                if np.random.rand() < self.local_search_prob:
                    new_x = new_population[i] + np.random.uniform(-self.local_search_radius, self.local_search_radius, size=self.dim)
                    new_x = np.clip(new_x, func.bounds.lb, func.bounds.ub)
                    new_population[i] = new_x
            
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
            
            # Decay exploration rate
            exploration_rate *= self.exploration_decay
        
        return self.f_opt, self.x_opt