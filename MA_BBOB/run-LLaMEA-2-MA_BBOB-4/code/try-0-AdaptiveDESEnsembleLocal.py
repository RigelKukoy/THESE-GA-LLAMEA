import numpy as np

class AdaptiveDESEnsembleLocal:
    def __init__(self, budget=10000, dim=10, pop_size=40, F_base=0.5, Cr_base=0.9, local_search_prob=0.1, diversity_threshold=0.1):
        self.budget = budget
        self.dim = dim
        self.pop_size = pop_size
        self.F_base = F_base
        self.Cr_base = Cr_base
        self.local_search_prob = local_search_prob
        self.diversity_threshold = diversity_threshold

    def __call__(self, func):
        # Initialization
        population = np.random.uniform(func.bounds.lb, func.bounds.ub, size=(self.pop_size, self.dim))
        fitness = np.array([func(x) for x in population])
        self.budget -= self.pop_size
        
        self.f_opt = np.min(fitness)
        self.x_opt = population[np.argmin(fitness)]
        
        iteration = 0
        
        while self.budget > self.pop_size:
            iteration += 1
            
            # Adaptive Mutation and Ensemble Crossover
            new_population = np.copy(population)
            new_fitness = np.zeros(self.pop_size)
            
            for i in range(self.pop_size):
                # Adaptive F (Mutation factor)
                F = self.F_base * np.random.uniform(0.5, 1.5)  # Self-adjusting F
                
                # Mutation
                indices = [j for j in range(self.pop_size) if j != i]
                idxs = np.random.choice(indices, size=3, replace=False)
                x_r1, x_r2, x_r3 = population[idxs[0]], population[idxs[1]], population[idxs[2]]
                mutant = population[i] + F * (x_r1 - x_r2) + F * (x_r3 - population[i])
                mutant = np.clip(mutant, func.bounds.lb, func.bounds.ub)

                # Ensemble Crossover: Choose between different Cr values
                Cr = self.Cr_base * np.random.uniform(0.5, 1.5)
                crossover_strategy = np.random.choice(['bin', 'exp'])
                
                if crossover_strategy == 'bin':  # Binomial Crossover
                    for j in range(self.dim):
                        if np.random.rand() < Cr or j == np.random.randint(self.dim):
                            new_population[i, j] = mutant[j]
                        else:
                            new_population[i, j] = population[i, j]
                else:  # Exponential Crossover
                    L = 0
                    while L < self.dim and np.random.rand() < Cr:
                        L += 1
                    for j in range(L):
                        new_population[i, (np.random.randint(self.dim) + j) % self.dim] = mutant[(np.random.randint(self.dim) + j) % self.dim]
                
                new_population[i] = np.clip(new_population[i], func.bounds.lb, func.bounds.ub)
                
                # Local Search
                if np.random.rand() < self.local_search_prob:
                    step_size = 0.01 * (func.bounds.ub - func.bounds.lb)
                    new_x = new_population[i] + np.random.uniform(-step_size, step_size, size=self.dim)
                    new_x = np.clip(new_x, func.bounds.lb, func.bounds.ub)
                    
                    new_f = func(new_x)
                    self.budget -= 1
                    
                    if new_f < func(new_population[i]):
                        new_population[i] = new_x
                        new_fitness[i] = new_f
                    else:
                        new_fitness[i] = func(new_population[i])
                        self.budget -= 1

                else:
                     new_fitness[i] = func(new_population[i])
                     self.budget -=1
                
            # Selection
            for i in range(self.pop_size):
                if new_fitness[i] < fitness[i]:
                    population[i] = new_population[i]
                    fitness[i] = new_fitness[i]
                    
                    if new_fitness[i] < self.f_opt:
                        self.f_opt = new_fitness[i]
                        self.x_opt = new_population[i]
            
            # Diversity Maintenance
            if np.std(fitness) < self.diversity_threshold:
                # Introduce new random solutions to increase diversity
                num_to_replace = int(self.pop_size * 0.2)  # Replace 20% of the population
                replace_indices = np.random.choice(self.pop_size, size=num_to_replace, replace=False)
                
                population[replace_indices] = np.random.uniform(func.bounds.lb, func.bounds.ub, size=(num_to_replace, self.dim))
                fitness[replace_indices] = np.array([func(x) for x in population[replace_indices]])
                self.budget -= num_to_replace

                if np.min(fitness) < self.f_opt:
                    self.f_opt = np.min(fitness)
                    self.x_opt = population[np.argmin(fitness)]
            
        return self.f_opt, self.x_opt