import numpy as np

class SANE:
    def __init__(self, budget=10000, dim=10, pop_size=40, F_init=0.5, Cr_init=0.9, neighborhood_size=5, cauchy_scale=0.1):
        self.budget = budget
        self.dim = dim
        self.pop_size = pop_size
        self.F_init = F_init
        self.Cr_init = Cr_init
        self.neighborhood_size = neighborhood_size
        self.cauchy_scale = cauchy_scale
        self.F = np.full(pop_size, F_init)
        self.Cr = np.full(pop_size, Cr_init)

    def __call__(self, func):
        # Initialization
        population = np.random.uniform(func.bounds.lb, func.bounds.ub, size=(self.pop_size, self.dim))
        fitness = np.array([func(x) for x in population])
        self.budget -= self.pop_size
        
        self.f_opt = np.min(fitness)
        self.x_opt = population[np.argmin(fitness)]
        
        while self.budget > self.pop_size:
            # Mutation and Crossover
            new_population = np.copy(population)
            for i in range(self.pop_size):
                # Adaptive F and Cr
                self.F[i] = np.clip(self.F[i] + 0.1 * np.random.normal(0, 1), 0.1, 1.0)
                self.Cr[i] = np.clip(self.Cr[i] + 0.1 * np.random.normal(0, 1), 0.1, 1.0)
                
                # Mutation
                indices = [j for j in range(self.pop_size) if j != i]
                idxs = np.random.choice(indices, size=3, replace=False)
                x_r1, x_r2, x_r3 = population[idxs[0]], population[idxs[1]], population[idxs[2]]

                mutant = population[i] + self.F[i] * (x_r1 - x_r2) + self.F[i] * (x_r3 - population[i])

                # Crossover
                for j in range(self.dim):
                    if np.random.rand() < self.Cr[i]:
                        new_population[i, j] = mutant[j]
                    else:
                        new_population[i, j] = population[i, j]
                
                new_population[i] = np.clip(new_population[i], func.bounds.lb, func.bounds.ub)

                # Neighborhood Search
                for _ in range(self.neighborhood_size):
                    neighbor = new_population[i] + self.cauchy_scale * np.random.standard_cauchy(size=self.dim)
                    neighbor = np.clip(neighbor, func.bounds.lb, func.bounds.ub)
                    neighbor_fitness = func(neighbor)
                    self.budget -= 1
                    if neighbor_fitness < func(new_population[i]):
                        new_population[i] = neighbor
                    if self.budget <= self.pop_size:
                        break
            
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
        
        return self.f_opt, self.x_opt