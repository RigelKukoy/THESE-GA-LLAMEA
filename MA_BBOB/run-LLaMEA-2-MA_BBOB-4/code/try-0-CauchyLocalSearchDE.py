import numpy as np

class CauchyLocalSearchDE:
    def __init__(self, budget=10000, dim=10, pop_size=40, Cr=0.9, F_initial=0.5, stagnation_threshold=100, local_search_prob=0.1, cauchy_scale_initial=1.0):
        self.budget = budget
        self.dim = dim
        self.pop_size = pop_size
        self.Cr = Cr
        self.F = F_initial
        self.stagnation_threshold = stagnation_threshold
        self.local_search_prob = local_search_prob
        self.best_fitness_history = []
        self.last_improvement = 0
        self.generation = 0
        self.cauchy_scale = cauchy_scale_initial # Initial scale for Cauchy distribution

    def local_search(self, x, func, scale=0.1):
        """Performs a local search around x using Cauchy mutations."""
        x_new = x + scale * np.random.standard_cauchy(size=self.dim)
        x_new = np.clip(x_new, func.bounds.lb, func.bounds.ub)
        f_new = func(x_new)
        self.budget -=1
        if f_new < func(x):
            return x_new, f_new
        else:
            return x, func(x)

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
                # Cauchy Mutation with adaptive step size
                indices = [j for j in range(self.pop_size) if j != i]
                idxs = np.random.choice(indices, size=3, replace=False)
                x_r1, x_r2, x_r3 = population[idxs[0]], population[idxs[1]], population[idxs[2]]
                
                mutant = population[i] + self.F * (x_r1 - x_r2) + self.cauchy_scale * np.random.standard_cauchy(size=self.dim) #Cauchy dist

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
            
            # Stagnation check and local search
            if (self.generation - self.last_improvement) > self.stagnation_threshold:
                # Reduce Cauchy scale upon stagnation
                self.cauchy_scale *= 0.8
                self.cauchy_scale = max(self.cauchy_scale, 0.01) # Ensure it does not become zero

                # Local search around the best solution with probability
                if np.random.rand() < self.local_search_prob and self.budget > 0:
                    self.x_opt, self.f_opt = self.local_search(self.x_opt, func)
                    self.last_improvement = self.generation

            self.generation += 1
        
        return self.f_opt, self.x_opt