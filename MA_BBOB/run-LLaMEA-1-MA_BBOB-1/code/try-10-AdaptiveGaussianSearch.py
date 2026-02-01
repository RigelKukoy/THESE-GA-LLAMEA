import numpy as np

class AdaptiveGaussianSearch:
    def __init__(self, budget=10000, dim=10, pop_size=20, initial_std=1.0, shrink_factor=0.99, momentum=0.1):
        self.budget = budget
        self.dim = dim
        self.pop_size = pop_size
        self.initial_std = initial_std
        self.shrink_factor = shrink_factor
        self.momentum = momentum
        self.lb = -5.0
        self.ub = 5.0
        self.velocity = np.zeros((pop_size, dim))  # Initialize velocity

    def __call__(self, func):
        self.f_opt = np.Inf
        self.x_opt = None
        self.eval_count = 0
        
        # Initialize population
        population = np.random.uniform(self.lb, self.ub, size=(self.pop_size, self.dim))
        fitness = np.array([func(x) for x in population])
        self.eval_count += self.pop_size
        
        # Find best initial solution
        best_index = np.argmin(fitness)
        if fitness[best_index] < self.f_opt:
            self.f_opt = fitness[best_index]
            self.x_opt = population[best_index]
            
        std = self.initial_std
        
        while self.eval_count < self.budget:
            # Generate offspring using Gaussian mutation with momentum
            noise = np.random.normal(0, std, size=(self.pop_size, self.dim))
            self.velocity = self.momentum * self.velocity + (1 - self.momentum) * noise
            offspring = population + self.velocity
            
            # Clip offspring to stay within bounds
            offspring = np.clip(offspring, self.lb, self.ub)
            
            # Evaluate offspring
            offspring_fitness = np.array([func(x) for x in offspring])
            self.eval_count += self.pop_size
            
            # Update best solution
            best_offspring_index = np.argmin(offspring_fitness)
            if offspring_fitness[best_offspring_index] < self.f_opt:
                self.f_opt = offspring_fitness[best_offspring_index]
                self.x_opt = offspring[best_offspring_index]
                
                # Adaptive shrink factor adjustment: Reduce shrink factor if improvement is good
                if self.f_opt < np.min(fitness):
                    self.shrink_factor = min(0.999, self.shrink_factor + 0.001) # increase shrink factor
                else:
                    self.shrink_factor = max(0.9, self.shrink_factor - 0.001) # decrease shrink factor
            
            # Select survivors (replace the worst individuals with the best offspring)
            worst_index = np.argmax(fitness)
            best_offspring_index = np.argmin(offspring_fitness)
            if offspring_fitness[best_offspring_index] < fitness[worst_index]:
                population[worst_index] = offspring[best_offspring_index]
                fitness[worst_index] = offspring_fitness[best_offspring_index]
            
            # Shrink the search space (reduce standard deviation)
            std *= self.shrink_factor
            
            # Adjust bounds (shrink towards the best solution)
            self.lb = np.maximum(self.lb, self.x_opt - 2.5 * std)
            self.ub = np.minimum(self.ub, self.x_opt + 2.5 * std)
            
            population = np.clip(population, self.lb, self.ub)
            
            if self.eval_count >= self.budget:
                break
                
        return self.f_opt, self.x_opt