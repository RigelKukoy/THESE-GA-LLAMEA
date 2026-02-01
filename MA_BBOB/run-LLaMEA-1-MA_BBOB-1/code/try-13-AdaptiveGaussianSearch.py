import numpy as np

class AdaptiveGaussianSearch:
    def __init__(self, budget=10000, dim=10, pop_size=20, initial_std=1.0, shrink_factor=0.99, momentum=0.1):
        self.budget = budget
        self.dim = dim
        self.pop_size = pop_size
        self.initial_std = initial_std
        self.shrink_factor = shrink_factor
        self.lb = -5.0
        self.ub = 5.0
        self.momentum = momentum
        self.velocity = np.zeros((pop_size, dim))

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
        learning_rate = 1.0
        
        while self.eval_count < self.budget:
            # Generate offspring using Gaussian mutation and momentum
            mutation = np.random.normal(0, std, size=(self.pop_size, self.dim))
            self.velocity = self.momentum * self.velocity + learning_rate * mutation
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
            
            # Selection: Tournament selection
            selected_indices = np.random.choice(self.pop_size, size=self.pop_size, replace=True)
            for i in range(self.pop_size):
                if offspring_fitness[i] < fitness[selected_indices[i]]:
                    population[selected_indices[i]] = offspring[i]
                    fitness[selected_indices[i]] = offspring_fitness[i]
            
            # Shrink the search space (reduce standard deviation)
            std *= self.shrink_factor
            learning_rate *= self.shrink_factor
            
            # Adjust bounds (shrink towards the best solution)
            self.lb = np.maximum(self.lb, self.x_opt - 2.5 * std)
            self.ub = np.minimum(self.ub, self.x_opt + 2.5 * std)
            
            population = np.clip(population, self.lb, self.ub)
            
            if self.eval_count >= self.budget:
                break
                
        return self.f_opt, self.x_opt