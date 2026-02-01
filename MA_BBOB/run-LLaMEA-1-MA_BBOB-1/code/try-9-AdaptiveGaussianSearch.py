import numpy as np

class AdaptiveGaussianSearch:
    def __init__(self, budget=10000, dim=10, pop_size=20, initial_std=1.0, shrink_factor=0.99, success_history_length=10):
        self.budget = budget
        self.dim = dim
        self.pop_size = pop_size
        self.initial_std = initial_std
        self.shrink_factor = shrink_factor
        self.lb = -5.0
        self.ub = 5.0
        self.success_history_length = success_history_length
        self.success_history = []
        self.momentum = 0.1  # Momentum for std update
        self.std = self.initial_std
        self.v = 0 # velocity of std

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
        
        
        while self.eval_count < self.budget:
            # Generate offspring using Gaussian mutation
            offspring = population + np.random.normal(0, self.std, size=(self.pop_size, self.dim))
            
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
                success = True
            else:
                success = False
            
            # Select survivors (replace the worst individuals with the best offspring)
            worst_index = np.argmax(fitness)
            best_offspring_index = np.argmin(offspring_fitness)
            if offspring_fitness[best_offspring_index] < fitness[worst_index]:
                population[worst_index] = offspring[best_offspring_index]
                fitness[worst_index] = offspring_fitness[best_offspring_index]
            
            # Update success history
            self.success_history.append(int(success))
            if len(self.success_history) > self.success_history_length:
                self.success_history.pop(0)
            
            # Adjust step size based on success rate
            if len(self.success_history) == self.success_history_length:
                success_rate = np.mean(self.success_history)
                if success_rate > 0.4:
                    self.v = self.momentum * self.v + (1 - self.momentum) * 0.1
                elif success_rate < 0.2:
                    self.v = self.momentum * self.v - (1 - self.momentum) * 0.1
                
                self.std *= np.exp(self.v)
                self.std = max(self.std, 1e-6) # Minimum std

            # Adjust bounds (shrink towards the best solution)
            self.lb = np.maximum(self.lb, self.x_opt - 2.5 * self.std)
            self.ub = np.minimum(self.ub, self.x_opt + 2.5 * self.std)
            
            population = np.clip(population, self.lb, self.ub)
            
            if self.eval_count >= self.budget:
                break
                
        return self.f_opt, self.x_opt