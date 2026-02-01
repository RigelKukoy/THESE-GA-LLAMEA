import numpy as np

class AdaptiveGaussianSearch:
    def __init__(self, budget=10000, dim=10, initial_pop_size=20, initial_std=1.0, shrink_factor=0.99, success_history_length=10, pop_size_adapt_freq=50):
        self.budget = budget
        self.dim = dim
        self.initial_pop_size = initial_pop_size
        self.pop_size = initial_pop_size
        self.initial_std = initial_std
        self.shrink_factor = shrink_factor
        self.lb = -5.0
        self.ub = 5.0
        self.success_history_length = success_history_length
        self.success_history = []
        self.momentum = 0.1  # Momentum for std update
        self.std = self.initial_std
        self.v = 0 # velocity of std
        self.eval_count = 0
        self.pop_size_adapt_freq = pop_size_adapt_freq
        self.learning_rate = 0.1


    def __call__(self, func):
        self.f_opt = np.Inf
        self.x_opt = None
        
        # Initialize population
        population = np.random.uniform(self.lb, self.ub, size=(self.pop_size, self.dim))
        fitness = np.array([func(x) for x in population])
        self.eval_count += self.pop_size
        
        # Find best initial solution
        best_index = np.argmin(fitness)
        if fitness[best_index] < self.f_opt:
            self.f_opt = fitness[best_index]
            self.x_opt = population[best_index]
        
        
        generation = 0
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
                
                # Adaptive Learning Rate
                if success_rate > 0.4:
                    lr = self.learning_rate * 1.1  # Increase learning rate if doing well
                elif success_rate < 0.2:
                    lr = self.learning_rate * 0.9  # Decrease learning rate if not doing well
                else:
                    lr = self.learning_rate
                
                lr = np.clip(lr, 0.01, 0.2)  # Keep learning rate within bounds
                self.learning_rate = lr
                
                self.v = self.momentum * self.v + (1 - self.momentum) * (success_rate - 0.3) * lr  # Adjusted update
                
                self.std *= np.exp(self.v)
                self.std = max(self.std, 1e-6) # Minimum std

            # Adjust bounds (shrink towards the best solution)
            self.lb = np.maximum(self.lb, self.x_opt - 2.5 * self.std)
            self.ub = np.minimum(self.ub, self.x_opt + 2.5 * self.std)
            
            population = np.clip(population, self.lb, self.ub)

            generation += 1
            if generation % self.pop_size_adapt_freq == 0:
                # Dynamically adjust population size
                if success_rate > 0.4 and self.pop_size < 2 * self.initial_pop_size:
                    self.pop_size = min(self.pop_size + 5, 2 * self.initial_pop_size)  # Increase if successful
                    population = np.vstack((population, np.random.uniform(self.lb, self.ub, size=(5, self.dim))))
                    new_fitness = np.array([func(x) for x in population[-5:]])
                    fitness = np.concatenate((fitness, new_fitness))
                    self.eval_count += 5

                elif success_rate < 0.2 and self.pop_size > self.initial_pop_size // 2:
                    self.pop_size = max(self.pop_size - 5, self.initial_pop_size // 2)  # Decrease if unsuccessful
                    population = population[:self.pop_size]
                    fitness = fitness[:self.pop_size]
                
            if self.eval_count >= self.budget:
                break
                
        return self.f_opt, self.x_opt