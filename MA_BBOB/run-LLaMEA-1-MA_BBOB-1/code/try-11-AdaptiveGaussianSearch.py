import numpy as np

class AdaptiveGaussianSearch:
    def __init__(self, budget=10000, dim=10, pop_size=20, initial_std=1.0, shrink_factor=0.99, momentum=0.1, success_rate_alpha=0.1):
        self.budget = budget
        self.dim = dim
        self.pop_size = pop_size
        self.initial_std = initial_std
        self.shrink_factor = shrink_factor
        self.lb = -5.0
        self.ub = 5.0
        self.momentum = momentum
        self.success_rate_alpha = success_rate_alpha
        self.std = initial_std  # Current std, allows adaptation
        self.velocity = np.zeros((pop_size, dim)) #Initialize velocity
        self.success_rate = 0.5  # Initialize success rate for step size adaptation

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
            # Generate offspring using Gaussian mutation with momentum
            # Update velocity based on previous velocity and a Gaussian random variable
            self.velocity = self.momentum * self.velocity + np.sqrt(1 - self.momentum**2) * np.random.normal(0, self.std, size=(self.pop_size, self.dim))
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
            
            # Select survivors (replace the worst individuals with the best offspring)
            num_improved = 0
            for i in range(self.pop_size):
                if offspring_fitness[i] < fitness[i]:
                    population[i] = offspring[i].copy()
                    fitness[i] = offspring_fitness[i]
                    num_improved += 1

            # Update success rate
            self.success_rate = (1 - self.success_rate_alpha) * self.success_rate + self.success_rate_alpha * (num_improved / self.pop_size)

            # Adjust step size based on success rate
            if self.success_rate > 0.2:
                self.std *= 1.1  # Increase step size
            elif self.success_rate < 0.1:
                self.std *= 0.9  # Decrease step size

            self.std = min(self.std, self.initial_std) #Cap to avoid excessive std
            
            # Shrink the search space (reduce standard deviation)
            #self.std *= self.shrink_factor #Moved to adaptation based on success

            # Adjust bounds (shrink towards the best solution)
            self.lb = np.maximum(self.lb, self.x_opt - 2.5 * self.std)
            self.ub = np.minimum(self.ub, self.x_opt + 2.5 * self.std)
            
            population = np.clip(population, self.lb, self.ub)
            
            if self.eval_count >= self.budget:
                break
                
        return self.f_opt, self.x_opt