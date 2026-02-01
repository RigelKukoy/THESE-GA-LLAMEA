import numpy as np

class AdaptiveGaussianSearch:
    def __init__(self, budget=10000, dim=10, pop_size=20, lr=0.1, momentum=0.9):
        self.budget = budget
        self.dim = dim
        self.pop_size = pop_size
        self.lr = lr  # Learning rate for step size adaptation
        self.momentum = momentum # Momentum for step size adaptation
        self.step_size = 1.0 # Initial step size
        self.step_size_change = 0.0 # Initialize momentum for step size

    def orthogonal_initialization(self, func):
        # Latin Hypercube Sampling for better initial coverage
        population = np.zeros((self.pop_size, self.dim))
        for j in range(self.dim):
            population[:, j] = np.random.permutation(self.pop_size)
        population = (population + np.random.rand(self.pop_size, self.dim)) / self.pop_size
        population = func.bounds.lb + population * (func.bounds.ub - func.bounds.lb)
        return population
    
    def __call__(self, func):
        # Initialize population within bounds using orthogonal initialization
        population = self.orthogonal_initialization(func)
        fitness = np.array([func(x) for x in population])
        self.budget -= self.pop_size

        self.f_opt = np.min(fitness)
        self.x_opt = population[np.argmin(fitness)]
        
        generation = 0
        while self.budget > 0:
            generation += 1
            # Mutation: Gaussian perturbation with adaptive step size
            mutation = np.random.normal(0, self.step_size, size=(self.pop_size, self.dim))
            offspring = population + mutation
            
            # Clip offspring to remain within bounds
            offspring = np.clip(offspring, func.bounds.lb, func.bounds.ub)
            
            # Evaluate offspring
            offspring_fitness = np.array([func(x) for x in offspring])
            available_evals = min(self.pop_size, self.budget)
            offspring_fitness[:available_evals] = [func(x) for x in offspring[:available_evals]]
            self.budget -= available_evals
            if self.budget < 0:
                break # Stop if budget exhausted
            
            # Selection: Replace parents with better offspring
            for i in range(self.pop_size):
                if offspring_fitness[i] < fitness[i]:
                    fitness[i] = offspring_fitness[i]
                    population[i] = offspring[i]
            
            # Update best solution
            best_index = np.argmin(fitness)
            if fitness[best_index] < self.f_opt:
                self.f_opt = fitness[best_index]
                self.x_opt = population[best_index]
            
            # Adapt step size with momentum
            if generation % 10 == 0: # Adjust step size every 10 generations
                success_rate = np.sum(offspring_fitness < fitness) / self.pop_size
                
                # Update step size change with momentum
                self.step_size_change = self.momentum * self.step_size_change + self.lr * (success_rate - 0.2)
                
                # Apply the step size change
                self.step_size *= (1 + self.step_size_change)
                
                self.step_size = max(self.step_size, 1e-6) # Ensure step size doesn't become too small
        
        return self.f_opt, self.x_opt