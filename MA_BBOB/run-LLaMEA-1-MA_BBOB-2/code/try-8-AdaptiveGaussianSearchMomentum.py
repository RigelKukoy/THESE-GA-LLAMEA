import numpy as np

class AdaptiveGaussianSearchMomentum:
    def __init__(self, budget=10000, dim=10, pop_size=20, lr=0.1, momentum=0.9):
        self.budget = budget
        self.dim = dim
        self.pop_size = pop_size
        self.lr = lr  # Learning rate for step size adaptation
        self.step_size = 1.0 # Initial step size
        self.momentum = momentum
        self.step_size_change = 0.0  # Initialize momentum-related variable

    def __call__(self, func):
        # Initialize population within bounds
        population = np.random.uniform(func.bounds.lb, func.bounds.ub, size=(self.pop_size, self.dim))
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
            evals = len(offspring_fitness)
            self.budget -= evals
            if self.budget < 0:
                offspring_fitness = offspring_fitness[:evals + self.budget]

            
            # Selection: Replace parents with better offspring
            for i in range(len(offspring_fitness)):
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
                success_rate = np.sum(offspring_fitness < fitness) / len(offspring_fitness)
                step_size_update = self.lr * (success_rate - 0.2) # Center around 0.2 success
                self.step_size_change = self.momentum * self.step_size_change + (1 - self.momentum) * step_size_update
                self.step_size *= (1 + self.step_size_change)
                self.step_size = max(self.step_size, 1e-6) # Ensure step size doesn't become too small

            # Dynamic population size
            if generation % 50 == 0:
                if success_rate > 0.3:
                    self.pop_size = min(self.pop_size + 5, 50)  # Increase pop size if doing well
                elif success_rate < 0.1:
                    self.pop_size = max(self.pop_size - 5, 10)  # Decrease pop size if struggling
                
                # Regenerate population with new size
                new_population = np.random.uniform(func.bounds.lb, func.bounds.ub, size=(self.pop_size, self.dim))
                new_fitness = np.array([func(x) for x in new_population])
                self.budget -= self.pop_size
                if self.budget < 0:
                    new_fitness = new_fitness[:self.pop_size + self.budget]
                
                population[:len(new_fitness)] = new_population[:len(new_fitness)]
                fitness[:len(new_fitness)] = new_fitness[:len(new_fitness)]
                
                
        return self.f_opt, self.x_opt