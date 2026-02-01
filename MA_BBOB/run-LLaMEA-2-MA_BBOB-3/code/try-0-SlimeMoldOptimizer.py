import numpy as np

class SlimeMoldOptimizer:
    def __init__(self, budget=10000, dim=10, population_size=20, initial_step_size=0.1):
        self.budget = budget
        self.dim = dim
        self.population_size = population_size
        self.initial_step_size = initial_step_size

    def __call__(self, func):
        lb = func.bounds.lb
        ub = func.bounds.ub
        
        # Initialize population
        population = np.random.uniform(lb, ub, size=(self.population_size, self.dim))
        fitness = np.array([func(x) for x in population])
        self.budget -= self.population_size # Subtract initial population evals from budget

        best_index = np.argmin(fitness)
        self.f_opt = fitness[best_index]
        self.x_opt = population[best_index]
        
        step_size = np.full(self.population_size, self.initial_step_size)
        
        while self.budget > 0:
            # Update weights based on fitness (simulating slime mold behavior)
            weights = np.exp(-np.abs(fitness - self.f_opt) / (np.mean(np.abs(fitness - self.f_opt)) + 1e-8))
            weights /= np.sum(weights)

            # Update positions
            for i in range(self.population_size):
                if self.budget <=0:
                  break
                # Select a random individual from the population, weighted by fitness
                j = np.random.choice(self.population_size, p=weights)
                
                # Slime mold update rule with adaptive step size
                direction = population[j] - population[i]
                new_position = population[i] + step_size[i] * direction * np.random.uniform(-1, 1, self.dim)
                
                # Clip to bounds
                new_position = np.clip(new_position, lb, ub)
                
                new_fitness = func(new_position)
                self.budget -= 1
                
                # Update if better
                if new_fitness < fitness[i]:
                    fitness[i] = new_fitness
                    population[i] = new_position
                    
                    # Adjust step size (increase if improvement, decrease if not)
                    step_size[i] *= 1.2  # Increase step size
                    if step_size[i] > 1.0:
                        step_size[i] = 1.0 #Limit step size
                else:
                    step_size[i] *= 0.8 #Reduce Step size

                # Update global best
                if fitness[i] < self.f_opt:
                    self.f_opt = fitness[i]
                    self.x_opt = population[i]

        return self.f_opt, self.x_opt