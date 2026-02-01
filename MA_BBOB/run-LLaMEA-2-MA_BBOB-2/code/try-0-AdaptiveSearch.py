import numpy as np

class AdaptiveSearch:
    def __init__(self, budget=10000, dim=10, pop_size=20, local_search_iterations=5):
        self.budget = budget
        self.dim = dim
        self.pop_size = pop_size
        self.local_search_iterations = local_search_iterations
        self.exploration_rate = 0.7  # Initial exploration rate
        self.exploration_decay = 0.995 # Decay rate for exploration

    def __call__(self, func):
        self.f_opt = np.Inf
        self.x_opt = None
        self.population = np.random.uniform(func.bounds.lb, func.bounds.ub, size=(self.pop_size, self.dim))
        self.fitness = np.array([func(x) for x in self.population])
        self.budget -= self.pop_size

        while self.budget > 0:
            # Exploration phase: Generate new solutions using a global search strategy
            if np.random.rand() < self.exploration_rate:
                new_population = np.random.uniform(func.bounds.lb, func.bounds.ub, size=(self.pop_size, self.dim))
                new_fitness = np.array([func(x) for x in new_population])
                self.budget -= self.pop_size
            else:  # Exploitation phase: Refine existing solutions using local search
                new_population = np.copy(self.population)
                new_fitness = np.copy(self.fitness)

                for i in range(self.pop_size):
                    for _ in range(self.local_search_iterations):
                        if self.budget <=0:
                            break

                        # Create a small perturbation around the current solution
                        perturbation = np.random.normal(0, 0.1, size=self.dim)
                        new_x = self.population[i] + perturbation
                        new_x = np.clip(new_x, func.bounds.lb, func.bounds.ub)  # Keep within bounds
                        
                        new_f = func(new_x)
                        self.budget -= 1
                        if new_f < new_fitness[i]:
                            new_population[i] = new_x
                            new_fitness[i] = new_f
            

            # Update the population by selecting the best solutions from the old and new populations
            combined_population = np.concatenate((self.population, new_population))
            combined_fitness = np.concatenate((self.fitness, new_fitness))
            
            indices = np.argsort(combined_fitness)[:self.pop_size]
            self.population = combined_population[indices]
            self.fitness = combined_fitness[indices]

            # Update the best solution found so far
            best_index = np.argmin(self.fitness)
            if self.fitness[best_index] < self.f_opt:
                self.f_opt = self.fitness[best_index]
                self.x_opt = self.population[best_index]
            
            self.exploration_rate *= self.exploration_decay # Reduce exploration over time

        return self.f_opt, self.x_opt