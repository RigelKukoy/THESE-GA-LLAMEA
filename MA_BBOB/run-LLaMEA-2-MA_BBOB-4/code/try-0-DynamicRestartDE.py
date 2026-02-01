import numpy as np

class DynamicRestartDE:
    def __init__(self, budget=10000, dim=10, pop_size=40, F_initial=0.5, Cr=0.9, stagnation_threshold=50, restart_probability=0.1, F_decay=0.99):
        self.budget = budget
        self.dim = dim
        self.pop_size = pop_size
        self.F = F_initial
        self.Cr = Cr
        self.stagnation_threshold = stagnation_threshold
        self.restart_probability = restart_probability
        self.F_decay = F_decay
        self.best_fitness_history = []
        self.last_improvement = 0
        self.lb = -5.0
        self.ub = 5.0

    def __call__(self, func):
        # Initialization
        population = np.random.uniform(self.lb, self.ub, size=(self.pop_size, self.dim))
        fitness = np.array([func(x) for x in population])
        self.budget -= self.pop_size
        
        self.f_opt = np.min(fitness)
        self.x_opt = population[np.argmin(fitness)]
        self.best_fitness_history.append(self.f_opt)
        
        generation = 0

        while self.budget > self.pop_size:
            # Mutation and Crossover
            new_population = np.copy(population)
            for i in range(self.pop_size):
                indices = [j for j in range(self.pop_size) if j != i]
                idxs = np.random.choice(indices, size=3, replace=False)
                x_r1, x_r2, x_r3 = population[idxs[0]], population[idxs[1]], population[idxs[2]]
                
                mutant = population[i] + self.F * (x_r1 - x_r2)
                
                # Crossover
                for j in range(self.dim):
                    if np.random.rand() < self.Cr:
                        new_population[i, j] = mutant[j]
                    
                new_population[i] = np.clip(new_population[i], self.lb, self.ub)

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
                        self.last_improvement = generation
                        
            self.best_fitness_history.append(self.f_opt)

            # Stagnation Check and Restart Mechanism
            if (generation - self.last_improvement) > self.stagnation_threshold:
                if np.random.rand() < self.restart_probability:
                    # Restart a portion of the population
                    num_to_restart = int(self.pop_size * 0.25)  # Restart 25% of the population
                    indices_to_restart = np.random.choice(self.pop_size, size=num_to_restart, replace=False)
                    population[indices_to_restart] = np.random.uniform(self.lb, self.ub, size=(num_to_restart, self.dim))
                    fitness[indices_to_restart] = np.array([func(x) for x in population[indices_to_restart]])
                    self.budget -= num_to_restart
                    
                    # Decay mutation factor
                    self.F *= self.F_decay
                    
                    # Update best solution
                    min_fitness_index = np.argmin(fitness)
                    if fitness[min_fitness_index] < self.f_opt:
                        self.f_opt = fitness[min_fitness_index]
                        self.x_opt = population[min_fitness_index]
                    
                    self.last_improvement = generation # Reset last improvement to current generation


            generation += 1
        
        return self.f_opt, self.x_opt