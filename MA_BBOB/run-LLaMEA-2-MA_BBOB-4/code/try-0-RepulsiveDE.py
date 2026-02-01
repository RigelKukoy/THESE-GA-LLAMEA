import numpy as np

class RepulsiveDE:
    def __init__(self, budget=10000, dim=10, pop_size=40, F_init=0.5, Cr_init=0.7, repulsion_factor=0.1, adaptation_rate=0.1):
        self.budget = budget
        self.dim = dim
        self.pop_size = pop_size
        self.F = F_init
        self.Cr = Cr_init
        self.repulsion_factor = repulsion_factor
        self.adaptation_rate = adaptation_rate
        self.best_fitness_history = []
        self.stagnation_counter = 0
        self.last_improvement = 0

    def __call__(self, func):
        # Initialization
        population = np.random.uniform(func.bounds.lb, func.bounds.ub, size=(self.pop_size, self.dim))
        fitness = np.array([func(x) for x in population])
        self.budget -= self.pop_size
        
        self.f_opt = np.min(fitness)
        self.x_opt = population[np.argmin(fitness)].copy()
        self.best_fitness_history.append(self.f_opt)
        
        generation = 0

        while self.budget > self.pop_size:
            # Adaptive Parameter Adjustment
            if generation > 0:
                if self.f_opt == self.best_fitness_history[-1]:
                    self.stagnation_counter += 1
                else:
                    self.stagnation_counter = 0
            
            # Dynamically adjust F and Cr
            if self.stagnation_counter > 50:
                self.F = min(1.0, self.F * (1 + self.adaptation_rate))  # Increase F for more exploration
                self.Cr = max(0.0, self.Cr * (1 - self.adaptation_rate))  # Decrease Cr for focused search
            else:
                self.F = max(0.1, self.F * (1 - self.adaptation_rate/2))  # Decrease F for exploitation
                self.Cr = min(0.9, self.Cr * (1 + self.adaptation_rate/2))  # Increase Cr for exploration

            # Mutation and Crossover
            new_population = np.copy(population)
            for i in range(self.pop_size):
                # Mutation
                indices = [j for j in range(self.pop_size) if j != i]
                idxs = np.random.choice(indices, size=3, replace=False)
                x_r1, x_r2, x_r3 = population[idxs[0]], population[idxs[1]], population[idxs[2]]

                # Repulsion from the worst individual
                worst_index = np.argmax(fitness)
                x_worst = population[worst_index]

                mutant = population[i] + self.F * (x_r1 - x_r2) + self.repulsion_factor * (population[i] - x_worst)
                
                # Crossover
                for j in range(self.dim):
                    if np.random.rand() < self.Cr:
                        new_population[i, j] = mutant[j]
                    else:
                        new_population[i, j] = population[i, j]
                
                new_population[i] = np.clip(new_population[i], func.bounds.lb, func.bounds.ub)
            
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
                        self.x_opt = new_population[i].copy()  # Deep copy
                        self.last_improvement = generation
                        
            self.best_fitness_history.append(self.f_opt)
                
            generation += 1
        
        return self.f_opt, self.x_opt