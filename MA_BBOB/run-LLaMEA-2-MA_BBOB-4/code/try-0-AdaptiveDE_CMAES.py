import numpy as np

class AdaptiveDE_CMAES:
    def __init__(self, budget=10000, dim=10, pop_size=40, stagnation_threshold=50, cmaes_probability=0.1):
        self.budget = budget
        self.dim = dim
        self.pop_size = pop_size
        self.stagnation_threshold = stagnation_threshold
        self.cmaes_probability = cmaes_probability
        self.best_fitness_history = []
        self.last_improvement = 0
        self.F = 0.5
        self.Cr = 0.9

    def __call__(self, func):
        # Initialization
        population = np.random.uniform(func.bounds.lb, func.bounds.ub, size=(self.pop_size, self.dim))
        fitness = np.array([func(x) for x in population])
        self.budget -= self.pop_size
        
        self.f_opt = np.min(fitness)
        self.x_opt = population[np.argmin(fitness)]
        self.best_fitness_history.append(self.f_opt)
        
        generation = 0

        while self.budget > self.pop_size:
            # Adaptive F and Cr
            if np.random.rand() < 0.1:
                self.F = np.random.uniform(0.1, 0.9)
            if np.random.rand() < 0.1:
                self.Cr = np.random.uniform(0.1, 0.9)
            
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
                        self.x_opt = new_population[i]
                        self.last_improvement = generation
                        
            self.best_fitness_history.append(self.f_opt)

            # CMA-ES Local Search
            if (generation - self.last_improvement) > self.stagnation_threshold:
                for i in range(self.pop_size):
                    if np.random.rand() < self.cmaes_probability:
                        # Perform CMA-ES local search around individual i
                        x_current = population[i].copy()
                        sigma = 0.1  # Initial step size
                        C = np.eye(self.dim)  # Initial covariance matrix

                        for _ in range(5):  # Limited budget for local search
                            z = np.random.randn(self.dim)
                            x_new = x_current + sigma * np.dot(C, z)
                            x_new = np.clip(x_new, func.bounds.lb, func.bounds.ub)
                            f_new = func(x_new)
                            self.budget -= 1
                            if self.budget <= 0:
                                return self.f_opt, self.x_opt

                            if f_new < fitness[i]:
                                population[i] = x_new
                                fitness[i] = f_new
                                x_current = x_new
                                if f_new < self.f_opt:
                                    self.f_opt = f_new
                                    self.x_opt = x_new
                                    self.last_improvement = generation

            generation += 1

        return self.f_opt, self.x_opt