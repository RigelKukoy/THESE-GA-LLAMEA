import numpy as np

class CauchyAdaptiveDE:
    def __init__(self, budget=10000, dim=10, initial_pop_size=50, F=0.5, Cr=0.7, cauchy_scale=0.1, adaptive_pop_factor=0.1, restart_trigger=100):
        self.budget = budget
        self.dim = dim
        self.pop_size = initial_pop_size
        self.F = F
        self.Cr = Cr
        self.cauchy_scale = cauchy_scale
        self.adaptive_pop_factor = adaptive_pop_factor
        self.restart_trigger = restart_trigger
        self.best_fitness_history = []
        self.no_improvement_counter = 0

    def __call__(self, func):
        # Initialization
        population = np.random.uniform(func.bounds.lb, func.bounds.ub, size=(self.pop_size, self.dim))
        fitness = np.array([func(x) for x in population])
        self.budget -= self.pop_size
        
        self.f_opt = np.min(fitness)
        self.x_opt = population[np.argmin(fitness)]
        self.best_fitness_history.append(self.f_opt)

        while self.budget > self.pop_size:
            new_population = np.copy(population)
            for i in range(self.pop_size):
                # Mutation with Cauchy distribution
                idxs = np.random.choice(self.pop_size, 3, replace=False)
                x_r1, x_r2, x_r3 = population[idxs]
                cauchy_noise = self.cauchy_scale * np.random.standard_cauchy(size=self.dim)
                mutant = population[i] + self.F * (x_r2 - x_r3) + cauchy_noise
                mutant = np.clip(mutant, func.bounds.lb, func.bounds.ub)

                # Crossover
                for j in range(self.dim):
                    if np.random.rand() < self.Cr:
                        new_population[i, j] = mutant[j]
                    else:
                        new_population[i, j] = population[i, j]

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
                        self.no_improvement_counter = 0  # Reset counter if improvement found
                else:
                    self.no_improvement_counter +=1

            self.best_fitness_history.append(self.f_opt)
            
            # Adaptive Population Sizing
            if self.no_improvement_counter > self.restart_trigger // 2:
                new_pop_size = int(self.pop_size * (1 + self.adaptive_pop_factor))
                new_pop_size = min(new_pop_size, self.budget) # Limit pop size by remaining budget
                if new_pop_size > self.pop_size:
                    additional_individuals = np.random.uniform(func.bounds.lb, func.bounds.ub, size=(new_pop_size - self.pop_size, self.dim))
                    population = np.vstack((population, additional_individuals))
                    fitness = np.concatenate((fitness, np.array([func(x) for x in additional_individuals])))
                    self.budget -= (new_pop_size - self.pop_size)
                    self.pop_size = new_pop_size

            # Restart Mechanism
            if self.no_improvement_counter > self.restart_trigger:
                population = np.random.uniform(func.bounds.lb, func.bounds.ub, size=(self.pop_size, self.dim))
                fitness = np.array([func(x) for x in population])
                self.budget -= self.pop_size
                self.f_opt = np.min(fitness)
                self.x_opt = population[np.argmin(fitness)]
                self.no_improvement_counter = 0  # Reset counter after restart
                self.best_fitness_history.append(self.f_opt)
                

        return self.f_opt, self.x_opt