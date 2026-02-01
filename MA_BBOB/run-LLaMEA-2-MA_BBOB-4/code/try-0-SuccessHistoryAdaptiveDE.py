import numpy as np

class SuccessHistoryAdaptiveDE:
    def __init__(self, budget=10000, dim=10, pop_size=40, memory_size=10, initial_F=0.5, initial_Cr=0.5):
        self.budget = budget
        self.dim = dim
        self.pop_size = pop_size
        self.memory_size = memory_size
        self.F_memory = np.full(memory_size, initial_F)
        self.Cr_memory = np.full(memory_size, initial_Cr)
        self.memory_index = 0
        self.best_fitness_history = []

    def __call__(self, func):
        # Initialization
        population = np.random.uniform(func.bounds.lb, func.bounds.ub, size=(self.pop_size, self.dim))
        fitness = np.array([func(x) for x in population])
        self.budget -= self.pop_size
        
        self.f_opt = np.min(fitness)
        self.x_opt = population[np.argmin(fitness)]
        self.best_fitness_history.append(self.f_opt)

        success_F_list = []
        success_Cr_list = []
        
        while self.budget > self.pop_size:
            new_population = np.copy(population)
            new_fitness = np.zeros(self.pop_size)
            
            for i in range(self.pop_size):
                # Adaptation of F and Cr
                F = self.F_memory[np.random.randint(self.memory_size)]
                Cr = self.Cr_memory[np.random.randint(self.memory_size)]
                
                # Mutation
                idxs = np.random.choice(self.pop_size, 3, replace=False)
                x_r1, x_r2, x_r3 = population[idxs]
                
                # Cauchy mutation with probability 0.1, otherwise standard DE mutation
                if np.random.rand() < 0.1:
                   mutant = population[i] + F * (x_r1 - population[i]) + 0.1 * np.random.standard_cauchy(size=self.dim)
                else:
                   mutant = population[i] + F * (x_r2 - x_r3)
                
                mutant = np.clip(mutant, func.bounds.lb, func.bounds.ub)
                
                # Crossover
                for j in range(self.dim):
                    if np.random.rand() < Cr:
                        new_population[i, j] = mutant[j]
                    else:
                        new_population[i, j] = population[i, j]
                        
            # Evaluation
            new_fitness = np.array([func(x) for x in new_population])
            self.budget -= self.pop_size

            # Selection and update success history
            for i in range(self.pop_size):
                if new_fitness[i] < fitness[i]:
                    success_F_list.append(F)
                    success_Cr_list.append(Cr)
                    population[i] = new_population[i]
                    fitness[i] = new_fitness[i]
                    if new_fitness[i] < self.f_opt:
                        self.f_opt = new_fitness[i]
                        self.x_opt = new_population[i]
            self.best_fitness_history.append(self.f_opt)
            
            # Update memory
            if success_F_list:
                self.F_memory[self.memory_index] = np.mean(success_F_list)
                self.Cr_memory[self.memory_index] = np.mean(success_Cr_list)
                self.memory_index = (self.memory_index + 1) % self.memory_size

            success_F_list = []
            success_Cr_list = []
            
        return self.f_opt, self.x_opt