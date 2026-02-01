import numpy as np

class RankBasedAdaptiveDE:
    def __init__(self, budget=10000, dim=10, pop_size=40, Cr=0.9, aging_rate=0.02):
        self.budget = budget
        self.dim = dim
        self.pop_size = pop_size
        self.Cr = Cr
        self.aging_rate = aging_rate
        self.best_fitness_history = []
        self.population = None
        self.fitness = None
        self.ages = None

    def __call__(self, func):
        # Initialization
        self.population = np.random.uniform(func.bounds.lb, func.bounds.ub, size=(self.pop_size, self.dim))
        self.fitness = np.array([func(x) for x in self.population])
        self.budget -= self.pop_size
        
        self.f_opt = np.min(self.fitness)
        self.x_opt = self.population[np.argmin(self.fitness)]
        self.best_fitness_history.append(self.f_opt)
        self.ages = np.zeros(self.pop_size)
        
        generation = 0

        while self.budget > self.pop_size:
            # Sort population based on fitness
            ranked_indices = np.argsort(self.fitness)
            ranked_population = self.population[ranked_indices]

            new_population = np.copy(self.population)
            for i in range(self.pop_size):
                # Rank-based mutation factor
                rank = np.where(ranked_indices == i)[0][0]  # Find the rank of individual i
                F = 0.1 + 0.9 * (rank / (self.pop_size - 1))  # F increases with rank

                indices = [j for j in range(self.pop_size) if j != i]
                idxs = np.random.choice(indices, size=3, replace=False)
                x_r1, x_r2, x_r3 = self.population[idxs[0]], self.population[idxs[1]], self.population[idxs[2]]
                
                mutant = self.population[i] + F * (x_r1 - x_r2)
                
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
                if new_fitness[i] < self.fitness[i]:
                    self.population[i] = new_population[i]
                    self.fitness[i] = new_fitness[i]
                    self.ages[i] = 0  # Reset age if improved
                    
                    if new_fitness[i] < self.f_opt:
                        self.f_opt = new_fitness[i]
                        self.x_opt = new_population[i]
                else:
                    self.ages[i] += 1 # Increment age if not improved

            self.best_fitness_history.append(self.f_opt)

            # Aging mechanism: Replace old individuals with new random ones
            for i in range(self.pop_size):
                if self.ages[i] > (1 / self.aging_rate):  # Age threshold
                    self.population[i] = np.random.uniform(func.bounds.lb, func.bounds.ub, size=self.dim)
                    self.fitness[i] = func(self.population[i])
                    self.budget -= 1
                    self.ages[i] = 0
                    if self.fitness[i] < self.f_opt:
                        self.f_opt = self.fitness[i]
                        self.x_opt = self.population[i]

            generation += 1

        return self.f_opt, self.x_opt