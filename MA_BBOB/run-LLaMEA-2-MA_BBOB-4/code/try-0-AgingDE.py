import numpy as np

class AgingDE:
    def __init__(self, budget=10000, dim=10, pop_size=40, aging_rate=0.01):
        self.budget = budget
        self.dim = dim
        self.pop_size = pop_size
        self.aging_rate = aging_rate
        self.F = 0.5
        self.Cr = 0.9

    def __call__(self, func):
        # Initialization
        population = np.random.uniform(func.bounds.lb, func.bounds.ub, size=(self.pop_size, self.dim))
        fitness = np.array([func(x) for x in population])
        self.budget -= self.pop_size

        self.f_opt = np.min(fitness)
        self.x_opt = population[np.argmin(fitness)]

        age = np.zeros(self.pop_size)  # Initialize age for each individual

        while self.budget > self.pop_size:
            # Mutation and Crossover
            new_population = np.copy(population)
            new_fitness = np.zeros(self.pop_size)

            for i in range(self.pop_size):
                # Mutation: Use best individual info + random individual
                best_idx = np.argmin(fitness)
                random_idx = np.random.randint(self.pop_size)
                while random_idx == i:
                    random_idx = np.random.randint(self.pop_size)
                
                indices = [j for j in range(self.pop_size) if j != i and j != best_idx and j!= random_idx]
                if len(indices) < 1: 
                    mutant = np.random.uniform(func.bounds.lb, func.bounds.ub, size=self.dim)
                else:
                    idx_r1 = np.random.choice(indices, size=1, replace=False)[0]
                    x_r1 = population[idx_r1]

                    mutant = population[i] + self.F * (population[best_idx] - population[i]) + self.F * (x_r1 - population[random_idx]) # Novel Mutation

                mutant = np.clip(mutant, func.bounds.lb, func.bounds.ub)


                # Crossover
                for j in range(self.dim):
                    if np.random.rand() < self.Cr:
                        new_population[i, j] = mutant[j]

                new_population[i] = np.clip(new_population[i], func.bounds.lb, func.bounds.ub)
                new_fitness[i] = func(new_population[i])
                self.budget -= 1

            # Selection and Aging
            for i in range(self.pop_size):
                if new_fitness[i] < fitness[i]:
                    population[i] = new_population[i]
                    fitness[i] = new_fitness[i]
                    age[i] = 0  # Reset age
                    if new_fitness[i] < self.f_opt:
                        self.f_opt = new_fitness[i]
                        self.x_opt = new_population[i]
                else:
                    age[i] += 1 # Increment age


                # Aging mechanism: replace old individuals
                if age[i] > (self.budget/self.pop_size * self.aging_rate): #Age is relative to budget.
                    population[i] = np.random.uniform(func.bounds.lb, func.bounds.ub, size=self.dim)
                    fitness[i] = func(population[i])
                    self.budget -= 1 # Account for new function evaluation
                    age[i] = 0 # Reset age of the new individual
                    if fitness[i] < self.f_opt:
                        self.f_opt = fitness[i]
                        self.x_opt = population[i]

        return self.f_opt, self.x_opt