import numpy as np

class EnsembleAdaptiveDE:
    def __init__(self, budget=10000, dim=10, pop_size=40, memory_size=10, strategy_probabilities=[0.3, 0.3, 0.4]):
        self.budget = budget
        self.dim = dim
        self.pop_size = pop_size
        self.memory_size = memory_size
        self.strategy_probabilities = strategy_probabilities
        self.F_memory = np.ones(self.memory_size) * 0.5
        self.Cr_memory = np.ones(self.memory_size) * 0.5
        self.archive_rate = 0.1
        self.archive = None
        self.strategy_successes = np.zeros(len(strategy_probabilities))
        self.population = None
        self.fitness = None
        self.f_opt = np.Inf
        self.x_opt = None

    def __call__(self, func):
        # Initialization
        self.population = np.random.uniform(func.bounds.lb, func.bounds.ub, size=(self.pop_size, self.dim))
        self.fitness = np.array([func(x) for x in self.population])
        self.budget -= self.pop_size
        self.f_opt = np.min(self.fitness)
        self.x_opt = self.population[np.argmin(self.fitness)]
        self.archive = np.copy(self.population[:int(self.pop_size * self.archive_rate)])

        memory_index = 0

        while self.budget > self.pop_size:
            new_population = np.copy(self.population)
            new_fitness = np.zeros(self.pop_size)

            # Sample F and Cr from memory
            F = np.random.choice(self.F_memory)
            Cr = np.random.choice(self.Cr_memory)

            for i in range(self.pop_size):
                # Choose a mutation strategy based on probabilities
                strategy_choice = np.random.choice(len(self.strategy_probabilities), p=self.strategy_probabilities)

                if strategy_choice == 0:  # DE/rand/1
                    indices = np.random.choice(self.pop_size, size=3, replace=False)
                    x_r1, x_r2, x_r3 = self.population[indices[0]], self.population[indices[1]], self.population[indices[2]]
                    mutant = x_r1 + F * (x_r2 - x_r3)
                elif strategy_choice == 1:  # DE/current-to-best/1
                    indices = np.random.choice(self.pop_size, size=2, replace=False)
                    x_r1, x_r2 = self.population[indices[0]], self.population[indices[1]]
                    mutant = self.population[i] + F * (self.x_opt - self.population[i]) + F * (x_r1 - x_r2)
                else:  # DE/rand/2
                    indices = np.random.choice(self.pop_size, size=5, replace=False)
                    x_r1, x_r2, x_r3, x_r4, x_r5 = self.population[indices[0]], self.population[indices[1]], self.population[indices[2]], self.population[indices[4]], self.population[indices[4]]
                    mutant = x_r1 + F * (x_r2 - x_r3) + F * (x_r4-x_r5)

                # Crossover
                for j in range(self.dim):
                    if np.random.rand() < Cr:
                        new_population[i, j] = mutant[j]
                    else:
                        new_population[i, j] = self.population[i, j]
                
                new_population[i] = np.clip(new_population[i], func.bounds.lb, func.bounds.ub)
                new_fitness[i] = func(new_population[i])
                self.budget -= 1

            # Selection
            for i in range(self.pop_size):
                if new_fitness[i] < self.fitness[i]:
                    self.population[i] = new_population[i]
                    self.fitness[i] = new_fitness[i]
                    if new_fitness[i] < self.f_opt:
                        self.f_opt = new_fitness[i]
                        self.x_opt = new_population[i]
                
            # Update memory (simplified - replace oldest)
            self.F_memory[memory_index] = F
            self.Cr_memory[memory_index] = Cr
            memory_index = (memory_index + 1) % self.memory_size
        return self.f_opt, self.x_opt