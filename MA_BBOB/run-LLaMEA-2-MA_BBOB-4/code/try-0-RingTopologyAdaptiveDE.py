import numpy as np

class RingTopologyAdaptiveDE:
    def __init__(self, budget=10000, dim=10, pop_size=40, initial_Cr=0.5, initial_F=0.7, learning_rate=0.1, ring_neighborhood=3):
        self.budget = budget
        self.dim = dim
        self.pop_size = pop_size
        self.Cr = np.full(pop_size, initial_Cr)  # Individual crossover rates
        self.F = np.full(pop_size, initial_F)  # Individual mutation factors
        self.learning_rate = learning_rate
        self.ring_neighborhood = ring_neighborhood
        self.population = None
        self.fitness = None
        self.f_opt = np.Inf
        self.x_opt = None
        self.generation = 0
        self.success_Cr = np.zeros(pop_size)
        self.success_F = np.zeros(pop_size)
        self.success_count = np.zeros(pop_size)
        self.min_F = 0.1
        self.max_F = 1.0
        self.epsilon = 1e-6

    def initialize_population(self, func):
        self.population = np.random.uniform(func.bounds.lb, func.bounds.ub, size=(self.pop_size, self.dim))
        self.fitness = np.array([func(x) for x in self.population])
        self.budget -= self.pop_size
        self.f_opt = np.min(self.fitness)
        self.x_opt = self.population[np.argmin(self.fitness)]

    def evolve(self, func):
        new_population = np.copy(self.population)
        new_fitness = np.copy(self.fitness)

        for i in range(self.pop_size):
            # Ring Topology Selection: Select neighbors
            neighbors = [(i + j) % self.pop_size for j in range(-self.ring_neighborhood, self.ring_neighborhood + 1) if j != 0]
            
            # Mutation: Use the best neighbor within the ring
            best_neighbor_idx = neighbors[np.argmin(self.fitness[neighbors])]
            
            available_indices = [idx for idx in range(self.pop_size) if idx not in [i, best_neighbor_idx]]
            if len(available_indices) < 2:
                # Handle edge cases (very small population)
                r1, r2 = np.random.choice(self.pop_size, 2, replace=False)
                while r1 == i or r2 == i:
                     r1, r2 = np.random.choice(self.pop_size, 2, replace=False)
            else:    
                r1, r2 = np.random.choice(available_indices, 2, replace=False)  # Ensure r1 and r2 are different
            
            mutant = self.population[i] + self.F[i] * (self.population[best_neighbor_idx] - self.population[i]) + self.F[i] * (self.population[r1] - self.population[r2])

            # Crossover
            for j in range(self.dim):
                if np.random.rand() < self.Cr[i]:
                    new_population[i, j] = mutant[j]
                else:
                    new_population[i, j] = self.population[i, j]

            new_population[i] = np.clip(new_population[i], func.bounds.lb, func.bounds.ub)

            # Evaluation
            new_fitness[i] = func(new_population[i])
            self.budget -= 1

            # Selection and Parameter Adaptation
            if new_fitness[i] < self.fitness[i]:
                self.success_Cr[i] = self.Cr[i]
                self.success_F[i] = self.F[i]
                self.success_count[i] += 1
                self.population[i] = new_population[i]
                self.fitness[i] = new_fitness[i]

                if new_fitness[i] < self.f_opt:
                    self.f_opt = new_fitness[i]
                    self.x_opt = new_population[i]

        # Update Cr and F values: Adaptation based on success history
        for i in range(self.pop_size):
            if self.success_count[i] > 0:
                # Update Cr and F based on successful values
                self.Cr[i] = (1 - self.learning_rate) * self.Cr[i] + self.learning_rate * self.success_Cr[i]
                self.F[i] = (1 - self.learning_rate) * self.F[i] + self.learning_rate * self.success_F[i]

                # Reset success counters
                self.success_Cr[i] = 0.0
                self.success_F[i] = 0.0
                self.success_count[i] = 0

            # Apply bounds
            self.F[i] = np.clip(self.F[i], self.min_F, self.max_F)


    def __call__(self, func):
        self.initialize_population(func)
        while self.budget > self.pop_size:
            self.evolve(func)
            self.generation += 1
        return self.f_opt, self.x_opt