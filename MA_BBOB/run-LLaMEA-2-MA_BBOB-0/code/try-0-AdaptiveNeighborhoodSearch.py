import numpy as np

class AdaptiveNeighborhoodSearch:
    def __init__(self, budget=10000, dim=10, pop_size=40, neighborhood_size=5, initial_exploration_rate=0.5, exploration_decay=0.995):
        self.budget = budget
        self.dim = dim
        self.pop_size = pop_size
        self.neighborhood_size = neighborhood_size
        self.initial_exploration_rate = initial_exploration_rate
        self.exploration_rate = initial_exploration_rate
        self.exploration_decay = exploration_decay
        self.population = None
        self.fitness = None
        self.x_opt = None
        self.f_opt = np.inf

    def __call__(self, func):
        self.population = np.random.uniform(func.bounds.lb, func.bounds.ub, size=(self.pop_size, self.dim))
        self.fitness = np.array([func(x) for x in self.population])
        self.budget -= self.pop_size

        best_index = np.argmin(self.fitness)
        self.x_opt = self.population[best_index]
        self.f_opt = self.fitness[best_index]

        while self.budget > 0:
            for i in range(self.pop_size):
                # Select a random neighbor
                neighbors = np.random.choice(self.pop_size, self.neighborhood_size, replace=False)
                best_neighbor_index = neighbors[np.argmin(self.fitness[neighbors])]

                # Adaptive mutation based on neighborhood
                mutation_scale = np.abs(self.population[best_neighbor_index] - self.population[i])
                mutation = np.random.normal(0, self.exploration_rate * mutation_scale, size=self.dim)
                new_position = self.population[i] + mutation
                new_position = np.clip(new_position, func.bounds.lb, func.bounds.ub)

                # Explore with decaying probability
                if np.random.rand() < self.exploration_rate:
                    new_position = np.random.uniform(func.bounds.lb, func.bounds.ub)

                f_new = func(new_position)
                self.budget -= 1

                if f_new < self.f_opt:
                    self.f_opt = f_new
                    self.x_opt = new_position

                if f_new < self.fitness[i]:
                    self.fitness[i] = f_new
                    self.population[i] = new_position
            
            # Decay the exploration rate
            self.exploration_rate *= self.exploration_decay

            best_index = np.argmin(self.fitness)
            if self.fitness[best_index] < self.f_opt:
                self.f_opt = self.fitness[best_index]
                self.x_opt = self.population[best_index]

        return self.f_opt, self.x_opt