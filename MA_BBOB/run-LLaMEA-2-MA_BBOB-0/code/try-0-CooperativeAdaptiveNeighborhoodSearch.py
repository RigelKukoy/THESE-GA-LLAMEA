import numpy as np

class CooperativeAdaptiveNeighborhoodSearch:
    def __init__(self, budget=10000, dim=10, num_neighborhoods=5, pop_size=20, initial_radius=0.5, radius_decay=0.995, migration_rate=0.1):
        self.budget = budget
        self.dim = dim
        self.num_neighborhoods = num_neighborhoods
        self.pop_size = pop_size
        self.initial_radius = initial_radius
        self.radius = [initial_radius] * num_neighborhoods  # radius for each neighborhood
        self.radius_decay = radius_decay
        self.migration_rate = migration_rate
        self.neighborhoods = []
        self.fitness = []
        self.x_opt = None
        self.f_opt = np.inf

    def initialize_neighborhood(self, func):
        population = np.random.uniform(func.bounds.lb, func.bounds.ub, size=(self.pop_size, self.dim))
        fitness = np.array([func(x) for x in population])
        return population, fitness

    def __call__(self, func):
        # Initialize neighborhoods
        for _ in range(self.num_neighborhoods):
            population, fitness = self.initialize_neighborhood(func)
            self.neighborhoods.append(population)
            self.fitness.append(fitness)
            self.budget -= self.pop_size

        # Initial best solution
        for i in range(self.num_neighborhoods):
            best_index = np.argmin(self.fitness[i])
            if self.fitness[i][best_index] < self.f_opt:
                self.f_opt = self.fitness[i][best_index]
                self.x_opt = self.neighborhoods[i][best_index]

        while self.budget > 0:
            for i in range(self.num_neighborhoods):
                for j in range(self.pop_size):
                    # Sample within the neighborhood
                    mutation = np.random.normal(0, self.radius[i], size=self.dim)
                    new_position = self.neighborhoods[i][j] + mutation
                    new_position = np.clip(new_position, func.bounds.lb, func.bounds.ub)

                    f_new = func(new_position)
                    self.budget -= 1

                    if f_new < self.f_opt:
                        self.f_opt = f_new
                        self.x_opt = new_position

                    if f_new < self.fitness[i][j]:
                        self.fitness[i][j] = f_new
                        self.neighborhoods[i][j] = new_position

                # Decay neighborhood radius
                self.radius[i] *= self.radius_decay

                # Migration of solutions between neighborhoods
                if np.random.rand() < self.migration_rate:
                    # Select a random neighborhood to migrate to
                    target_neighborhood = np.random.choice(self.num_neighborhoods)
                    if target_neighborhood != i:
                        # Replace a random individual in the target neighborhood
                        replace_index = np.random.randint(self.pop_size)
                        best_index = np.argmin(self.fitness[i])
                        self.neighborhoods[target_neighborhood][replace_index] = self.neighborhoods[i][best_index].copy()
                        self.fitness[target_neighborhood][replace_index] = self.fitness[i][best_index]

            # Update global best solution
            for i in range(self.num_neighborhoods):
                best_index = np.argmin(self.fitness[i])
                if self.fitness[i][best_index] < self.f_opt:
                    self.f_opt = self.fitness[i][best_index]
                    self.x_opt = self.neighborhoods[i][best_index]

        return self.f_opt, self.x_opt