import numpy as np

class SOMAdaptiveDE:
    def __init__(self, budget=10000, dim=10, pop_size=50, som_grid_size=10, learning_rate=0.1, sigma=1.0):
        self.budget = budget
        self.dim = dim
        self.pop_size = pop_size
        self.som_grid_size = som_grid_size  # Size of the SOM grid (som_grid_size x som_grid_size)
        self.learning_rate = learning_rate  # SOM learning rate
        self.sigma = sigma  # SOM neighborhood radius
        self.som = np.random.rand(som_grid_size, som_grid_size, dim)  # Initialize SOM weights randomly
        self.F = 0.5
        self.CR = 0.9

    def __call__(self, func):
        self.f_opt = np.Inf
        self.x_opt = None
        self.population = np.random.uniform(func.bounds.lb, func.bounds.ub, size=(self.pop_size, self.dim))
        fitness = np.array([func(x) for x in self.population])
        self.budget -= self.pop_size

        for i in range(self.pop_size):
            if fitness[i] < self.f_opt:
                self.f_opt = fitness[i]
                self.x_opt = self.population[i]

        while self.budget > 0:
            # SOM Training Step
            for x in self.population:
                self.train_som(x)

            for i in range(self.pop_size):
                # Find Best Matching Unit (BMU) in SOM for current individual
                bmu_row, bmu_col = self.find_bmu(self.population[i])

                # Calculate neighborhood diversity around the BMU
                diversity = self.calculate_neighborhood_diversity(bmu_row, bmu_col)

                # Adjust F and CR based on neighborhood diversity
                self.F = 0.1 + 0.8 * diversity  # F ranges from 0.1 to 0.9
                self.CR = 0.1 + 0.8 * (1 - diversity) # CR ranges from 0.1 to 0.9 (inverse relation)


                # Mutation
                idxs = np.random.choice(self.pop_size, 3, replace=False)
                x_1, x_2, x_3 = self.population[idxs]
                mutant = x_1 + self.F * (x_2 - x_3)
                mutant = np.clip(mutant, func.bounds.lb, func.bounds.ub)

                # Crossover
                crossover = np.random.uniform(size=self.dim) < self.CR
                trial = np.where(crossover, mutant, self.population[i])

                # Selection
                f_trial = func(trial)
                self.budget -= 1

                if f_trial < self.f_opt:
                    self.f_opt = f_trial
                    self.x_opt = trial

                if f_trial < fitness[i]:
                    fitness[i] = f_trial
                    self.population[i] = trial

        return self.f_opt, self.x_opt

    def train_som(self, x):
        # Find Best Matching Unit (BMU)
        bmu_row, bmu_col = self.find_bmu(x)

        # Update SOM weights in the neighborhood of the BMU
        for row in range(self.som_grid_size):
            for col in range(self.som_grid_size):
                distance = np.sqrt((row - bmu_row)**2 + (col - bmu_col)**2)
                if distance <= self.sigma:
                    influence = np.exp(-distance**2 / (2 * self.sigma**2))
                    self.som[row, col] += self.learning_rate * influence * (x - self.som[row, col])

    def find_bmu(self, x):
        # Find the Best Matching Unit (BMU) for a given input vector x
        min_dist = np.inf
        bmu_row, bmu_col = -1, -1
        for row in range(self.som_grid_size):
            for col in range(self.som_grid_size):
                dist = np.linalg.norm(x - self.som[row, col])
                if dist < min_dist:
                    min_dist = dist
                    bmu_row, bmu_col = row, col
        return bmu_row, bmu_col

    def calculate_neighborhood_diversity(self, row, col):
        # Calculate the average Euclidean distance between SOM weights in the neighborhood
        distances = []
        for i in range(max(0, row - 1), min(self.som_grid_size, row + 2)):
            for j in range(max(0, col - 1), min(self.som_grid_size, col + 2)):
                if i != row or j != col:
                    distances.append(np.linalg.norm(self.som[row, col] - self.som[i, j]))
        if distances:
            return np.mean(distances) / np.linalg.norm(self.som[row, col])  # Normalize by magnitude of BMU vector
        else:
            return 0.0  # If no neighbors, diversity is 0