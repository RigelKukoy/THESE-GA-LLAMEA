import numpy as np

class SOSBee:
    def __init__(self, budget=10000, dim=10, colony_size=50, scout_bees=5, neighborhood_size=3, reduction_factor=0.9):
        self.budget = budget
        self.dim = dim
        self.colony_size = colony_size
        self.scout_bees = scout_bees
        self.neighborhood_size = neighborhood_size  # Number of neighbors to consider for adaptation
        self.reduction_factor = reduction_factor # Reduction factor for step size adaptation

        self.population = None
        self.fitness = None
        self.step_sizes = None
        self.f_opt = np.Inf
        self.x_opt = None
        self.eval_count = 0


    def initialize_population(self, func):
        self.population = np.random.uniform(func.bounds.lb, func.bounds.ub, size=(self.colony_size, self.dim))
        self.fitness = np.array([func(x) for x in self.population])
        self.step_sizes = np.full((self.colony_size, self.dim), 0.1 * (func.bounds.ub - func.bounds.lb))  # Initialize step sizes
        self.eval_count += self.colony_size

        best_index = np.argmin(self.fitness)
        self.f_opt = self.fitness[best_index]
        self.x_opt = self.population[best_index]


    def scout_phase(self, func):
        for _ in range(self.scout_bees):
            index = np.random.randint(self.colony_size)  # Randomly select a bee to become a scout
            x_new = np.random.uniform(func.bounds.lb, func.bounds.ub)
            f_new = func(x_new)
            self.eval_count += 1

            if f_new < self.fitness[index]:
                self.fitness[index] = f_new
                self.population[index] = x_new
                self.step_sizes[index] = np.full(self.dim, 0.1 * (func.bounds.ub - func.bounds.lb)) # Reset step size for scout

                if f_new < self.f_opt:
                    self.f_opt = f_new
                    self.x_opt = x_new


    def employed_bee_phase(self, func):
        for i in range(self.colony_size):
            neighbor_indices = np.random.choice(self.colony_size, self.neighborhood_size, replace=False)
            
            # Adaptive step size adjustment based on neighbors
            delta = np.mean(self.population[neighbor_indices] - self.population[i], axis=0)
            x_new = self.population[i] + self.step_sizes[i] * delta
            x_new = np.clip(x_new, func.bounds.lb, func.bounds.ub)
            
            f_new = func(x_new)
            self.eval_count += 1

            if f_new < self.fitness[i]:
                self.fitness[i] = f_new
                self.population[i] = x_new

                # Reduce step size if improvement is found
                self.step_sizes[i] *= self.reduction_factor

                if f_new < self.f_opt:
                    self.f_opt = f_new
                    self.x_opt = x_new
            else:
                # Increase step size if no improvement is found, to explore more
                self.step_sizes[i] /= self.reduction_factor # Increase exploration


    def __call__(self, func):
        self.initialize_population(func)

        while self.eval_count < self.budget:
            self.employed_bee_phase(func)
            self.scout_phase(func)

        return self.f_opt, self.x_opt