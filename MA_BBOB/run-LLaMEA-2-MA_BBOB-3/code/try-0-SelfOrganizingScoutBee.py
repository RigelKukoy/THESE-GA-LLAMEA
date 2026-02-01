import numpy as np

class SelfOrganizingScoutBee:
    def __init__(self, budget=10000, dim=10, initial_pop_size=20, scout_rate=0.1, min_pop_size=5):
        self.budget = budget
        self.dim = dim
        self.initial_pop_size = initial_pop_size
        self.scout_rate = scout_rate  # Percentage of scouts to explore
        self.min_pop_size = min_pop_size
        self.population = None
        self.fitness = None
        self.step_size = 1.0
        self.shrink_factor = 0.95 # Step size adaptation
        self.enlarge_factor = 1.1
        self.local_search_iterations = 5 # Local search intensity

    def initialize_population(self, func):
        self.population = np.random.uniform(func.bounds.lb, func.bounds.ub, size=(self.initial_pop_size, self.dim))
        self.fitness = np.array([func(x) for x in self.population])
        self.budget -= self.initial_pop_size

    def scout_bee_phase(self, func):
        num_scouts = max(1, int(len(self.population) * self.scout_rate))
        worst_indices = np.argsort(self.fitness)[-num_scouts:]  # Indices of worst bees

        for i in worst_indices:
            new_position = np.random.uniform(func.bounds.lb, func.bounds.ub)
            new_fitness = func(new_position)
            self.budget -= 1

            if new_fitness < self.fitness[i]:
                self.population[i] = new_position
                self.fitness[i] = new_fitness

            # Local Search around the current best scout
            best_index = np.argmin(self.fitness)
            best_position = self.population[best_index]
            for _ in range(self.local_search_iterations):
                perturbation = np.random.uniform(-self.step_size, self.step_size, size=self.dim)
                new_position = best_position + perturbation
                new_position = np.clip(new_position, func.bounds.lb, func.bounds.ub)
                new_fitness = func(new_position)
                self.budget -=1

                if new_fitness < self.fitness[i]:
                  self.population[i] = new_position
                  self.fitness[i] = new_fitness


        # Dynamic population adjustment: remove duplicated individuals
        unique_rows, indices = np.unique(self.population, axis=0, return_index=True)
        self.population = self.population[np.sort(indices)]
        self.fitness = self.fitness[np.sort(indices)]

        # Population control if it drops too low
        if len(self.population) < self.min_pop_size and self.budget > self.min_pop_size:
            num_needed = self.min_pop_size - len(self.population)
            new_individuals = np.random.uniform(func.bounds.lb, func.bounds.ub, size=(num_needed, self.dim))
            new_fitnesses = [func(x) for x in new_individuals]
            self.budget -= num_needed
            self.population = np.vstack((self.population, new_individuals))
            self.fitness = np.concatenate((self.fitness, new_fitnesses))


    def employed_bee_phase(self, func):
        for i in range(len(self.population)):
            neighbor_index = np.random.choice(len(self.population))
            while neighbor_index == i:
                neighbor_index = np.random.choice(len(self.population))

            # Perturb the current solution
            perturbation = np.random.uniform(-self.step_size, self.step_size, size=self.dim)
            new_position = self.population[i] + perturbation
            new_position = np.clip(new_position, func.bounds.lb, func.bounds.ub)
            new_fitness = func(new_position)
            self.budget -= 1

            # Greedy selection
            if new_fitness < self.fitness[i]:
                self.population[i] = new_position
                self.fitness[i] = new_fitness

    def adjust_step_size(self):
        # Simple step size adaptation based on fitness variance
        fitness_variance = np.var(self.fitness)
        if fitness_variance < 1e-6:
            self.step_size *= self.enlarge_factor  # Increase exploration
        else:
            self.step_size *= self.shrink_factor  # Reduce step size for exploitation
        self.step_size = np.clip(self.step_size, 0.01, 2.0) # Limit the step size

    def __call__(self, func):
        self.f_opt = np.Inf
        self.x_opt = None
        self.initialize_population(func)

        while self.budget > 0:
            self.employed_bee_phase(func)
            self.scout_bee_phase(func)
            self.adjust_step_size()

            best_index = np.argmin(self.fitness)
            if self.fitness[best_index] < self.f_opt:
                self.f_opt = self.fitness[best_index]
                self.x_opt = self.population[best_index]

        return self.f_opt, self.x_opt