import numpy as np

class SelfOrganizingScoutBee:
    def __init__(self, budget=10000, dim=10, pop_size=50, scout_bees=5, scout_frequency_initial=0.1, scout_frequency_decay=0.95, elite_fraction=0.1):
        self.budget = budget
        self.dim = dim
        self.pop_size = pop_size
        self.scout_bees = scout_bees
        self.scout_frequency = scout_frequency_initial
        self.scout_frequency_decay = scout_frequency_decay
        self.elite_fraction = elite_fraction

    def __call__(self, func):
        # Initialize population
        population = np.random.uniform(func.bounds.lb, func.bounds.ub, size=(self.pop_size, self.dim))
        fitness = np.array([func(x) for x in population])
        self.budget -= self.pop_size

        self.f_opt = np.min(fitness)
        self.x_opt = population[np.argmin(fitness)]

        while self.budget > self.pop_size:
            # Sort population by fitness
            sorted_indices = np.argsort(fitness)
            population = population[sorted_indices]
            fitness = fitness[sorted_indices]

            # Employed bees phase (exploitation)
            for i in range(self.pop_size):
                # Select a random neighbor
                neighbor_index = np.random.randint(0, self.pop_size)
                while neighbor_index == i:
                    neighbor_index = np.random.randint(0, self.pop_size)

                # Create a new solution by modifying the current solution
                phi = np.random.uniform(-1, 1, size=self.dim)
                new_solution = population[i] + phi * (population[i] - population[neighbor_index])
                new_solution = np.clip(new_solution, func.bounds.lb, func.bounds.ub)

                # Evaluate the new solution
                new_fitness = func(new_solution)
                self.budget -= 1
                if self.budget <= 0:
                    return self.f_opt, self.x_opt

                # Greedy selection
                if new_fitness < fitness[i]:
                    population[i] = new_solution
                    fitness[i] = new_fitness

                    if new_fitness < self.f_opt:
                        self.f_opt = new_fitness
                        self.x_opt = new_solution

            # Scout bees phase (exploration) - Adaptive frequency
            num_elite = int(self.elite_fraction * self.pop_size)
            mean_fitness_elite = np.mean(fitness[:num_elite])
            mean_fitness_non_elite = np.mean(fitness[num_elite:])

            # Adjust scout bee frequency based on elite vs non-elite fitness
            if mean_fitness_non_elite > mean_fitness_elite:
                self.scout_frequency *= (2 - self.scout_frequency_decay) # Increase scout frequency
            else:
                self.scout_frequency *= self.scout_frequency_decay # Decrease scout frequency

            self.scout_frequency = np.clip(self.scout_frequency, 0.01, 0.5) # limit the scout frequency

            for i in range(self.pop_size):
                 if np.random.rand() < self.scout_frequency:
                    # Replace with a new random solution (scout bee)
                    population[i] = np.random.uniform(func.bounds.lb, func.bounds.ub, size=self.dim)
                    fitness[i] = func(population[i])
                    self.budget -= 1
                    if self.budget <= 0:
                        return self.f_opt, self.x_opt
                    
                    if fitness[i] < self.f_opt:
                        self.f_opt = fitness[i]
                        self.x_opt = population[i]
        
        return self.f_opt, self.x_opt