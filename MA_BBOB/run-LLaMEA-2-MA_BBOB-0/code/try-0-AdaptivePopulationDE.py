import numpy as np

class AdaptivePopulationDE:
    def __init__(self, budget=10000, dim=10, initial_pop_size=50, min_pop_size=10, max_pop_size=100, initial_mutation_factor=0.5, initial_crossover_rate=0.7, diversity_threshold=0.1):
        self.budget = budget
        self.dim = dim
        self.pop_size = initial_pop_size
        self.min_pop_size = min_pop_size
        self.max_pop_size = max_pop_size
        self.mutation_factor = initial_mutation_factor
        self.crossover_rate = initial_crossover_rate
        self.population = None
        self.fitness = None
        self.x_opt = None
        self.f_opt = np.inf
        self.success_mutation_factors = []
        self.success_crossover_rates = []
        self.diversity_threshold = diversity_threshold

    def initialize_population(self, lb, ub):
        self.population = np.random.uniform(lb, ub, size=(self.pop_size, self.dim))

    def calculate_diversity(self):
        # Calculate the average pairwise distance between individuals
        if self.pop_size <= 1:
            return 1.0  # Maximum diversity if only one individual

        distances = np.sum((self.population[:, None, :] - self.population[None, :, :]) ** 2, axis=2)
        distances = np.triu(distances, k=1)  # Upper triangle to avoid duplicates
        mean_distance = np.sum(distances) / (self.pop_size * (self.pop_size - 1) / 2)

        # Normalize diversity to be between 0 and 1
        diversity = np.clip(mean_distance / (10 * self.dim), 0, 1)  # scaling factor adjusted

        return diversity


    def adjust_population_size(self):
        diversity = self.calculate_diversity()
        if diversity < self.diversity_threshold and self.pop_size < self.max_pop_size:
            # Low diversity, increase population size
            increase_amount = min(int(self.pop_size * 0.1), self.max_pop_size - self.pop_size)
            if increase_amount > 0:
                new_individuals = np.random.uniform(func.bounds.lb, func.bounds.ub, size=(increase_amount, self.dim))
                self.population = np.vstack((self.population, new_individuals))
                new_fitness = np.array([func(x) for x in new_individuals])
                self.fitness = np.concatenate((self.fitness, new_fitness))
                self.pop_size += increase_amount
                self.budget -= increase_amount

        elif diversity > (1-self.diversity_threshold) and self.pop_size > self.min_pop_size:
            # High diversity, decrease population size
            decrease_amount = min(int(self.pop_size * 0.1), self.pop_size - self.min_pop_size)
            if decrease_amount > 0:
                #Remove worst performing individuals
                worst_indices = np.argsort(self.fitness)[-decrease_amount:]
                keep_indices = np.setdiff1d(np.arange(self.pop_size), worst_indices)
                self.population = self.population[keep_indices]
                self.fitness = self.fitness[keep_indices]
                self.pop_size -= decrease_amount


    def differential_evolution(self, func, lb, ub):
        new_population = np.copy(self.population)
        new_fitness = np.copy(self.fitness)

        for i in range(self.pop_size):
            idxs = np.random.choice(self.pop_size, 3, replace=False)
            x_r1, x_r2, x_r3 = self.population[idxs]

            # Diversity-guided mutation
            diversity = self.calculate_diversity()
            if diversity < self.diversity_threshold:
                # If diversity is low, explore more
                mutant = x_r1 + self.mutation_factor * (x_r2 - x_r3) + np.random.normal(0, 0.1, size=self.dim) #Increased Exploration
            else:
                # If diversity is high, exploit more
                mutant = x_r1 + self.mutation_factor * (x_r2 - x_r3)

            mutant = np.clip(mutant, lb, ub)

            crossover = np.random.uniform(size=self.dim) < self.crossover_rate
            trial = np.where(crossover, mutant, self.population[i])

            f_trial = func(trial)
            self.budget -= 1

            if f_trial < self.f_opt:
                self.f_opt = f_trial
                self.x_opt = trial

            if f_trial < self.fitness[i]:
                new_fitness[i] = f_trial
                new_population[i] = trial

                # Store successful parameters
                self.success_mutation_factors.append(self.mutation_factor)
                self.success_crossover_rates.append(self.crossover_rate)

        self.population = new_population
        self.fitness = new_fitness

    def adapt_parameters(self):
        # Adapt mutation factor and crossover rate based on success history
        if len(self.success_mutation_factors) > 0:
            self.mutation_factor = np.mean(self.success_mutation_factors)
            self.crossover_rate = np.mean(self.success_crossover_rates)

            # Add some noise to prevent stagnation
            self.mutation_factor = np.clip(self.mutation_factor + np.random.normal(0, 0.05), 0.1, 1.0) #Reduced Noise
            self.crossover_rate = np.clip(self.crossover_rate + np.random.normal(0, 0.05), 0.1, 1.0) #Reduced Noise

            self.success_mutation_factors = []
            self.success_crossover_rates = []
        else:
            # If no success, increase exploration
            self.mutation_factor = np.clip(self.mutation_factor + 0.1, 0.1, 1.0)
            self.crossover_rate = np.clip(self.crossover_rate - 0.1, 0.1, 1.0)

    def __call__(self, func):
        lb, ub = func.bounds.lb, func.bounds.ub
        self.initialize_population(lb, ub)
        self.fitness = np.array([func(x) for x in self.population])
        self.budget -= self.pop_size

        best_index = np.argmin(self.fitness)
        self.x_opt = self.population[best_index]
        self.f_opt = self.fitness[best_index]

        while self.budget > 0:
            self.adjust_population_size()
            self.differential_evolution(func, lb, ub)
            self.adapt_parameters()

        return self.f_opt, self.x_opt