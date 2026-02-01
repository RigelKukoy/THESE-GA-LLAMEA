import numpy as np

class AdaptiveDE_Ortho_Archive_Improved2:
    def __init__(self, budget=10000, dim=10, pop_size_min=20, pop_size_max=100, archive_size=10, F_initial=0.5, CR_initial=0.5, F_learning_rate=0.1, CR_learning_rate=0.1, stagnation_threshold=1000, ortho_group_size=5, archive_update_frequency=5, lars_tolerance=1e-5, convex_coeff=0.25):
        self.budget = budget
        self.dim = dim
        self.pop_size_min = pop_size_min
        self.pop_size_max = pop_size_max
        self.pop_size = pop_size_max  # Start with a larger population
        self.archive_size = archive_size
        self.F = F_initial  # Mutation factor
        self.CR = CR_initial  # Crossover rate
        self.population = None
        self.fitness = None
        self.archive = []
        self.F_learning_rate = F_learning_rate
        self.CR_learning_rate = CR_learning_rate
        self.successful_F = []
        self.successful_CR = []
        self.F_momentum = 0.0
        self.CR_momentum = 0.0
        self.momentum_coeff = 0.9
        self.stagnation_threshold = stagnation_threshold
        self.stagnation_counter = 0
        self.best_fitness_history = []
        self.ortho_group_size = ortho_group_size  # Size of groups for orthogonal crossover
        self.pop_size_adaptation_rate = 0.1  # Rate to adjust pop size
        self.archive_update_frequency = archive_update_frequency
        self.generation = 0
        self.lars_tolerance = lars_tolerance
        self.convex_coeff = convex_coeff # Coefficient for convex crossover

    def initialize_population(self, func):
        self.population = np.random.uniform(func.bounds.lb, func.bounds.ub, size=(self.pop_size, self.dim))
        self.fitness = np.array([func(x) for x in self.population])
        self.budget -= self.pop_size
        self.archive = []

    def mutate(self):
        mutated_population = np.zeros_like(self.population)
        for i in range(self.pop_size):
            if np.random.rand() < 0.5:  # Current-to-best mutation
                best_idx = np.argmin(self.fitness)
                idxs = np.random.choice(self.pop_size, 2, replace=False)
                x1, x2 = self.population[idxs]
                mutated_population[i] = self.population[i] + self.F * (self.population[best_idx] - self.population[i]) + self.F * (x1 - x2)
            else:  # Random mutation
                idxs = np.random.choice(self.pop_size, 3, replace=False)
                x1, x2, x3 = self.population[idxs]
                mutated_population[i] = x1 + self.F * (x2 - x3)

        # Archive-guided mutation
        if self.archive:
            archive_idx = np.random.randint(len(self.archive))
            archived_vector = self.archive[archive_idx]
            idxs = np.random.choice(self.pop_size, 2, replace=False)
            x1, x2 = self.population[idxs]
            mutated_population[np.random.randint(self.pop_size)] = archived_vector + self.F * (x1 - x2)

        return mutated_population

    def crossover(self, mutated_population):
        crossed_population = np.zeros_like(self.population)
        for i in range(self.pop_size):
            # Perform orthogonal crossover in groups
            group_idx = i // self.ortho_group_size
            start_idx = group_idx * self.ortho_group_size
            end_idx = min((group_idx + 1) * self.ortho_group_size, self.pop_size)

            group = np.arange(start_idx, end_idx)
            for j in range(self.dim):
                if np.random.rand() < self.CR or j == np.random.randint(0, self.dim):
                     # Convex crossover
                    crossed_population[i, j] = self.convex_coeff * mutated_population[i, j] + (1 - self.convex_coeff) * self.population[i, j]
                else:
                    crossed_population[i, j] = self.population[i, j]

        return crossed_population

    def handle_bounds(self, population, func):
        lb = func.bounds.lb
        ub = func.bounds.ub
        return np.clip(population, lb, ub)

    def stochastic_ranking(self, pop, fitness):
        N = len(pop)
        indices = np.arange(N)
        np.random.shuffle(indices)

        def compare(i, j):
            fi = fitness[i]
            fj = fitness[j]

            if (fi < 0 and fj < 0) or (fi >= 0 and fj >= 0):
                return fi - fj
            elif fi < 0 and fj >= 0:
                return -1
            else:
                return 1

        ranked_indices = sorted(indices, key=lambda k: fitness[k])
        return pop[ranked_indices], fitness[ranked_indices]

    def crowding_distance(self, population, fitness):
        distances = np.zeros(len(population))
        for m in range(self.dim):  # Consider each dimension
            dimension_values = population[:, m]
            sorted_indices = np.argsort(dimension_values)
            distances[sorted_indices[0]] = distances[sorted_indices[-1]] = np.inf  # Boundary individuals

            for i in range(1, len(population) - 1):
                distances[sorted_indices[i]] += (dimension_values[sorted_indices[i + 1]] - dimension_values[sorted_indices[i - 1]])

        return distances

    def select(self, func, crossed_population):
        new_population = np.copy(self.population)
        new_fitness = np.copy(self.fitness)

        improved_count = 0

        for i in range(self.pop_size):
            f = func(crossed_population[i])
            self.budget -= 1

            if f < self.fitness[i]:
                new_population[i] = crossed_population[i]
                new_fitness[i] = f
                improved_count += 1

                self.successful_F.append(self.F)
                self.successful_CR.append(self.CR)

        new_population, new_fitness = self.stochastic_ranking(new_population, new_fitness)

        # Update F and CR with momentum
        if self.successful_F:
            mean_F = np.mean(self.successful_F)
            mean_CR = np.mean(self.successful_CR)

            self.F_momentum = self.momentum_coeff * self.F_momentum + (1 - self.momentum_coeff) * mean_F
            self.CR_momentum = self.momentum_coeff * self.CR_momentum + (1 - self.momentum_coeff) * mean_CR

            self.F = np.clip((1 - self.F_learning_rate) * self.F + self.F_learning_rate * self.F_momentum, 0.1, 0.9) # Clip F
            self.CR = np.clip((1 - self.CR_learning_rate) * self.CR + self.CR_learning_rate * self.CR_momentum, 0.1, 0.9) # Clip CR

        self.successful_F = []
        self.successful_CR = []

        # Population size adaptation based on landscape awareness and budget
        if self.generation > 50:
            fitness_std = np.std(self.fitness)
            if fitness_std < self.lars_tolerance and self.pop_size > self.pop_size_min:
                # Reduce population size only if enough budget remains
                if self.budget > self.pop_size:
                     self.pop_size = max(self.pop_size - int(self.pop_size_adaptation_rate * (self.pop_size - self.pop_size_min)), self.pop_size_min)
            elif fitness_std > self.lars_tolerance and self.pop_size < self.pop_size_max:
                self.pop_size = min(self.pop_size + int(self.pop_size_adaptation_rate * (self.pop_size_max - self.pop_size)), self.pop_size_max)


        return new_population, new_fitness

    def update_archive(self):
        combined_population = np.concatenate((self.population, np.array(self.archive) if self.archive else self.population[:0]))
        combined_fitness = np.concatenate((self.fitness, np.array([np.inf] * len(self.archive)) if self.archive else self.fitness[:0]))

        # Use stochastic ranking to improve archive diversity
        combined_population, combined_fitness = self.stochastic_ranking(combined_population, combined_fitness)

        # Select top archive_size individuals + some random individuals to promote diversity
        num_elite = int(0.8 * self.archive_size)  # Keep top 80%
        elite_indices = np.argsort(combined_fitness)[:num_elite] # Select based on fitness
        remaining = self.archive_size - num_elite
        random_indices = np.random.choice(len(combined_population), remaining, replace=False)

        selected_indices = np.concatenate((elite_indices, random_indices))
        np.random.shuffle(selected_indices)  # Shuffle to mix elite and random
        
        self.archive = list(combined_population[selected_indices])

    def check_stagnation(self):
        if len(self.best_fitness_history) >= self.stagnation_threshold:
            if np.std(self.best_fitness_history[-self.stagnation_threshold:]) < 1e-8:
                self.stagnation_counter += 1
            else:
                self.stagnation_counter = 0

        if self.stagnation_counter >= 3:
            return True
        return False

    def restart(self, func):
        self.population = np.random.uniform(func.bounds.lb, func.bounds.ub, size=(self.pop_size, self.dim))
        self.fitness = np.array([func(x) for x in self.population])
        self.budget -= self.pop_size
        self.stagnation_counter = 0  # Reset stagnation counter
        self.best_fitness_history = []

    def __call__(self, func):
        self.initialize_population(func)

        self.f_opt = np.min(self.fitness)
        self.x_opt = self.population[np.argmin(self.fitness)]
        self.best_fitness_history.append(self.f_opt)

        while self.budget > self.pop_size:  # Ensure enough budget for population updates
            mutated_population = self.mutate()
            crossed_population = self.crossover(mutated_population)
            crossed_population = self.handle_bounds(crossed_population, func)
            self.population, self.fitness = self.select(func, crossed_population)

            current_best_fitness = np.min(self.fitness)
            if current_best_fitness < self.f_opt:
                self.f_opt = current_best_fitness
                self.x_opt = self.population[np.argmin(self.fitness)]

            self.best_fitness_history.append(current_best_fitness)

            self.generation += 1
            if self.generation % self.archive_update_frequency == 0:
                self.update_archive()

            if self.check_stagnation():
                self.restart(func)
                self.best_fitness_history.append(np.min(self.fitness))  # Record new best fitness after restart

        return self.f_opt, self.x_opt