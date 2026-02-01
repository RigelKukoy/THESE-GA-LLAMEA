import numpy as np

class AdaptiveDE_OrthoAgingRestart:
    def __init__(self, budget=10000, dim=10, pop_size=None, archive_size=10, F_initial=0.5, CR_initial=0.5, F_learning_rate=0.1, CR_learning_rate=0.1, pop_adapt_rate=0.05, aging_rate=0.01, stagnation_threshold=1e-4, stagnation_reset_probability=0.1):
        self.budget = budget
        self.dim = dim
        self.pop_size = pop_size if pop_size is not None else int(4 + 3 * np.log(dim))  # Adaptive population size
        self.archive_size = archive_size
        self.F = F_initial  # Mutation factor
        self.CR = CR_initial  # Crossover rate
        self.population = None
        self.fitness = None
        self.archive = None
        self.F_learning_rate = F_learning_rate
        self.CR_learning_rate = CR_learning_rate
        self.successful_F = []
        self.successful_CR = []
        self.F_momentum = 0.0
        self.CR_momentum = 0.0
        self.momentum_beta = 0.9
        self.pop_adapt_rate = pop_adapt_rate
        self.aging_rate = aging_rate
        self.age = None # Age of each individual
        self.stagnation_threshold = stagnation_threshold
        self.stagnation_reset_probability = stagnation_reset_probability
        self.best_fitness_history = []

    def initialize_population(self, func):
        self.population = np.random.uniform(func.bounds.lb, func.bounds.ub, size=(self.pop_size, self.dim))
        self.fitness = np.array([func(x) for x in self.population])
        self.budget -= self.pop_size
        self.archive = []
        self.age = np.zeros(self.pop_size)
        self.best_fitness_history = [np.min(self.fitness)]

    def mutate(self):
        mutated_population = np.zeros_like(self.population)
        for i in range(self.pop_size):
            idxs = np.random.choice(self.pop_size, 3, replace=False)
            x1, x2, x3 = self.population[idxs]
            mutated_population[i] = x1 + self.F * (x2 - x3)
        return mutated_population

    def orthogonal_crossover(self, mutated_population):
        crossed_population = np.copy(self.population)
        for i in range(self.pop_size):
            # Select two parents randomly
            parent1_idx, parent2_idx = np.random.choice(self.pop_size, 2, replace=False)
            parent1 = self.population[parent1_idx]
            parent2 = mutated_population[parent2_idx]

            # Generate an orthogonal matrix (Hadamard matrix)
            H = self.generate_hadamard_matrix(self.dim)

            # Create offspring using orthogonal crossover
            for j in range(self.dim):
                crossed_population[i, j] = 0.5 * (parent1[j] + parent2[j]) + 0.5 * H[i % self.dim, j] * (parent1[j] - parent2[j])

        return crossed_population

    def generate_hadamard_matrix(self, n):
        # Ensure n is a power of 2
        if n & (n - 1) != 0:
            n = 2 ** int(np.ceil(np.log2(n)))  # Round up to the nearest power of 2

        if n == 1:
            return np.array([[1]])

        H = np.array([[1]])
        while H.shape[0] < n:
            H = np.kron(H, np.array([[1, 1], [1, -1]]))
        return H

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

    def calculate_crowding_distance(self, self, pop, fitness):
        distances = np.zeros(len(pop))

        if len(pop) <= 2:
            distances[:] = np.inf  # Assign infinite distance to all in small populations
            return distances

        # Normalize each dimension
        normalized_pop = (pop - np.min(pop, axis=0)) / (np.max(pop, axis=0) - np.min(pop, axis=0) + 1e-8)

        for i in range(self.dim):
            sorted_indices = np.argsort(normalized_pop[:, i])
            distances[sorted_indices[0]] = np.inf
            distances[sorted_indices[-1]] = np.inf

            for j in range(1, len(pop) - 1):
                distances[sorted_indices[j]] += normalized_pop[sorted_indices[j+1], i] - normalized_pop[sorted_indices[j-1], i]

        return distances

    def select(self, func, crossed_population):
        new_population = np.copy(self.population)
        new_fitness = np.copy(self.fitness)

        for i in range(self.pop_size):
            f = func(crossed_population[i])
            self.budget -= 1

            if f < self.fitness[i]:
                new_population[i] = crossed_population[i]
                new_fitness[i] = f
                self.age[i] = 0  # Reset age
                self.successful_F.append(self.F)
                self.successful_CR.append(self.CR)

                if len(self.archive) < self.archive_size:
                    self.archive.append(self.population[i].copy())
                else:
                    idx_to_replace = np.random.randint(0, self.archive_size)
                    self.archive[idx_to_replace] = self.population[i].copy()
            else:
                self.age[i] += 1  # Increase age

        # Stochastic Ranking Selection
        combined_population = np.vstack((self.population, crossed_population))
        combined_fitness = np.concatenate((self.fitness, [func(x) for x in crossed_population]))
        self.budget -= len(crossed_population)

        ranked_population, ranked_fitness = self.stochastic_ranking(combined_population, combined_fitness)

        # Crowding Distance
        crowding_distances = self.calculate_crowding_distance(ranked_population, ranked_fitness)

        # Select based on rank and crowding distance
        selected_indices = np.argsort(-crowding_distances)[:self.pop_size]  # Take top pop_size based on crowding distance
        new_population = ranked_population[selected_indices]
        new_fitness = ranked_fitness[selected_indices]

        # Update F and CR with momentum
        if self.successful_F:
            mean_F = np.mean(self.successful_F)
            mean_CR = np.mean(self.successful_CR)

            self.F_momentum = self.momentum_beta * self.F_momentum + (1 - self.momentum_beta) * mean_F
            self.CR_momentum = self.momentum_beta * self.CR_momentum + (1 - self.momentum_beta) * mean_CR

            self.F = (1 - self.F_learning_rate) * self.F + self.F_learning_rate * self.F_momentum
            self.CR = (1 - self.CR_learning_rate) * self.CR + self.CR_learning_rate * self.CR_momentum

        self.successful_F = []
        self.successful_CR = []

        # Aging mechanism: replace old individuals
        for i in range(self.pop_size):
            if self.age[i] > self.dim * 2:  # Threshold for aging
                new_population[i] = np.random.uniform(func.bounds.lb, func.bounds.ub)
                new_fitness[i] = func(new_population[i])
                self.budget -= 1
                self.age[i] = 0

        # Adaptive population size
        diversity = np.std(new_fitness)
        if diversity < self.stagnation_threshold:  # Stagnation detection
            self.pop_size = min(self.pop_size + int(self.pop_adapt_rate * self.dim), 200)
        else:
            self.pop_size = max(int(4 + 3 * np.log(self.dim)), int(self.pop_size * (1 - self.pop_adapt_rate)))

        return new_population, new_fitness

    def check_stagnation(self):
        # Check if the optimization has stagnated
        if len(self.best_fitness_history) > 20:
            recent_improvements = np.diff(self.best_fitness_history[-20:])
            if np.max(recent_improvements) < self.stagnation_threshold:
                return True
        return False

    def restart_population(self, func):
        # Restart the population with new random individuals
        new_population = np.random.uniform(func.bounds.lb, func.bounds.ub, size=(self.pop_size, self.dim))
        new_fitness = np.array([func(x) for x in new_population])
        self.budget -= self.pop_size
        self.population = new_population
        self.fitness = new_fitness
        self.age = np.zeros(self.pop_size)
        self.successful_F = []
        self.successful_CR = []
        print("Restarting population due to stagnation.")

    def __call__(self, func):
        self.initialize_population(func)

        self.f_opt = np.min(self.fitness)
        self.x_opt = self.population[np.argmin(self.fitness)]

        while self.budget > self.pop_size:
            mutated_population = self.mutate()
            crossed_population = self.orthogonal_crossover(mutated_population)
            crossed_population = self.handle_bounds(crossed_population, func)
            self.population, self.fitness = self.select(func, crossed_population)

            current_best_fitness = np.min(self.fitness)
            if current_best_fitness < self.f_opt:
                self.f_opt = current_best_fitness
                self.x_opt = self.population[np.argmin(self.fitness)]
            self.best_fitness_history.append(current_best_fitness)

            # Stagnation check and restart
            if self.check_stagnation() and np.random.rand() < self.stagnation_reset_probability:
                self.restart_population(func)

        return self.f_opt, self.x_opt