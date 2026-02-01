import numpy as np

class AdaptiveDE_Ortho_Archive_LocalSearch_v4:
    def __init__(self, budget=10000, dim=10, pop_size_min=20, pop_size_max=100, archive_size=10, F_initial=0.5, CR_initial=0.5, F_learning_rate=0.1, CR_learning_rate=0.1, stagnation_threshold=1000, ortho_group_size=5, archive_update_frequency=5, lars_tolerance=1e-5, local_search_radius=0.1, local_search_frequency=10, cr_adaptation_rate=0.1, ls_success_threshold=0.25, ls_step_size_reduction=0.5):
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
        self.local_search_radius = local_search_radius  # Initial local search radius
        self.local_search_frequency = local_search_frequency # How often to perform local search
        self.cr_adaptation_rate = cr_adaptation_rate
        self.cr_success_rate = 0.5
        self.cr_success_history = []
        self.ls_success_threshold = ls_success_threshold # Threshold to reduce local search step size
        self.ls_step_size_reduction = ls_step_size_reduction # Reduction factor for local search step size

    def initialize_population(self, func):
        self.population = np.random.uniform(func.bounds.lb, func.bounds.ub, size=(self.pop_size, self.dim))
        self.fitness = np.array([func(x) for x in self.population])
        self.budget -= self.pop_size
        self.archive = []

    def mutate(self):
        mutated_population = np.zeros_like(self.population)
        for i in range(self.pop_size):
            rand = np.random.rand()
            if rand < 0.33:  # Current-to-best mutation
                best_idx = np.argmin(self.fitness)
                idxs = np.random.choice(self.pop_size, 2, replace=False)
                x1, x2 = self.population[idxs]
                mutated_population[i] = self.population[i] + self.F * (self.population[best_idx] - self.population[i]) + self.F * (x1 - x2)
            elif rand < 0.66:  # Random mutation
                idxs = np.random.choice(self.pop_size, 3, replace=False)
                x1, x2, x3 = self.population[idxs]
                mutated_population[i] = x1 + self.F * (x2 - x3)
            else:  # Cauchy mutation
                scale = 0.1 * self.F  # Adjust scale as needed
                cauchy_vector = scale * np.random.standard_cauchy(size=self.dim)
                mutated_population[i] = self.population[i] + cauchy_vector

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
                    crossed_population[i, j] = mutated_population[i, j]
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
                self.cr_success_history.append(1)
            else:
                 self.cr_success_history.append(0)
                
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
            #self.CR = np.clip((1 - self.CR_learning_rate) * self.CR + self.CR_learning_rate * self.CR_momentum, 0.1, 0.9) # Clip CR

        self.successful_F = []
        self.successful_CR = []

        # Population size adaptation based on landscape awareness
        if self.generation > 50:
            fitness_std = np.std(self.fitness)
            if fitness_std < self.lars_tolerance and self.pop_size > self.pop_size_min:
                self.pop_size = max(self.pop_size - int(self.pop_size_adaptation_rate * (self.pop_size - self.pop_size_min)), self.pop_size_min)
            elif fitness_std > self.lars_tolerance and self.pop_size < self.pop_size_max:
                self.pop_size = min(self.pop_size + int(self.pop_size_adaptation_rate * (self.pop_size_max - self.pop_size)), self.pop_size_max)

        # Update CR based on success rate
        if self.cr_success_history:
            success_rate = np.mean(self.cr_success_history)
            self.cr_success_rate = (1 - self.cr_adaptation_rate) * self.cr_success_rate + self.cr_adaptation_rate * success_rate
            self.CR = np.clip(self.cr_success_rate, 0.1, 0.9)
            self.cr_success_history = []

        return new_population, new_fitness

    def update_archive(self):
        combined_population = np.concatenate((self.population, np.array(self.archive) if self.archive else self.population[:0]))
        combined_fitness = np.concatenate((self.fitness, np.array([np.inf] * len(self.archive)) if self.archive else self.fitness[:0]))

        # Use stochastic ranking to improve archive diversity
        combined_population, combined_fitness = self.stochastic_ranking(combined_population, combined_fitness)

        # Select top archive_size individuals + some random individuals to promote diversity
        num_elite = int(0.8 * self.archive_size)  # Keep top 80%
        elite_indices = np.arange(num_elite)
        random_indices = np.random.choice(len(combined_population), self.archive_size - num_elite, replace=False)
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
        # Re-initialize population around the best solution with added noise
        best_x = self.population[np.argmin(self.fitness)]
        self.population = np.random.normal(loc=best_x, scale=0.5, size=(self.pop_size, self.dim))  # Gaussian distribution around best
        self.population = self.handle_bounds(self.population, func)
        self.fitness = np.array([func(x) for x in self.population])
        self.budget -= self.pop_size
        self.stagnation_counter = 0  # Reset stagnation counter
        self.best_fitness_history = []
        self.local_search_radius = 0.1 # Reset the local search radius

    def local_search(self, func, x_opt):
        # Perform local search around the best solution
        num_evals = min(self.pop_size, self.budget) # Adaptive local evals depending on budget
        success_count = 0
        for _ in range(num_evals):
            # Generate a random perturbation within the radius
            perturbation = np.random.uniform(-self.local_search_radius, self.local_search_radius, size=self.dim)
            x_neighbor = x_opt + perturbation
            x_neighbor = self.handle_bounds(np.array([x_neighbor]), func)[0] # Keep within bounds

            f_neighbor = func(x_neighbor)
            self.budget -= 1

            if f_neighbor < self.f_opt:
                self.f_opt = f_neighbor
                self.x_opt = x_neighbor
                success_count += 1
        
        success_rate = success_count / num_evals if num_evals > 0 else 0

        if success_rate < self.ls_success_threshold:
            self.local_search_radius *= self.ls_step_size_reduction  # Reduce step size if not successful
        
        self.local_search_radius = max(self.local_search_radius, 1e-6) # Avoid zero radius

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

            # Adjust local search frequency based on stagnation and fitness variance
            if self.check_stagnation() or (self.generation % self.local_search_frequency == 0 and np.std(self.best_fitness_history[-min(self.stagnation_threshold, len(self.best_fitness_history)):]) < self.lars_tolerance):
                self.local_search(func, self.x_opt)
                if self.check_stagnation():
                    self.restart(func)
                    self.best_fitness_history.append(np.min(self.fitness))  # Record new best fitness after restart


        # Final local search at the end
        if self.budget > 0:
            self.local_search(func, self.x_opt)

        return self.f_opt, self.x_opt