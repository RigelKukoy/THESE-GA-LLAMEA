import numpy as np

class AdaptiveDE_Ortho_Archive_LocalSearch_v3:
    def __init__(self, budget=10000, dim=10, pop_size_min=20, pop_size_max=100, archive_size=10, F_initial=0.5, CR_initial=0.5, F_learning_rate=0.1, CR_learning_rate=0.1, stagnation_threshold=1000, ortho_group_size=5, archive_update_frequency=5, local_search_frequency=10, local_search_radius=0.1, local_search_radius_decay=0.95, restart_probability=0.05, ls_prob_initial=0.1, ls_radius_min = 0.001):
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
        self.local_search_frequency = local_search_frequency
        self.local_search_radius = local_search_radius
        self.local_search_radius_decay = local_search_radius_decay
        self.restart_probability = restart_probability
        self.ls_prob = ls_prob_initial # Local search probability
        self.ls_prob_initial = ls_prob_initial
        self.ls_success_rate = 0.0
        self.ls_success_history = []
        self.ls_history_length = 20
        self.ls_radius_min = ls_radius_min
        self.pop_success_rate = 0.0
        self.pop_success_history = []
        self.pop_history_length = 20


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

            self.F = (1 - self.F_learning_rate) * self.F + self.F_learning_rate * self.F_momentum
            self.CR = (1 - self.CR_learning_rate) * self.CR + self.CR_learning_rate * self.CR_momentum

        self.successful_F = []
        self.successful_CR = []

        # Population size adaptation based on a moving average success rate
        self.pop_success_history.append(improved_count / self.pop_size)
        if len(self.pop_success_history) > self.pop_history_length:
            self.pop_success_history.pop(0)

        self.pop_success_rate = np.mean(self.pop_success_history)

        if self.pop_success_rate > 0.3 and self.pop_size < self.pop_size_max:
            self.pop_size = min(self.pop_size + int(self.pop_size_adaptation_rate * (self.pop_size_max - self.pop_size)), self.pop_size_max)
        elif self.pop_success_rate < 0.1 and self.pop_size > self.pop_size_min:
            self.pop_size = max(self.pop_size - int(self.pop_size_adaptation_rate * (self.pop_size - self.pop_size_min)), self.pop_size_min)

        return new_population, new_fitness

    def update_archive(self):
        combined_population = np.concatenate((self.population, np.array(self.archive) if self.archive else self.population[:0]))
        combined_fitness = np.concatenate((self.fitness, np.array([np.inf] * len(self.archive)) if self.archive else self.fitness[:0]))

        # Use stochastic ranking to improve archive diversity
        combined_population, combined_fitness = self.stochastic_ranking(combined_population, combined_fitness)

        self.archive = list(combined_population[:self.archive_size])  # Select top archive_size individuals

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
        if np.random.rand() < self.restart_probability:
            self.population = np.random.uniform(func.bounds.lb, func.bounds.ub, size=(self.pop_size, self.dim))
            self.fitness = np.array([func(x) for x in self.population])
            self.budget -= self.pop_size
            self.stagnation_counter = 0  # Reset stagnation counter
            self.best_fitness_history = []
            print("Restarting population.")
            return True
        return False

    def local_search(self, func):
        best_idx = np.argmin(self.fitness)
        best_x = self.population[best_idx].copy()
        best_f = self.fitness[best_idx]
        success = False

        # Adaptive number of local search steps based on ls_success_rate
        num_steps = int(self.dim * (1 + self.ls_success_rate))  # Increase steps if successful

        for _ in range(num_steps): # Budget conscious local search
            if self.budget <= 0:
                break

            direction = np.random.uniform(-1, 1, size=self.dim)
            direction = direction / np.linalg.norm(direction)  # Normalize

            new_x = best_x + self.local_search_radius * direction
            new_x = self.handle_bounds(new_x[None, :], func)[0] # Handle bounds for single vector

            new_f = func(new_x)
            self.budget -= 1

            if new_f < best_f:
                self.population[best_idx] = new_x
                self.fitness[best_idx] = new_f
                best_x = new_x
                best_f = new_f
                success = True

        if self.ls_success_rate < 0.2 and self.local_search_radius > self.ls_radius_min:
            self.local_search_radius *= self.local_search_radius_decay  # Reduce radius if not successful

        return success

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

            # Adaptive local search application
            if np.random.rand() < self.ls_prob:
                success = self.local_search(func)
                self.ls_success_history.append(int(success))

                if len(self.ls_success_history) > self.ls_history_length:
                    self.ls_success_history.pop(0)

                self.ls_success_rate = np.mean(self.ls_success_history)
                # Adapt local search probability
                if self.ls_success_rate > 0.4:
                    self.ls_prob = min(1.0, self.ls_prob * 1.1)
                else:
                    self.ls_prob = max(0.01, self.ls_prob * 0.9)

            if self.check_stagnation():
                if not self.restart(func):
                    self.local_search_radius = 0.1 # Reset radius
                    self.ls_prob = self.ls_prob_initial # Reset ls_prob
        
        # Final local search at the end, if budget allows:
        if self.budget > 0:
            self.local_search(func)

        return self.f_opt, self.x_opt