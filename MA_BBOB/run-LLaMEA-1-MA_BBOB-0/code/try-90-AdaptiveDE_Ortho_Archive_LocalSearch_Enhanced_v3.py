import numpy as np

class AdaptiveDE_Ortho_Archive_LocalSearch_Enhanced_v3:
    def __init__(self, budget=10000, dim=10, pop_size_min=20, pop_size_max=100, archive_size=10, F_initial=0.5, CR_initial=0.5, F_learning_rate=0.2, CR_learning_rate=0.2, stagnation_threshold=1000, ortho_group_size=5, archive_update_frequency=5, local_search_frequency=10, local_search_radius=0.1, local_search_radius_decay=0.95, restart_probability=0.05, restart_fitness_variance_threshold=1e-6, angle_threshold=0.9, cauchy_mutation_prob=0.1):
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
        self.F_EMA = F_initial # Exponential Moving Average for F
        self.CR_EMA = CR_initial # Exponential Moving Average for CR
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
        self.restart_fitness_variance_threshold = restart_fitness_variance_threshold
        self.budget_consumed_threshold = 0.75 # Restart if budget consumed above threshold.
        self.best_fitness_since_restart = np.inf
        self.angle_threshold = angle_threshold # Cosine of angle above which restart is considered
        self.previous_step = None
        self.cauchy_mutation_prob = cauchy_mutation_prob


    def initialize_population(self, func):
        self.population = np.random.uniform(func.bounds.lb, func.bounds.ub, size=(self.pop_size, self.dim))
        self.fitness = np.array([func(x) for x in self.population])
        self.budget -= self.pop_size
        self.archive = []
        self.best_fitness_since_restart = np.min(self.fitness)
        self.previous_step = None

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
                
            if np.random.rand() < self.cauchy_mutation_prob: # Cauchy Mutation
                scale = 0.1 * (func.bounds.ub - func.bounds.lb)
                cauchy_values = np.random.standard_cauchy(size=self.dim) * scale
                mutated_population[i] += cauchy_values

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

        # Update F and CR using EMA
        if self.successful_F:
            mean_F = np.mean(self.successful_F)
            mean_CR = np.mean(self.successful_CR)
            
            self.F_EMA = (1 - self.F_learning_rate) * self.F_EMA + self.F_learning_rate * mean_F
            self.CR_EMA = (1 - self.CR_learning_rate) * self.CR_EMA + self.CR_learning_rate * mean_CR
            
            self.F = self.F_EMA
            self.CR = self.CR_EMA
            

        self.successful_F = []
        self.successful_CR = []

        # Population size adaptation
        improvement_ratio = improved_count / self.pop_size
        if improvement_ratio > 0.3 and self.pop_size < self.pop_size_max:
            self.pop_size = min(self.pop_size + int(self.pop_size_adaptation_rate * (self.pop_size_max - self.pop_size)), self.pop_size_max)
        elif improvement_ratio < 0.1 and self.pop_size > self.pop_size_min:
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
        # Restart if fitness variance is low or budget consumption is high, or angle between steps is high, but only if there's been no recent improvement.
        budget_consumed = 1 - (self.budget / 10000)
        fitness_variance_low = len(self.best_fitness_history) >= self.stagnation_threshold and np.std(self.best_fitness_history[-self.stagnation_threshold:]) < self.restart_fitness_variance_threshold
        budget_exceeded_threshold = budget_consumed > self.budget_consumed_threshold

        no_recent_improvement = self.f_opt >= self.best_fitness_since_restart

        # Check angle between previous and current steps
        if self.previous_step is not None:
            current_step = self.x_opt - self.population[np.argmin(self.fitness)]
            cos_angle = np.dot(self.previous_step, current_step) / (np.linalg.norm(self.previous_step) * np.linalg.norm(current_step) + 1e-8)  # Adding small value to avoid division by zero
            similar_direction = cos_angle > self.angle_threshold
        else:
            similar_direction = False
            cos_angle = 0.0

        if (fitness_variance_low or budget_exceeded_threshold or similar_direction) and no_recent_improvement and np.random.rand() < self.restart_probability:
            self.population = np.random.uniform(func.bounds.lb, func.bounds.ub, size=(self.pop_size, self.dim))
            self.fitness = np.array([func(x) for x in self.population])
            self.budget -= self.pop_size
            self.stagnation_counter = 0  # Reset stagnation counter
            self.best_fitness_history = []
            self.best_fitness_since_restart = np.min(self.fitness)
            self.previous_step = None

            print(f"Restarting population. Budget consumed: {budget_consumed:.2f}, Fitness Variance: {np.std(self.best_fitness_history[-self.stagnation_threshold:]) if len(self.best_fitness_history) >= self.stagnation_threshold else 'N/A'}, Cosine Angle: {cos_angle:.2f}")

            return True
        return False

    def local_search(self, func):
        best_idx = np.argmin(self.fitness)
        best_x = self.population[best_idx].copy()
        best_f = self.fitness[best_idx]
        original_radius = self.local_search_radius

        # Aggressive local search: adapt radius based on success
        success_count = 0
        num_iterations = self.dim * 4 #Increased iterations
        
        step_size = self.local_search_radius / 5.0 # Introduce a step size
        
        for _ in range(num_iterations):
            if self.budget <= 0:
                break

            direction = np.random.uniform(-1, 1, size=self.dim)
            direction = direction / np.linalg.norm(direction)  # Normalize

            new_x = best_x + step_size * direction  # Use step size
            new_x = self.handle_bounds(new_x[None, :], func)[0] # Handle bounds for single vector

            new_f = func(new_x)
            self.budget -= 1

            if new_f < best_f:
                self.population[best_idx] = new_x
                self.fitness[best_idx] = new_f
                best_x = new_x
                best_f = new_f
                success_count += 1
                self.local_search_radius *= 1.05  # Slightly increase radius if successful
                step_size *= 1.05 # Increase step size too

            else:
                self.local_search_radius *= 0.95 # Reduce radius if unsuccessful
                step_size *= 0.95 # Reduce step size too

            self.local_search_radius = min(self.local_search_radius, original_radius * 2) # Radius cap
            step_size = min(step_size, self.local_search_radius / 5.0) # Step size cap

        if success_count == 0:
            self.local_search_radius = original_radius * self.local_search_radius_decay
        else:
            self.local_search_radius = original_radius * (1 + (success_count / num_iterations) * (1 - self.local_search_radius_decay))
            self.local_search_radius = min(self.local_search_radius, 0.1)


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
            current_best_x = self.population[np.argmin(self.fitness)]
            if current_best_fitness < self.f_opt:
                self.previous_step = self.x_opt - current_best_x # Save step
                self.f_opt = current_best_fitness
                self.x_opt = current_best_x
                self.best_fitness_since_restart = self.f_opt # Update best fitness since last restart

            self.best_fitness_history.append(current_best_fitness)

            self.generation += 1
            if self.generation % self.archive_update_frequency == 0:
                self.update_archive()

            if self.generation % self.local_search_frequency == 0:
                self.local_search(func)

            if self.check_stagnation():
                if not self.restart(func):
                    self.local_search_radius = 0.1 # Reset radius
        
        # Final local search at the end, if budget allows:
        if self.budget > 0:
            self.local_search(func)

        return self.f_opt, self.x_opt