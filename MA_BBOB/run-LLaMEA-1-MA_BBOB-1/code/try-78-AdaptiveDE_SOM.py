import numpy as np
from minisom import MiniSom

class AdaptiveDE_SOM:
    def __init__(self, budget=10000, dim=10, pop_size_min=20, pop_size_max=100, F_base=0.5, CR_base=0.7, archive_size=100, F_range=0.3, CR_range=0.3, orthogonal_learning_rate=0.1, restart_patience=50, archive_prob=0.1, current_to_best_prob=0.2, archive_success_threshold=0.1, aging_factor=0.95, som_grid_size=10, som_sigma=0.3, som_learning_rate=0.5, historical_memory_size=10):
        self.budget = budget
        self.dim = dim
        self.pop_size_min = pop_size_min
        self.pop_size_max = pop_size_max
        self.pop_size = pop_size_max  # Start with max population
        self.F_base = F_base
        self.CR_base = CR_base
        self.archive_size = archive_size
        self.F_range = F_range
        self.CR_range = CR_range
        self.archive = []
        self.archive_fitness = []
        self.orthogonal_learning_rate = orthogonal_learning_rate
        self.restart_patience = restart_patience
        self.stagnation_counter = 0
        self.previous_best_fitness = np.Inf
        self.F_history = []
        self.CR_history = []
        self.archive_prob = archive_prob
        self.current_to_best_prob = current_to_best_prob
        self.archive_success_rate = 0.0
        self.archive_successes = 0
        self.archive_trials = 0
        self.archive_success_threshold = archive_success_threshold
        self.aging_factor = aging_factor
        self.som_grid_size = som_grid_size
        self.som_sigma = som_sigma
        self.som_learning_rate = som_learning_rate
        self.som = MiniSom(self.som_grid_size, self.som_grid_size, self.dim, sigma=self.som_sigma, learning_rate=self.som_learning_rate)
        self.historical_memory_size = historical_memory_size
        self.successful_F = []
        self.successful_CR = []


    def __call__(self, func):
        self.f_opt = np.Inf
        self.x_opt = None

        # Initialize population
        population = np.random.uniform(func.bounds.lb, func.bounds.ub, size=(self.pop_size, self.dim))
        fitness = np.array([func(x) for x in population])
        self.budget -= self.pop_size

        # Train SOM initially
        self.som.train_random(population, 100)

        # Update optimal solution
        best_index = np.argmin(fitness)
        if fitness[best_index] < self.f_opt:
            self.f_opt = fitness[best_index]
            self.x_opt = population[best_index]
            self.stagnation_counter = 0
            self.previous_best_fitness = self.f_opt
        else:
            self.stagnation_counter += 1

        generation = 0
        while self.budget > 0:
            generation += 1
            new_population = np.copy(population)
            new_fitness = np.copy(fitness)

            # Self-adaptive F and CR
            if self.successful_F:
                weights = np.arange(len(self.successful_F), 0, -1)
                weights = weights / np.sum(weights)
                self.F_base = np.sum(weights * np.array(self.successful_F))
            if self.successful_CR:
                weights = np.arange(len(self.successful_CR), 0, -1)
                weights = weights / np.sum(weights)
                self.CR_base = np.sum(weights * np.array(self.successful_CR))

            for i in range(self.pop_size):
                # Adaptive F and CR
                F = self.F_base + np.random.uniform(-self.F_range, self.F_range)
                CR = self.CR_base + np.random.uniform(-self.CR_range, self.CR_range)
                F = np.clip(F, 0.1, 1.0)
                CR = np.clip(CR, 0.1, 1.0)

                # Mutation: Combining current-to-best with random mutation
                indices = np.random.choice(self.pop_size, 2, replace=False)
                x1, x2 = population[indices]

                if np.random.rand() < self.current_to_best_prob:
                    mutant = population[i] + F * (self.x_opt - population[i]) + F * (x1 - x2)
                else:
                    x3 = population[np.random.choice(self.pop_size)]
                    mutant = x1 + F * (x2 - x3)

                # Incorporate archive
                use_archive = False
                if len(self.archive) > 0 and np.random.rand() < self.archive_prob:  # archive_prob chance to use archive
                    archive_index = np.random.randint(len(self.archive))
                    mutant = x1 + F * (x2 - self.archive[archive_index])
                    use_archive = True

                mutant = np.clip(mutant, func.bounds.lb, func.bounds.ub)

                # Crossover
                trial = np.copy(population[i])
                j_rand = np.random.randint(self.dim)
                for j in range(self.dim):
                    if np.random.rand() < CR or j == j_rand:
                        trial[j] = mutant[j]

                # Orthogonal Learning
                if np.random.rand() < self.orthogonal_learning_rate:
                    orthogonal_vector = self.x_opt + np.random.normal(0, 0.1, self.dim)  # Perturb best solution
                    orthogonal_vector = np.clip(orthogonal_vector, func.bounds.lb, func.bounds.ub)
                    trial = 0.5 * (trial + orthogonal_vector)  # Combine with trial vector
                    trial = np.clip(trial, func.bounds.lb, func.bounds.ub)

                # Quadratic approximation for local search
                if np.random.rand() < 0.1:
                    # Perturb trial solution slightly
                    trial_perturbed = trial + np.random.normal(0, 0.01, self.dim)
                    trial_perturbed = np.clip(trial_perturbed, func.bounds.lb, func.bounds.ub)
                    f_trial_perturbed = func(trial_perturbed)
                    self.budget -= 1
                    if f_trial_perturbed < func(trial):
                         trial = trial_perturbed

                # Selection
                f_trial = func(trial)
                self.budget -= 1
                if f_trial < fitness[i]:
                    new_population[i] = trial
                    new_fitness[i] = f_trial

                    # Record successful F and CR values
                    self.successful_F.append(F)
                    self.successful_CR.append(CR)
                    if len(self.successful_F) > self.historical_memory_size:
                        self.successful_F.pop(0)
                        self.successful_CR.pop(0)

                    # Add replaced vector to archive (combined strategy)
                    if len(self.archive) < self.archive_size:
                        self.archive.append(population[i])
                        self.archive_fitness.append(fitness[i])
                    else:
                        # Replace worst in archive
                        worst_archive_index = np.argmax(self.archive_fitness)
                        self.archive[worst_archive_index] = population[i]
                        self.archive_fitness[worst_archive_index] = fitness[i]

                    # Update optimal solution
                    if f_trial < self.f_opt:
                        self.f_opt = f_trial
                        self.x_opt = trial
                        self.stagnation_counter = 0
                        self.previous_best_fitness = self.f_opt

                    if use_archive:
                        self.archive_successes += 1

                else:
                    # Add trial vector to archive (combined strategy)
                    if len(self.archive) < self.archive_size:
                        self.archive.append(trial)
                        self.archive_fitness.append(f_trial)
                    else:
                        # Replace worst in archive
                        worst_archive_index = np.argmax(self.archive_fitness)
                        self.archive[worst_archive_index] = trial
                        self.archive_fitness[worst_archive_index] = f_trial

                if use_archive:
                    self.archive_trials += 1

            population = new_population
            fitness = new_fitness

            # Restart population if stagnating, using SOM for stagnation detection
            if self.stagnation_counter > self.restart_patience:
                # Train SOM with current population
                self.som.train_random(population, 100)

                # Get quantisation error for each individual
                quantization_errors = [self.som.quantization_error(x) for x in population]
                
                # If mean quantization error is low, consider the population stagnant
                if np.mean(quantization_errors) < 0.01: # Adjust threshold as needed
                    # Opposition-based learning for restart
                    opposition_population = func.bounds.ub + func.bounds.lb - population
                    opposition_fitness = np.array([func(x) for x in opposition_population])
                    self.budget -= self.pop_size

                    combined_population = np.concatenate((population, opposition_population))
                    combined_fitness = np.concatenate((fitness, opposition_fitness))

                    # Select the best individuals from the combined population
                    indices = np.argsort(combined_fitness)[:self.pop_size]
                    population = combined_population[indices]
                    fitness = combined_fitness[indices]

                    best_index = np.argmin(fitness)
                    if fitness[best_index] < self.f_opt:
                        self.f_opt = fitness[best_index]
                        self.x_opt = population[best_index]
                        self.stagnation_counter = 0
                        self.previous_best_fitness = self.f_opt
                    else:
                        self.stagnation_counter += 1

                    # Reset F and CR history
                    self.successful_F = []
                    self.successful_CR = []

            # Adjust archive probability based on success rate
            if self.archive_trials > 10:
                self.archive_success_rate = self.archive_successes / self.archive_trials
                if self.archive_success_rate < self.archive_success_threshold:
                    self.archive_prob *= 0.9  # Decrease probability if success rate is low
                else:
                    self.archive_prob = min(1.0, self.archive_prob * 1.1)  # Increase if success is good
                self.archive_trials = 0
                self.archive_successes = 0

            # Dynamic population size adjustment
            if self.f_opt < self.previous_best_fitness:
                self.pop_size = min(self.pop_size_max, self.pop_size + 1)  # Increase pop size if improvement
                self.previous_best_fitness = self.f_opt
            else:
                self.pop_size = max(self.pop_size_min, self.pop_size - 1)  # Decrease pop size if no improvement

            # Age the archive
            for k in range(len(self.archive_fitness)):
                self.archive_fitness[k] *= self.aging_factor

            if self.budget <= 0:
                break

        return self.f_opt, self.x_opt