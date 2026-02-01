import numpy as np

class AdaptiveDE:
    def __init__(self, budget=10000, dim=10, pop_size_min=20, pop_size_max=100, F_base=0.5, CR_base=0.7, archive_size=50, F_range=0.3, CR_range=0.3, orthogonal_learning_rate=0.1, restart_patience=50, archive_prob=0.1, current_to_best_prob=0.2, gamma_F = 0.1, gamma_CR = 0.1, success_threshold = 0.2, age_archive_rate = 0.01, F_CR_update_interval = 10, stagnation_fitness_threshold = 1e-6):
        self.budget = budget
        self.dim = dim
        self.pop_size_min = pop_size_min
        self.pop_size_max = pop_size_max
        self.pop_size = pop_size_max  # Initialize with maximum population size
        self.F_base = F_base
        self.CR_base = CR_base
        self.archive_size = archive_size
        self.F_range = F_range
        self.CR_range = CR_range
        self.success_archive = []
        self.failure_archive = []
        self.orthogonal_learning_rate = orthogonal_learning_rate
        self.restart_patience = restart_patience
        self.stagnation_counter = 0
        self.previous_best_fitness = np.Inf
        self.F_success_history = []
        self.CR_success_history = []
        self.archive_prob = archive_prob
        self.current_to_best_prob = current_to_best_prob
        self.gamma_F = gamma_F
        self.gamma_CR = gamma_CR
        self.success_threshold = success_threshold
        self.age_archive_rate = age_archive_rate
        self.F_CR_update_interval = F_CR_update_interval
        self.generation = 0
        self.stagnation_fitness_threshold = stagnation_fitness_threshold  # Threshold for fitness improvement

    def __call__(self, func):
        self.f_opt = np.Inf
        self.x_opt = None
        
        # Initialize population
        population = np.random.uniform(func.bounds.lb, func.bounds.ub, size=(self.pop_size, self.dim))
        fitness = np.array([func(x) for x in population])
        self.budget -= self.pop_size

        # Update optimal solution
        best_index = np.argmin(fitness)
        if fitness[best_index] < self.f_opt:
            self.f_opt = fitness[best_index]
            self.x_opt = population[best_index]
            self.stagnation_counter = 0
            self.previous_best_fitness = self.f_opt
        else:
            self.stagnation_counter += 1

        while self.budget > 0:
            self.generation += 1
            new_population = np.copy(population)
            new_fitness = np.copy(fitness)
            
            # Self-adaptive F and CR - update at intervals
            if self.generation % self.F_CR_update_interval == 0:
                if self.F_success_history:
                    self.F_base = (1-self.gamma_F) * self.F_base + self.gamma_F * np.mean(self.F_success_history)
                    self.F_base = np.clip(self.F_base, 0.1, 1.0)
                if self.CR_success_history:
                    self.CR_base = (1-self.gamma_CR) * self.CR_base + self.gamma_CR * np.mean(self.CR_success_history)
                    self.CR_base = np.clip(self.CR_base, 0.1, 1.0)

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
                if len(self.success_archive) > 0 and np.random.rand() < self.archive_prob:
                    archive_index = np.random.randint(len(self.success_archive))
                    mutant = x1 + F * (x2 - self.success_archive[archive_index])

                mutant = np.clip(mutant, func.bounds.lb, func.bounds.ub)

                # Crossover
                trial = np.copy(population[i])
                j_rand = np.random.randint(self.dim)
                for j in range(self.dim):
                    if np.random.rand() < CR or j == j_rand:
                        trial[j] = mutant[j]
                
                # Orthogonal Learning
                if np.random.rand() < self.orthogonal_learning_rate:
                    orthogonal_vector = self.x_opt + np.random.normal(0, 0.1, self.dim)
                    orthogonal_vector = np.clip(orthogonal_vector, func.bounds.lb, func.bounds.ub)
                    trial = 0.5 * (trial + orthogonal_vector)
                    trial = np.clip(trial, func.bounds.lb, func.bounds.ub)

                # Selection
                f_trial = func(trial)
                self.budget -= 1
                
                # Define success based on threshold relative to current fitness
                success = f_trial < fitness[i] - self.success_threshold * np.abs(fitness[i])
                
                if success:
                    new_population[i] = trial
                    new_fitness[i] = f_trial
                    
                    # Record successful F and CR values
                    self.F_success_history.append(F)
                    self.CR_success_history.append(CR)
                    if len(self.F_success_history) > 50:
                        self.F_success_history.pop(0)
                        self.CR_success_history.pop(0)
                        self.CR_success_history.pop(0)

                    # Archive Update Strategy: Fitness-based replacement
                    if len(self.success_archive) < self.archive_size:
                        self.success_archive.append(population[i])  # Add replaced individual
                    else:
                        # Replace worst element in archive if new one is better
                        archive_fitness = [func(x) for x in self.success_archive]
                        worst_index = np.argmax(archive_fitness)
                        if f_trial < archive_fitness[worst_index]:
                            self.success_archive[worst_index] = population[i]  # Replace worst with successful individual
                            
                    # Apply aging (perturb the archive element slightly) - do this AFTER archiving the original point
                    if len(self.success_archive) > 0:
                         archive_index = np.random.randint(len(self.success_archive))
                         self.success_archive[archive_index] += np.random.normal(0, self.age_archive_rate, self.dim)
                         self.success_archive[archive_index] = np.clip(self.success_archive[archive_index], func.bounds.lb, func.bounds.ub)

                    # Update optimal solution
                    if f_trial < self.f_opt:
                        self.f_opt = f_trial
                        self.x_opt = trial
                        self.stagnation_counter = 0
                        self.previous_best_fitness = self.f_opt
                else:
                    # Archive Update Strategy: Fitness-based replacement (failure archive)
                    if len(self.failure_archive) < self.archive_size:
                        self.failure_archive.append(trial)  # Add the failed trial
                    else:
                        # Replace worst element in archive if new one is worse
                        archive_fitness = [func(x) for x in self.failure_archive]
                        best_index = np.argmin(archive_fitness)
                        if f_trial > archive_fitness[best_index]:
                            self.failure_archive[best_index] = trial
                            
                    # Apply aging (perturb the archive element slightly) - do this AFTER archiving the original point
                    if len(self.failure_archive) > 0:
                         archive_index = np.random.randint(len(self.failure_archive))
                         self.failure_archive[archive_index] += np.random.normal(0, self.age_archive_rate, self.dim)
                         self.failure_archive[archive_index] = np.clip(self.failure_archive[archive_index], func.bounds.lb, func.bounds.ub)

            population = new_population
            fitness = new_fitness
            
            # Adaptive Population Size Adjustment
            if self.generation % 20 == 0:  # Adjust every 20 iterations
                if np.abs(self.f_opt - self.previous_best_fitness) < self.stagnation_fitness_threshold:
                    self.pop_size = max(self.pop_size_min, int(self.pop_size * 0.9))  # Reduce population size
                    print("Reducing Popsize")
                else:
                    self.pop_size = min(self.pop_size_max, int(self.pop_size * 1.1))  # Increase population size
                    print("Increasing Popsize")

                self.pop_size = np.clip(self.pop_size, self.pop_size_min, self.pop_size_max)
                self.previous_best_fitness = self.f_opt
                
                # Regenerate Population, keeping the best
                best_index = np.argmin(fitness)
                best_individual = population[best_index]
                
                population = np.random.uniform(func.bounds.lb, func.bounds.ub, size=(self.pop_size, self.dim))
                population[0] = best_individual # Ensure we keep the best
                
                fitness = np.array([func(x) for x in population])
                self.budget -= (self.pop_size -1 )

            # Restart population if stagnating using opposition-based learning
            if self.stagnation_counter > self.restart_patience:
                # Generate opposition population
                opposition_population = func.bounds.ub + func.bounds.lb - population
                opposition_fitness = np.array([func(x) for x in opposition_population])
                self.budget -= self.pop_size

                # Combine original and opposition populations
                combined_population = np.vstack((population, opposition_population))
                combined_fitness = np.concatenate((fitness, opposition_fitness))

                # Select the best individuals to form the new population
                best_indices = np.argsort(combined_fitness)[:self.pop_size]
                population = combined_population[best_indices]
                fitness = combined_fitness[best_indices]

                best_index = np.argmin(fitness)
                if fitness[best_index] < self.f_opt:
                    self.f_opt = fitness[best_index]
                    self.x_opt = population[best_index]
                    self.stagnation_counter = 0
                    self.previous_best_fitness = self.f_opt
                else:
                    self.stagnation_counter +=1
                    
                # Reset F and CR history
                self.F_success_history = []
                self.CR_success_history = []

            if self.budget <= 0:
                break
                
        return self.f_opt, self.x_opt