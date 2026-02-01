import numpy as np

class AdaptiveDE:
    def __init__(self, budget=10000, dim=10, pop_size=50, F_base=0.5, CR_base=0.7, archive_size=50, F_range=0.3, CR_range=0.3, orthogonal_learning_rate=0.1, restart_patience=50, archive_prob=0.1, current_to_best_prob=0.2, gamma_F = 0.1, gamma_CR = 0.1, historical_memory_size = 10):
        self.budget = budget
        self.dim = dim
        self.pop_size = pop_size
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
        self.F_failure_history = []
        self.CR_failure_history = []
        self.archive_prob = archive_prob
        self.current_to_best_prob = current_to_best_prob
        self.gamma_F = gamma_F
        self.gamma_CR = gamma_CR
        self.historical_memory_size = historical_memory_size
        self.historical_best_fitness = []
        self.historical_best_solutions = []


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

            self.historical_best_fitness.append(self.f_opt)
            self.historical_best_solutions.append(np.copy(self.x_opt))
            if len(self.historical_best_fitness) > self.historical_memory_size:
                self.historical_best_fitness.pop(0)
                self.historical_best_solutions.pop(0)

        else:
            self.stagnation_counter +=1

        generation = 0
        while self.budget > 0:
            generation += 1
            new_population = np.copy(population)
            new_fitness = np.copy(fitness)
            
            # Self-adaptive F and CR
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

                # Mutation strategy selection based on success/failure
                if len(self.F_success_history) > len(self.F_failure_history):
                    # Current-to-best mutation with archive
                    indices = np.random.choice(self.pop_size, 2, replace=False)
                    x1, x2 = population[indices]

                    if np.random.rand() < self.current_to_best_prob:
                        mutant = population[i] + F * (self.x_opt - population[i]) + F * (x1 - x2)
                    else:
                        x3 = population[np.random.choice(self.pop_size)]
                        mutant = x1 + F * (x2 - x3)
                    
                    # Incorporate archive
                    if len(self.success_archive) > 0 and np.random.rand() < self.archive_prob:  # archive_prob chance to use archive
                        archive_index = np.random.randint(len(self.success_archive))
                        mutant = x1 + F * (x2 - self.success_archive[archive_index])

                else:
                    # Random mutation with archive
                    indices = np.random.choice(self.pop_size, 3, replace=False)
                    x1, x2, x3 = population[indices]
                    mutant = x1 + F * (x2 - x3)

                    # Incorporate failure archive (exploration)
                    if len(self.failure_archive) > 0 and np.random.rand() < self.archive_prob:
                        archive_index = np.random.randint(len(self.failure_archive))
                        mutant = x1 + F * (x2 - self.failure_archive[archive_index])

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

                # Selection
                f_trial = func(trial)
                self.budget -= 1
                if f_trial < fitness[i]:
                    new_population[i] = trial
                    new_fitness[i] = f_trial
                    
                    # Record successful F and CR values
                    self.F_success_history.append(F)
                    self.CR_success_history.append(CR)
                    if len(self.F_success_history) > 50:
                        self.F_success_history.pop(0)
                        self.CR_success_history.pop(0)

                    # Add replaced vector to success archive
                    if len(self.success_archive) < self.archive_size:
                        self.success_archive.append(population[i])
                    else:
                        self.success_archive[np.random.randint(self.archive_size)] = population[i]

                    # Update optimal solution
                    if f_trial < self.f_opt:
                        self.f_opt = f_trial
                        self.x_opt = trial
                        self.stagnation_counter = 0
                        self.previous_best_fitness = self.f_opt

                        self.historical_best_fitness.append(self.f_opt)
                        self.historical_best_solutions.append(np.copy(self.x_opt))
                        if len(self.historical_best_fitness) > self.historical_memory_size:
                            self.historical_best_fitness.pop(0)
                            self.historical_best_solutions.pop(0)
                else:
                    self.F_failure_history.append(F)
                    self.CR_failure_history.append(CR)
                    if len(self.F_failure_history) > 50:
                         self.F_failure_history.pop(0)
                    if len(self.CR_failure_history) > 50:
                         self.CR_failure_history.pop(0)

                     # Add trial vector to failure archive
                    if len(self.failure_archive) < self.archive_size:
                        self.failure_archive.append(trial)
                    else:
                        self.failure_archive[np.random.randint(self.archive_size)] = trial

            population = new_population
            fitness = new_fitness
            
            # Restart population if stagnating using historical best solutions
            if self.stagnation_counter > self.restart_patience:
                # Incorporate historical best solutions into the population
                num_historical = min(len(self.historical_best_solutions), self.pop_size // 2)  # Use at most half the population
                
                if num_historical > 0:
                    indices = np.random.choice(len(self.historical_best_solutions), num_historical, replace=False)
                    population[:num_historical] = np.array(self.historical_best_solutions)[indices]
                    fitness[:num_historical] = np.array(self.historical_best_fitness)[indices]

                # Generate opposition population for the remaining individuals
                opposition_population = func.bounds.ub + func.bounds.lb - population[num_historical:]
                opposition_fitness = np.array([func(x) for x in opposition_population])
                self.budget -= (self.pop_size - num_historical)

                # Combine original and opposition populations
                combined_population = np.vstack((population[num_historical:], opposition_population))
                combined_fitness = np.concatenate((fitness[num_historical:], opposition_fitness))

                # Select the best individuals to form the new population
                best_indices = np.argsort(combined_fitness)[:(self.pop_size - num_historical)]
                population[num_historical:] = combined_population[best_indices]
                fitness[num_historical:] = combined_fitness[best_indices]


                best_index = np.argmin(fitness)
                if fitness[best_index] < self.f_opt:
                    self.f_opt = fitness[best_index]
                    self.x_opt = population[best_index]
                    self.stagnation_counter = 0
                    self.previous_best_fitness = self.f_opt

                    self.historical_best_fitness.append(self.f_opt)
                    self.historical_best_solutions.append(np.copy(self.x_opt))
                    if len(self.historical_best_fitness) > self.historical_memory_size:
                        self.historical_best_fitness.pop(0)
                        self.historical_best_solutions.pop(0)
                else:
                    self.stagnation_counter +=1
                    
                # Reset F and CR history
                self.F_success_history = []
                self.CR_success_history = []
                self.F_failure_history = []
                self.CR_failure_history = []
                

            if self.budget <= 0:
                break
                
        return self.f_opt, self.x_opt