import numpy as np

class AdaptiveDE:
    def __init__(self, budget=10000, dim=10, pop_size=50, F_base=0.5, CR_base=0.7, archive_size=50, F_range=0.3, CR_range=0.3, orthogonal_learning_rate=0.1, restart_patience=50, archive_prob=0.1, current_to_best_prob=0.2, gamma_F = 0.1, gamma_CR = 0.1, success_threshold = 0.2, mutation_strategy_probs = [0.3, 0.3, 0.4]):
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
        self.archive_prob = archive_prob
        self.current_to_best_prob = current_to_best_prob
        self.gamma_F = gamma_F
        self.gamma_CR = gamma_CR
        self.success_threshold = success_threshold # Threshold for considering a trial successful
        self.age_archive_rate = 0.01 # rate at which to apply "aging" in archive
        self.F_CR_update_interval = 10 # update F and CR every this many iterations
        self.generation = 0
        self.mutation_strategy_probs = mutation_strategy_probs # Probabilities for mutation strategies
        self.mutation_success_counts = [1] * len(mutation_strategy_probs) # Initialize success counts for each mutation strategy

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
            self.stagnation_counter +=1

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
                    
                # Normalize mutation strategy probabilities based on success counts
                total_success = sum(self.mutation_success_counts)
                self.mutation_strategy_probs = [count / total_success for count in self.mutation_success_counts]

            for i in range(self.pop_size):
                # Adaptive F and CR
                F = self.F_base + np.random.uniform(-self.F_range, self.F_range)
                CR = self.CR_base + np.random.uniform(-self.CR_range, self.CR_range)
                F = np.clip(F, 0.1, 1.0)
                CR = np.clip(CR, 0.1, 1.0)

                # Mutation strategy selection
                mutation_strategy = np.random.choice(len(self.mutation_strategy_probs), p=self.mutation_strategy_probs)
                
                # Mutation strategies
                if mutation_strategy == 0:  # Current-to-best
                    indices = np.random.choice(self.pop_size, 2, replace=False)
                    x1, x2 = population[indices]
                    mutant = population[i] + F * (self.x_opt - population[i]) + F * (x1 - x2)
                elif mutation_strategy == 1:  # Random mutation
                    x1, x2, x3 = population[np.random.choice(self.pop_size, 3, replace=False)]
                    mutant = x1 + F * (x2 - x3)
                else:  # Adding Gaussian perturbation to current best solution
                    mutant = self.x_opt + np.random.normal(0, 0.05, self.dim) # Reduced std for finer local search
                    
                # Incorporate archive
                if len(self.success_archive) > 0 and np.random.rand() < self.archive_prob:  # archive_prob chance to use archive
                    archive_index = np.random.randint(len(self.success_archive))
                    x1 = population[np.random.choice(self.pop_size)]
                    mutant = x1 + F * (mutant - self.success_archive[archive_index])

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

                    # Add replaced vector to success archive (with aging)
                    if len(self.success_archive) < self.archive_size:
                        self.success_archive.append(population[i])
                    else:
                        index_to_replace = np.random.randint(self.archive_size)
                        self.success_archive[index_to_replace] = population[i]

                        # Apply aging (perturb the archive element slightly)
                        self.success_archive[index_to_replace] += np.random.normal(0, self.age_archive_rate, self.dim)
                        self.success_archive[index_to_replace] = np.clip(self.success_archive[index_to_replace], func.bounds.lb, func.bounds.ub)

                    # Update optimal solution
                    if f_trial < self.f_opt:
                        self.f_opt = f_trial
                        self.x_opt = trial
                        self.stagnation_counter = 0
                        self.previous_best_fitness = self.f_opt
                        
                    self.mutation_success_counts[mutation_strategy] += 1
                else:
                    # Add trial vector to failure archive (with aging)
                    if len(self.failure_archive) < self.archive_size:
                        self.failure_archive.append(trial)
                    else:
                        index_to_replace = np.random.randint(self.archive_size)
                        self.failure_archive[index_to_replace] = trial
                        
                        # Apply aging (perturb the archive element slightly)
                        self.failure_archive[index_to_replace] += np.random.normal(0, self.age_archive_rate, self.dim)
                        self.failure_archive[index_to_replace] = np.clip(self.failure_archive[index_to_replace], func.bounds.lb, func.bounds.ub)
                    self.mutation_success_counts[mutation_strategy] = max(1, self.mutation_success_counts[mutation_strategy] - 0.1) # Reduce success count for unsuccessful strategy

            population = new_population
            fitness = new_fitness
            
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
                self.mutation_success_counts = [1] * len(self.mutation_success_counts) # Reset counts

            if self.budget <= 0:
                break
                
        return self.f_opt, self.x_opt