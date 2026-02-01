import numpy as np

class AdaptiveDE:
    def __init__(self, budget=10000, dim=10, pop_size=50, archive_size=50, orthogonal_learning_rate=0.1, restart_patience=50, archive_prob=0.1, pbest_proportion=0.1, gamma_F=0.1, gamma_CR=0.1, F_init=0.5, CR_init=0.7, archive_decay_rate=0.95, local_search_probability=0.1):
        self.budget = budget
        self.dim = dim
        self.pop_size = pop_size
        self.archive_size = archive_size
        self.orthogonal_learning_rate = orthogonal_learning_rate
        self.restart_patience = restart_patience
        self.stagnation_counter = 0
        self.previous_best_fitness = np.Inf
        self.F_success_history = []
        self.CR_success_history = []
        self.archive_prob = archive_prob
        self.pbest_proportion = pbest_proportion
        self.gamma_F = gamma_F
        self.gamma_CR = gamma_CR
        self.F_base = F_init
        self.CR_base = CR_init
        self.success_archive = []
        self.failure_archive = []
        self.archive_decay_rate = archive_decay_rate
        self.bounds_lb = None
        self.bounds_ub = None
        self.local_search_probability = local_search_probability

    def __call__(self, func):
        self.f_opt = np.Inf
        self.x_opt = None
        self.bounds_lb = func.bounds.lb
        self.bounds_ub = func.bounds.ub
        
        # Initialize population
        population = np.random.uniform(self.bounds_lb, self.bounds_ub, size=(self.pop_size, self.dim))
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

        generation = 0
        while self.budget > 0:
            generation += 1
            new_population = np.copy(population)
            new_fitness = np.copy(fitness)
            
            # Self-adaptive F and CR
            if self.F_success_history:
                self.F_base = (1 - self.gamma_F) * self.F_base + self.gamma_F * np.mean(self.F_success_history)
                self.F_base = np.clip(self.F_base, 0.1, 1.0)
            if self.CR_success_history:
                self.CR_base = (1 - self.gamma_CR) * self.CR_base + self.gamma_CR * np.mean(self.CR_success_history)
                self.CR_base = np.clip(self.CR_base, 0.1, 1.0)

            for i in range(self.pop_size):
                # Adaptive F and CR: Sample from log-normal distribution
                F = np.exp(np.random.normal(np.log(self.F_base), 0.1))
                CR = np.exp(np.random.normal(np.log(self.CR_base), 0.1))
                F = np.clip(F, 0.1, 1.0)
                CR = np.clip(CR, 0.1, 1.0)

                # Mutation: current-to-pbest
                pbest_count = max(1, int(self.pbest_proportion * self.pop_size))
                pbest_indices = np.argsort(fitness)[:pbest_count]
                pbest = population[np.random.choice(pbest_indices)]
                
                indices = np.random.choice(self.pop_size, 2, replace=False)
                x1, x2 = population[indices]

                mutant = population[i] + F * (pbest - population[i]) + F * (x1 - x2)
                
                # Incorporate archive
                if len(self.success_archive) > 0 and np.random.rand() < self.archive_prob:  # archive_prob chance to use archive
                    archive_index = np.random.randint(len(self.success_archive))
                    mutant = x1 + F * (x2 - self.success_archive[archive_index])

                mutant = np.clip(mutant, self.bounds_lb, self.bounds_ub)

                # Crossover
                trial = np.copy(population[i])
                j_rand = np.random.randint(self.dim)
                for j in range(self.dim):
                    if np.random.rand() < CR or j == j_rand:
                        trial[j] = mutant[j]
                
                # Orthogonal Learning
                if np.random.rand() < self.orthogonal_learning_rate:
                    orthogonal_vector = self.x_opt + np.random.normal(0, 0.1, self.dim)  # Perturb best solution
                    orthogonal_vector = np.clip(orthogonal_vector, self.bounds_lb, self.bounds_ub)
                    trial = 0.5 * (trial + orthogonal_vector)  # Combine with trial vector
                    trial = np.clip(trial, self.bounds_lb, self.bounds_ub)

                # Local Search
                if np.random.rand() < self.local_search_probability:
                    step_size = 0.01 * (self.bounds_ub - self.bounds_lb)  # Small step size
                    for j in range(self.dim):
                        # Try small steps in both directions
                        trial_plus = np.copy(trial)
                        trial_minus = np.copy(trial)
                        trial_plus[j] = np.clip(trial[j] + step_size[j], self.bounds_lb[j], self.bounds_ub[j])
                        trial_minus[j] = np.clip(trial[j] - step_size[j], self.bounds_lb[j], self.bounds_ub[j])

                        f_plus = func(trial_plus) if self.budget > 0 else np.inf
                        self.budget -= 1 if self.budget > 0 else 0
                        f_minus = func(trial_minus) if self.budget > 0 else np.inf
                        self.budget -= 1 if self.budget > 0 else 0
                        
                        if f_plus < func(trial) and f_plus < f_minus:
                            trial = trial_plus
                        elif f_minus < func(trial) and f_minus < f_plus:
                            trial = trial_minus


                # Selection
                f_trial = func(trial) if self.budget > 0 else np.inf
                self.budget -= 1 if self.budget > 0 else 0
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
                        self.success_archive[np.random.randint(len(self.success_archive))] = population[i]


                    # Update optimal solution
                    if f_trial < self.f_opt:
                        self.f_opt = f_trial
                        self.x_opt = trial
                        self.stagnation_counter = 0
                        self.previous_best_fitness = self.f_opt
                else:
                     # Add trial vector to failure archive
                    if len(self.failure_archive) < self.archive_size:
                        self.failure_archive.append(trial)
                    else:
                        self.failure_archive[np.random.randint(len(self.failure_archive))] = trial

            population = new_population
            fitness = new_fitness

            # Decay archive to promote diversity and remove potentially bad solutions
            self.success_archive = [self.success_archive[i] for i in range(len(self.success_archive)) if np.random.rand() < self.archive_decay_rate]
            self.failure_archive = [self.failure_archive[i] for i in range(len(self.failure_archive)) if np.random.rand() < self.archive_decay_rate]
            
            # Restart population if stagnating using opposition-based learning and shrinking bounds
            if self.stagnation_counter > self.restart_patience:
                # Generate opposition population within shrunken bounds
                center = self.x_opt
                bound_range = (self.bounds_ub - self.bounds_lb) * 0.5  # Shrink by half
                new_lb = np.maximum(self.bounds_lb, center - bound_range / 2)
                new_ub = np.minimum(self.bounds_ub, center + bound_range / 2)

                opposition_population = np.random.uniform(new_lb, new_ub, size=(self.pop_size, self.dim))
                opposition_fitness = np.array([func(x) for x in opposition_population]) if self.budget > self.pop_size else np.array([np.inf]*self.pop_size)
                self.budget -= self.pop_size if self.budget > self.pop_size else self.budget

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
                
                # Update bounds
                self.bounds_lb = new_lb
                self.bounds_ub = new_ub

            if self.budget <= 0:
                break
                
        return self.f_opt, self.x_opt