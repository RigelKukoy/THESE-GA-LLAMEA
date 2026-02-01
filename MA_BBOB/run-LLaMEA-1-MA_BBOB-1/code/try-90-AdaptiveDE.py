import numpy as np

class AdaptiveDE:
    def __init__(self, budget=10000, dim=10, pop_size=50, archive_size=50, orthogonal_learning_rate=0.1, restart_patience=50, archive_prob=0.1, pbest_proportion=0.1, gamma_F=0.1, gamma_CR=0.1, F_init=0.5, CR_init=0.7, archive_decay_rate=0.95, cauchy_mutation_prob=0.05):
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
        self.momentum_F = 0.0
        self.momentum_CR = 0.0
        self.cauchy_mutation_prob = cauchy_mutation_prob

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
            
            # Self-adaptive F and CR with Momentum
            if self.F_success_history:
                mean_F = np.mean(self.F_success_history)
                self.momentum_F = (0.9 * self.momentum_F) + (0.1 * mean_F)
                self.F_base = (1 - self.gamma_F) * self.F_base + self.gamma_F * self.momentum_F
                self.F_base = np.clip(self.F_base, 0.1, 1.0)

            if self.CR_success_history:
                mean_CR = np.mean(self.CR_success_history)
                self.momentum_CR = (0.9 * self.momentum_CR) + (0.1 * mean_CR)
                self.CR_base = (1 - self.gamma_CR) * self.CR_base + self.gamma_CR * self.momentum_CR
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
                    
                # Cauchy Mutation
                if np.random.rand() < self.cauchy_mutation_prob:
                    mutant += 0.01 * np.random.standard_cauchy(size=self.dim)  # Scale Cauchy distribution

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

                    # Add replaced vector to success archive, replacing worst solution
                    if len(self.success_archive) < self.archive_size:
                        self.success_archive.append(population[i])
                    else:
                        fitness_archive = [func(x) for x in self.success_archive]
                        worst_index = np.argmax(fitness_archive)
                        self.success_archive[worst_index] = population[i]

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
                        fitness_archive = [func(x) for x in self.failure_archive]
                        worst_index = np.argmax(fitness_archive)
                        self.failure_archive[worst_index] = trial

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
                self.momentum_F = 0.0
                self.momentum_CR = 0.0
                
                # Update bounds
                self.bounds_lb = new_lb
                self.bounds_ub = new_ub

            if self.budget <= 0:
                break
                
        return self.f_opt, self.x_opt