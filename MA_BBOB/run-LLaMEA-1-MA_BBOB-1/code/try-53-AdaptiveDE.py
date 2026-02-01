import numpy as np

class AdaptiveDE:
    def __init__(self, budget=10000, dim=10, pop_size=50, F_base=0.5, CR_base=0.7, archive_size=100, F_range=0.3, CR_range=0.3, orthogonal_learning_rate=0.1, restart_patience=50, aging_rate=0.05, archive_decay_rate=0.99, mutation_strategy="rand1"):
        self.budget = budget
        self.dim = dim
        self.pop_size = pop_size
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
        self.CR_history = []
        self.aging_rate = aging_rate
        self.archive_decay_rate = archive_decay_rate
        self.mutation_strategy = mutation_strategy
        self.F_covariance = np.eye(1) * 0.1  # Initial covariance for F adaptation
        self.CR_covariance = np.eye(1) * 0.1  # Initial covariance for CR adaptation

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

        generation = 0
        while self.budget > 0:
            generation += 1
            new_population = np.copy(population)
            new_fitness = np.copy(fitness)
            
            # Adaptive F and CR parameter control with covariance learning
            # Sample F and CR from Gaussian distributions with learned covariance
            F = np.random.multivariate_normal([self.F_base], self.F_covariance)[0]
            CR = np.random.multivariate_normal([self.CR_base], self.CR_covariance)[0]
            F = np.clip(F, 0.1, 1.0)
            CR = np.clip(CR, 0.1, 1.0)


            for i in range(self.pop_size):
                # Mutation
                if self.mutation_strategy == "rand1":
                    indices = np.random.choice(self.pop_size, 3, replace=False)
                    x1, x2, x3 = population[indices]
                    mutant = x1 + F * (x2 - x3)

                    # Incorporate archive
                    if len(self.archive) > 0 and np.random.rand() < 0.1:  # 10% chance to use archive
                        archive_index = np.random.randint(len(self.archive))
                        mutant = x1 + F * (x2 - self.archive[archive_index])

                elif self.mutation_strategy == "current_to_best":
                     indices = np.random.choice(self.pop_size, 2, replace=False)
                     x1, x2 = population[indices]
                     mutant = population[i] + F * (self.x_opt - population[i]) + F*(x1-x2)

                elif self.mutation_strategy == "rand2":
                    indices = np.random.choice(self.pop_size, 5, replace=False)
                    x1, x2, x3, x4, x5 = population[indices]
                    mutant = x1 + F * (x2 - x3) + F*(x4 - x5)
                else: # best2
                    indices = np.random.choice(self.pop_size, 4, replace=False)
                    x1, x2, x3, x4 = population[indices]
                    mutant = self.x_opt + F * (x1 - x2) + F*(x3-x4)


                # Repair mechanism
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
                    self.F_history.append(F)
                    self.CR_history.append(CR)
                    if len(self.F_history) > 50:
                        self.F_history.pop(0)
                        self.CR_history.pop(0)

                    # Update covariance matrices based on successful F and CR
                    self.F_covariance = np.cov(np.array(self.F_history).flatten(), rowvar=False) + np.eye(1)*0.001 # Add small value for stability
                    self.CR_covariance = np.cov(np.array(self.CR_history).flatten(), rowvar=False) + np.eye(1)*0.001

                    # Dynamic Archive management (fitness and diversity)
                    if len(self.archive) < self.archive_size:
                        self.archive.append(population[i])
                        self.archive_fitness.append(fitness[i])
                    else:
                        # Replace vector with worst fitness or closest to existing archive members
                        if np.random.rand() < 0.5:
                            #Replace based on fitness
                            worst_archive_index = np.argmax(self.archive_fitness)
                            self.archive[worst_archive_index] = population[i]
                            self.archive_fitness[worst_archive_index] = fitness[i]
                        else:
                            #Replace based on diversity
                            distances = [np.linalg.norm(population[i] - archive_member) for archive_member in self.archive]
                            closest_archive_index = np.argmin(distances)
                            self.archive[closest_archive_index] = population[i]
                            self.archive_fitness[closest_archive_index] = fitness[i]

                    # Update optimal solution
                    if f_trial < self.f_opt:
                        self.f_opt = f_trial
                        self.x_opt = trial
                        self.stagnation_counter = 0
                        self.previous_best_fitness = self.f_opt
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
                        

            population = new_population
            fitness = new_fitness
            
            # Multi-faceted Restart mechanism
            if self.stagnation_counter > self.restart_patience:
                if np.random.rand() < 0.3:
                    #Restart from best
                    population = self.x_opt + np.random.normal(0, 0.1, size=(self.pop_size, self.dim))
                    population = np.clip(population, func.bounds.lb, func.bounds.ub)

                elif np.random.rand() < 0.6:
                    # Restart using Latin Hypercube Sampling
                     population = np.random.uniform(0, 1, size=(self.pop_size, self.dim))
                     for j in range(self.dim):
                        idx = np.random.permutation(self.pop_size)
                        population[:, j] = (idx + population[:, j]) / self.pop_size
                        population[:,j] = func.bounds.lb[j] + population[:,j] * (func.bounds.ub[j] - func.bounds.lb[j])


                else:

                    # Restart with random population
                    population = np.random.uniform(func.bounds.lb, func.bounds.ub, size=(self.pop_size, self.dim))

                fitness = np.array([func(x) for x in population])
                self.budget -= self.pop_size
                best_index = np.argmin(fitness)
                if fitness[best_index] < self.f_opt:
                    self.f_opt = fitness[best_index]
                    self.x_opt = population[best_index]
                    self.stagnation_counter = 0
                    self.previous_best_fitness = self.f_opt
                else:
                    self.stagnation_counter +=1
                    
                # Reset F and CR history
                self.F_history = []
                self.CR_history = []
                
            # Archive aging
            for k in range(len(self.archive_fitness)):
                self.archive_fitness[k] *= self.archive_decay_rate # Gradual decay

            if self.budget <= 0:
                break
                
        return self.f_opt, self.x_opt