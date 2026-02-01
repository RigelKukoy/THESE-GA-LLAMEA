import numpy as np

class AdaptiveDE:
    def __init__(self, budget=10000, dim=10, pop_size_min=20, pop_size_max=100, F_base=0.5, CR_base=0.7, archive_size=100, F_range=0.3, CR_range=0.3, orthogonal_learning_rate=0.1, restart_patience=50, cauchy_mutation_prob=0.05, success_history_size=10):
        self.budget = budget
        self.dim = dim
        self.pop_size_min = pop_size_min
        self.pop_size_max = pop_size_max
        self.pop_size = pop_size_max  # Start with maximum population size
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
        self.success_history_size = success_history_size
        self.F_history = []
        self.CR_history = []
        self.success_count = 0  # Track successful generations for pop size adjustment
        self.cauchy_mutation_prob = cauchy_mutation_prob
        self.lb = -5.0
        self.ub = 5.0


    def toroidal_mutation(self, x1, x2, x3, F):
        mutant = x1 + F * (x2 - x3)
        # Toroidal handling of boundaries
        for i in range(self.dim):
            if mutant[i] < self.lb:
                mutant[i] = self.ub - (self.lb - mutant[i]) % (self.ub - self.lb)
            elif mutant[i] > self.ub:
                mutant[i] = self.lb + (mutant[i] - self.ub) % (self.ub - self.lb)
        return mutant


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
            
            # Self-adaptive F and CR (Success-History Based Adaptation)
            if self.F_history:
                self.F_base = np.mean(self.F_history)
            if self.CR_history:
                self.CR_base = np.mean(self.CR_history)

            for i in range(self.pop_size):
                # Adaptive F and CR
                F = self.F_base + np.random.uniform(-self.F_range, self.F_range)
                CR = self.CR_base + np.random.uniform(-self.CR_range, self.CR_range)
                F = np.clip(F, 0.1, 1.0)
                CR = np.clip(CR, 0.1, 1.0)

                # Mutation
                indices = np.random.choice(self.pop_size, 3, replace=False)
                x1, x2, x3 = population[indices]
                
                # Apply Cauchy mutation with a certain probability
                if np.random.rand() < self.cauchy_mutation_prob:
                    mutant = self.toroidal_mutation(x1, x2, x3, F) + 0.1 * np.random.standard_cauchy(size=self.dim)  # Cauchy mutation
                    mutant = np.clip(mutant, func.bounds.lb, func.bounds.ub)
                else:
                    mutant = self.toroidal_mutation(x1, x2, x3, F)

                # Incorporate archive
                if len(self.archive) > 0 and np.random.rand() < 0.1:  # 10% chance to use archive
                    archive_index = np.random.randint(len(self.archive))
                    mutant = self.toroidal_mutation(x1, x2, self.archive[archive_index], F)
                    

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
                    if len(self.F_history) > self.success_history_size:
                        self.F_history.pop(0)
                        self.CR_history.pop(0)

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
                        self.success_count += 1
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

            # Dynamic population size adjustment
            if self.success_count > 0.2 * self.pop_size: # Increase pop size
                self.pop_size = min(self.pop_size + 1, self.pop_size_max)
                self.success_count = 0
            elif self.stagnation_counter > self.restart_patience/2: # Decrease pop size
                 self.pop_size = max(self.pop_size - 1, self.pop_size_min)
                 self.stagnation_counter = 0
            
            # Restart population if stagnating
            if self.stagnation_counter > self.restart_patience:
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
                

            if self.budget <= 0:
                break
                
        return self.f_opt, self.x_opt