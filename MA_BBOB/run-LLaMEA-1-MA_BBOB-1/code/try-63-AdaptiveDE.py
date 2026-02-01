import numpy as np

class AdaptiveDE:
    def __init__(self, budget=10000, dim=10, pop_size=50, F_base=0.5, CR_base=0.7, archive_size=50, diversity_archive_size=50, F_range=0.3, CR_range=0.3, orthogonal_learning_rate=0.1, restart_patience=50, aging_rate=0.05, success_threshold=0.01):
        self.budget = budget
        self.dim = dim
        self.pop_size = pop_size
        self.F_base = F_base
        self.CR_base = CR_base
        self.archive_size = archive_size
        self.diversity_archive_size = diversity_archive_size
        self.F_range = F_range
        self.CR_range = CR_range
        self.success_archive = []
        self.success_archive_fitness = []
        self.diversity_archive = []
        self.diversity_archive_fitness = []
        self.orthogonal_learning_rate = orthogonal_learning_rate
        self.restart_patience = restart_patience
        self.stagnation_counter = 0
        self.previous_best_fitness = np.Inf
        self.F_history = []
        self.CR_history = []
        self.aging_rate = aging_rate # Rate at which archive fitness degrades
        self.success_threshold = success_threshold  # Threshold for considering a solution "successful"


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
            
            # Adaptive F and CR parameter control with trend adjustment
            F_trend = 0
            CR_trend = 0
            if len(self.F_history) > 1:
                F_trend = np.mean(np.diff(self.F_history))
            if len(self.CR_history) > 1:
                CR_trend = np.mean(np.diff(self.CR_history))

            self.F_base = np.clip(self.F_base + 0.1 * F_trend, 0.1, 0.9)
            self.CR_base = np.clip(self.CR_base + 0.1 * CR_trend, 0.1, 0.9)


            for i in range(self.pop_size):
                # Adaptive F and CR
                F = self.F_base + np.random.uniform(-self.F_range, self.F_range)
                CR = self.CR_base + np.random.uniform(-self.CR_range, self.CR_range)
                F = np.clip(F, 0.1, 1.0)
                CR = np.clip(CR, 0.1, 1.0)

                # Mutation
                indices = np.random.choice(self.pop_size, 3, replace=False)
                x1, x2, x3 = population[indices]
                mutant = x1 + F * (x2 - x3)
                
                # Incorporate archives
                if len(self.success_archive) > 0 and np.random.rand() < 0.05:  # 5% chance to use success archive
                    archive_index = np.random.randint(len(self.success_archive))
                    mutant = x1 + F * (x2 - self.success_archive[archive_index])

                if len(self.diversity_archive) > 0 and np.random.rand() < 0.05:  # 5% chance to use diversity archive
                    archive_index = np.random.randint(len(self.diversity_archive))
                    mutant = x1 + F * (x1 - self.diversity_archive[archive_index])  # Diversify using diversity archive
                    
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

                    # Add replaced vector to success archive if improvement is significant
                    if fitness[i] - f_trial > self.success_threshold:
                        if len(self.success_archive) < self.archive_size:
                            self.success_archive.append(population[i])
                            self.success_archive_fitness.append(fitness[i])
                        else:
                             # Replace worst in success archive
                            worst_archive_index = np.argmax(self.success_archive_fitness)
                            self.success_archive[worst_archive_index] = population[i]
                            self.success_archive_fitness[worst_archive_index] = fitness[i]

                    # Update optimal solution
                    if f_trial < self.f_opt:
                        self.f_opt = f_trial
                        self.x_opt = trial
                        self.stagnation_counter = 0
                        self.previous_best_fitness = self.f_opt
                
                # Add trial vector to diversity archive always, prioritizing diverse solutions
                if len(self.diversity_archive) < self.diversity_archive_size:
                    self.diversity_archive.append(trial)
                    self.diversity_archive_fitness.append(f_trial)
                else:
                    # Replace closest in diversity archive (replace the most similar one)
                    distances = [np.linalg.norm(trial - x) for x in self.diversity_archive]
                    closest_archive_index = np.argmin(distances)
                    self.diversity_archive[closest_archive_index] = trial
                    self.diversity_archive_fitness[closest_archive_index] = f_trial

            population = new_population
            fitness = new_fitness
            
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
                
            # Archive aging for success archive
            for k in range(len(self.success_archive_fitness)):
                self.success_archive_fitness[k] *= (1 - self.aging_rate)

            if self.budget <= 0:
                break
                
        return self.f_opt, self.x_opt