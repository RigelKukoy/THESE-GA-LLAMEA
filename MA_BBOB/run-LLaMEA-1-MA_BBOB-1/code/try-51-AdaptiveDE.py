import numpy as np

class AdaptiveDE:
    def __init__(self, budget=10000, dim=10, pop_size=50, archive_size_init=50, F_init=0.5, CR_init=0.7, stagnation_threshold=50):
        self.budget = budget
        self.dim = dim
        self.pop_size = pop_size
        self.archive_size_init = archive_size_init
        self.archive_size = archive_size_init #Dynamic archive size
        self.F_init = F_init
        self.CR_init = CR_init
        self.F_memory = np.ones(self.pop_size) * self.F_init
        self.CR_memory = np.ones(self.pop_size) * self.CR_init
        self.archive = []
        self.archive_fitness = []
        self.success_F = []
        self.success_CR = []
        self.stagnation_threshold = stagnation_threshold
        self.C = np.eye(dim) # Covariance matrix for CMA-ES-like mutation
        self.c_learn = 0.1 #Learning rate for CMA


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

        generation = 0
        stagnation_counter = 0
        prev_best_fitness = self.f_opt
        archive_clear_interval = 200
        
        while self.budget > 0:
            generation += 1
            new_population = np.copy(population)
            new_fitness = np.copy(fitness)

            for i in range(self.pop_size):
                # Adaptive F and CR using success history
                if self.success_F:
                    F_mean = np.mean(self.success_F)
                    self.F_memory[i] = np.clip(np.random.normal(F_mean, 0.3), 0.1, 1.0)
                else:
                    self.F_memory[i] = np.clip(np.random.normal(0.5, 0.3), 0.1, 1.0)

                if self.success_CR:
                    CR_mean = np.mean(self.success_CR)
                    self.CR_memory[i] = np.clip(np.random.normal(CR_mean, 0.1), 0.0, 1.0)
                else:
                    self.CR_memory[i] = np.clip(np.random.normal(0.7, 0.1), 0.0, 1.0)


                # Mutation
                indices = np.random.choice(self.pop_size, 2, replace=False)
                x1, x2 = population[indices]

                # Utilize archive with dynamic probability
                archive_prob = min(0.4, generation / 500)  # Increase archive usage over time
                if len(self.archive) > 0 and np.random.rand() < archive_prob:
                     archive_index = np.random.randint(len(self.archive))
                     x3 = self.archive[archive_index]
                else:
                    indices = np.random.choice(self.pop_size, 1, replace=False)
                    x3 = population[indices[0]]

                # Weighted difference vector with CMA-ES-like adaptation
                z = np.random.multivariate_normal(np.zeros(self.dim), self.C)
                mutant = population[i] + self.F_memory[i] * (x1 - x2) + self.F_memory[i] * 0.5 * (x3 - population[i]) + 0.1 * z # CMA-ES part
                mutant = np.clip(mutant, func.bounds.lb, func.bounds.ub)

                # Orthogonal Crossover
                trial = np.copy(population[i])
                num_changed_vars = 0
                for j in range(self.dim):
                    if np.random.rand() < self.CR_memory[i]:
                        trial[j] = mutant[j]
                        num_changed_vars += 1
                # Ensure at least one variable is changed
                if num_changed_vars == 0:
                    j_rand = np.random.randint(self.dim)
                    trial[j_rand] = mutant[j_rand]
                
                # Selection
                f_trial = func(trial)
                self.budget -= 1
                if f_trial < fitness[i]:
                    new_population[i] = trial
                    new_fitness[i] = f_trial

                    self.success_F.append(self.F_memory[i])
                    self.success_CR.append(self.CR_memory[i])
                    if len(self.success_F) > 10:
                        self.success_F.pop(0)
                        self.success_CR.pop(0)

                    # Dynamic archive management: replace worst in archive
                    if len(self.archive) < self.archive_size:
                        self.archive.append(population[i])
                        self.archive_fitness.append(fitness[i])
                    else:
                        worst_archive_index = np.argmax(self.archive_fitness)
                        if fitness[i] < self.archive_fitness[worst_archive_index]:
                            self.archive[worst_archive_index] = population[i]
                            self.archive_fitness[worst_archive_index] = fitness[i]

                    # Update optimal solution
                    if f_trial < self.f_opt:
                        self.f_opt = f_trial
                        self.x_opt = trial
                        stagnation_counter = 0
                        # CMA-ES Learning: Update Covariance matrix based on successful step
                        d = trial - population[i]
                        self.C = (1 - self.c_learn) * self.C + self.c_learn * np.outer(d, d)

                else:
                    # Dynamic archive management: replace worst in archive (trial vector)
                    if len(self.archive) < self.archive_size:
                        self.archive.append(trial)
                        self.archive_fitness.append(f_trial)
                    else:
                        worst_archive_index = np.argmax(self.archive_fitness)
                        if f_trial < self.archive_fitness[worst_archive_index]:
                            self.archive[worst_archive_index] = trial
                            self.archive_fitness[worst_archive_index] = f_trial

            population = new_population
            fitness = new_fitness

            # Aggressive stagnation check
            if abs(self.f_opt - prev_best_fitness) < 1e-9:
                stagnation_counter += 1
            else:
                stagnation_counter = 0
            prev_best_fitness = self.f_opt

            if stagnation_counter > self.stagnation_threshold:
                population = np.random.uniform(func.bounds.lb, func.bounds.ub, size=(self.pop_size, self.dim))
                fitness = np.array([func(x) for x in population])
                self.budget -= self.pop_size
                best_index = np.argmin(fitness)
                if fitness[best_index] < self.f_opt:
                    self.f_opt = fitness[best_index]
                    self.x_opt = population[best_index]
                stagnation_counter = 0
                self.archive = []
                self.archive_fitness = []
                self.success_F = []
                self.success_CR = []
                self.C = np.eye(self.dim) #Reset CMA

                #Adjust archive size upon restart
                self.archive_size = min(self.archive_size_init + 10, self.pop_size * 2) #Increase size after stagnation

            #Periodic archive clearing, less frequent and simpler
            if generation % archive_clear_interval == 0 and len(self.archive) > self.archive_size // 2:
                  worst_archive_index = np.argmax(self.archive_fitness)
                  del self.archive[worst_archive_index]
                  del self.archive_fitness[worst_archive_index]

            #Dynamically adjust archive size
            if generation % 100 == 0:
              if len(self.archive) < self.archive_size // 2:
                self.archive_size = max(self.archive_size - 5, 10) #Reduce archive if underutilized
              elif len(self.archive) == self.archive_size:
                self.archive_size = min(self.archive_size + 5, self.pop_size * 2) #Increase if full.


            if self.budget <= 0:
                break
                
        return self.f_opt, self.x_opt