import numpy as np

class AdaptiveDE:
    def __init__(self, budget=10000, dim=10, pop_size=50, archive_size=100, F_init=0.5, CR_init=0.7, ortho_group_size=5):
        self.budget = budget
        self.dim = dim
        self.pop_size = pop_size
        self.archive_size = archive_size
        self.F_init = F_init
        self.CR_init = CR_init
        self.F_memory = np.ones(self.pop_size) * self.F_init
        self.CR_memory = np.ones(self.pop_size) * self.CR_init
        self.archive = []
        self.archive_fitness = []
        self.ortho_group_size = ortho_group_size
        self.min_pop_size = 10
        self.max_pop_size = 100

    def orthogonal_crossover(self, x, mutant):
        group_size = min(self.ortho_group_size, self.dim)
        num_groups = self.dim // group_size
        remainder = self.dim % group_size

        trial = np.copy(x)
        for i in range(num_groups):
            start_index = i * group_size
            end_index = start_index + group_size
            selected_indices = np.arange(start_index, end_index)
            
            if np.random.rand() < 0.5: #Apply crossover with 50% probability to each group
                for j in selected_indices:
                    trial[j] = mutant[j]

        #Handle remainder dimensions
        if remainder > 0:
            start_index = num_groups * group_size
            selected_indices = np.arange(start_index, self.dim)
            if np.random.rand() < 0.5: #Apply crossover to the remaining dims
                for j in selected_indices:
                    trial[j] = mutant[j]

        return trial

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
        archive_refresh_counter = 0
        
        while self.budget > 0:
            generation += 1
            new_population = np.copy(population)
            new_fitness = np.copy(fitness)

            for i in range(self.pop_size):
                # Adaptive F and CR
                self.F_memory[i] = np.clip(np.random.normal(0.5, 0.3), 0.1, 1.0)
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

                # Weighted difference vector
                mutant = population[i] + self.F_memory[i] * (x1 - x2) + self.F_memory[i] * 0.5 * (x3 - population[i])  # Reduced pull to archive
                mutant = np.clip(mutant, func.bounds.lb, func.bounds.ub)

                # Orthogonal Crossover
                trial = self.orthogonal_crossover(population[i], mutant)

                # Selection
                f_trial = func(trial)
                self.budget -= 1
                if f_trial < fitness[i]:
                    new_population[i] = trial
                    new_fitness[i] = f_trial

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
                        stagnation_counter = 0 # Reset stagnation counter
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
            
            #Adaptive Population Size adjustment
            if abs(self.f_opt - prev_best_fitness) < 1e-8:
                stagnation_counter += 1
                if self.pop_size > self.min_pop_size and stagnation_counter > 200:
                    self.pop_size = max(self.min_pop_size, int(self.pop_size * 0.9))  # Reduce population
                    population = population[:self.pop_size]
                    fitness = fitness[:self.pop_size]
                    self.F_memory = self.F_memory[:self.pop_size]
                    self.CR_memory = self.CR_memory[:self.pop_size]
                    stagnation_counter = 0
            else:
                stagnation_counter = 0
                if self.pop_size < self.max_pop_size and generation % 50 == 0:
                     self.pop_size = min(self.max_pop_size, int(self.pop_size * 1.1)) #Increase Population
                     new_individuals = np.random.uniform(func.bounds.lb, func.bounds.ub, size=(self.pop_size - len(population), self.dim))
                     new_fitness_vals = np.array([func(x) for x in new_individuals])
                     self.budget -= len(new_fitness_vals)
                     population = np.vstack((population, new_individuals))
                     fitness = np.concatenate((fitness, new_fitness_vals))
                     self.F_memory = np.concatenate((self.F_memory, np.ones(len(new_fitness_vals))*self.F_init))
                     self.CR_memory = np.concatenate((self.CR_memory, np.ones(len(new_fitness_vals))*self.CR_init))



            prev_best_fitness = self.f_opt

            # Periodic Archive Refreshment
            archive_refresh_counter += 1
            if archive_refresh_counter > 300:
                self.archive = []
                self.archive_fitness = []
                archive_refresh_counter = 0

            # Restart population if stagnating
            if stagnation_counter > 500: #Increased stagnation threshold
                population = np.random.uniform(func.bounds.lb, func.bounds.ub, size=(self.pop_size, self.dim))
                fitness = np.array([func(x) for x in population])
                self.budget -= self.pop_size
                best_index = np.argmin(fitness)
                if fitness[best_index] < self.f_opt:
                    self.f_opt = fitness[best_index]
                    self.x_opt = population[best_index]
                stagnation_counter = 0  # Reset after restart
                self.archive = [] #Clear archive after restart
                self.archive_fitness = []

            if self.budget <= 0:
                break
                
        return self.f_opt, self.x_opt