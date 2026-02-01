import numpy as np

class AdaptiveDE:
    def __init__(self, budget=10000, dim=10, pop_size_base=50, F_base=0.5, CR_base=0.7, archive_size=100, F_range=0.3, CR_range=0.3, local_search_radius=0.1, momentum=0.1, ortho_group_size=5):
        self.budget = budget
        self.dim = dim
        self.pop_size_base = pop_size_base
        self.pop_size = pop_size_base  # Initial population size
        self.F_base = F_base
        self.CR_base = CR_base
        self.archive_size = archive_size
        self.F_range = F_range
        self.CR_range = CR_range
        self.archive = []
        self.archive_fitness = []
        self.local_search_radius = local_search_radius
        self.momentum = momentum
        self.velocity = np.zeros((pop_size_base, dim)) # Initialize velocity for momentum
        self.ortho_group_size = ortho_group_size  # Size of orthogonal groups

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
        CR_history = np.full(self.pop_size, self.CR_base) #CR history for self-adaptation

        while self.budget > 0:
            generation += 1
            new_population = np.copy(population)
            new_fitness = np.copy(fitness)

            # Dynamic population size adjustment
            fitness_std = np.std(fitness)
            if fitness_std > 1e-3:
                self.pop_size = min(self.pop_size_base * 2, self.budget // 10)  # Increase population if diverse
            else:
                self.pop_size = max(self.pop_size_base // 2, 10)  # Reduce population if stagnant

            if self.pop_size != population.shape[0]:
                 # Resize population
                if self.pop_size > population.shape[0]:
                    # Add new random individuals
                    num_new = self.pop_size - population.shape[0]
                    new_individuals = np.random.uniform(func.bounds.lb, func.bounds.ub, size=(num_new, self.dim))
                    new_fitnesses = np.array([func(x) for x in new_individuals])
                    self.budget -= num_new
                    population = np.vstack((population, new_individuals))
                    fitness = np.concatenate((fitness, new_fitnesses))
                    CR_history = np.concatenate((CR_history, np.full(num_new, self.CR_base)))
                    self.velocity = np.vstack((self.velocity, np.zeros((num_new, self.dim))))
                else:
                    # Remove worst individuals
                    num_remove = population.shape[0] - self.pop_size
                    worst_indices = np.argsort(fitness)[-num_remove:]
                    remaining_indices = np.setdiff1d(np.arange(population.shape[0]), worst_indices)
                    population = population[remaining_indices]
                    fitness = fitness[remaining_indices]
                    CR_history = CR_history[remaining_indices]
                    self.velocity = self.velocity[remaining_indices]


            for i in range(population.shape[0]): #Iterate over the current population
                # Adaptive F and CR
                F = self.F_base + np.random.uniform(-self.F_range, self.F_range)
                F = np.clip(F, 0.1, 1.0)

                # Self-adaptive CR
                CR = CR_history[i] + np.random.normal(0, self.CR_range)
                CR = np.clip(CR, 0.1, 0.9)

                # Mutation with momentum
                indices = np.random.choice(population.shape[0], 3, replace=False)
                x1, x2, x3 = population[indices]
                
                # Momentum update
                self.velocity[i] = self.momentum * self.velocity[i] + F * (x2 - x3)
                mutant = x1 + self.velocity[i]

                # Use archive
                if np.random.rand() < 0.1 and len(self.archive) > 0:
                    archive_index = np.random.randint(len(self.archive))
                    mutant = x1 + F * (x2 - self.archive[archive_index])

                mutant = np.clip(mutant, func.bounds.lb, func.bounds.ub)

                # Crossover
                trial = np.copy(population[i])
                j_rand = np.random.randint(self.dim)
                for j in range(self.dim):
                    if np.random.rand() < CR or j == j_rand:
                        trial[j] = mutant[j]
                
                # Orthogonal learning within a small group
                if (i % self.ortho_group_size == 0) and (i + self.ortho_group_size <= population.shape[0]):
                    group = population[i:i + self.ortho_group_size]
                    group_trials = np.copy(group)
                    for k in range(self.ortho_group_size):
                        j_rand_ortho = np.random.randint(self.dim)
                        for j in range(self.dim):
                            if np.random.rand() < CR or j == j_rand_ortho:
                                group_trials[k, j] = mutant[j]
                                group_trials[k] = np.clip(group_trials[k], func.bounds.lb, func.bounds.ub)
                    
                    group_fitness = np.array([func(x) if self.budget > 0 else np.inf for x in group_trials])
                    self.budget -= np.sum(group_fitness != np.inf) #Adjust budget
                    best_index_group = np.argmin(group_fitness)

                    if group_fitness[best_index_group] < fitness[i + best_index_group]:
                        new_population[i + best_index_group] = group_trials[best_index_group]
                        new_fitness[i + best_index_group] = group_fitness[best_index_group]
                        CR_history[i + best_index_group] = CR
                        trial = new_population[i + best_index_group]  # Update trial for archive and best solution

                        # Add replaced vector to archive (combined strategy)
                        if len(self.archive) < self.archive_size:
                            self.archive.append(population[i + best_index_group])
                            self.archive_fitness.append(fitness[i + best_index_group])
                        else:
                             # Replace worst in archive
                            worst_archive_index = np.argmax(self.archive_fitness)
                            self.archive[worst_archive_index] = population[i + best_index_group]
                            self.archive_fitness[worst_archive_index] = fitness[i + best_index_group]
                        if group_fitness[best_index_group] < self.f_opt:
                            self.f_opt = group_fitness[best_index_group]
                            self.x_opt = group_trials[best_index_group]

                # Selection
                trial = np.clip(trial, func.bounds.lb, func.bounds.ub)
                f_trial = func(trial) if self.budget > 0 else np.inf
                self.budget -= 1
                if f_trial < fitness[i]:
                    new_population[i] = trial
                    new_fitness[i] = f_trial
                    CR_history[i] = CR # store successful CR

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

            # Stagnation check and periodic population rejuvenation
            if generation % 50 == 0:
                if np.std(fitness) < 1e-6:  # Stagnation criterion
                    stagnation_counter += 1
                    if stagnation_counter >= 2:
                         # Perform local search around the best solution
                        x_local = np.copy(self.x_opt)
                        for _ in range(min(self.budget // 10, 100)):  # Limited local search budget
                            x_new = x_local + np.random.uniform(-self.local_search_radius, self.local_search_radius, size=self.dim)
                            x_new = np.clip(x_new, func.bounds.lb, func.bounds.ub)
                            f_new = func(x_new) if self.budget > 0 else np.inf
                            self.budget -= 1
                            if f_new < self.f_opt:
                                self.f_opt = f_new
                                self.x_opt = x_new
                                x_local = np.copy(x_new)  # Move center of local search
                        stagnation_counter = 0 # Reset stagnation

                else:
                    stagnation_counter = 0 # Reset stagnation

                #Periodic rejuvenation
                if generation % 200 == 0:
                    # Replace a fraction of the population with new random solutions
                    num_rejuvenated = int(0.2 * population.shape[0])
                    new_individuals = np.random.uniform(func.bounds.lb, func.bounds.ub, size=(num_rejuvenated, self.dim))
                    new_fitnesses = np.array([func(x) if self.budget > 0 else np.inf for x in new_individuals])
                    self.budget -= np.sum(new_fitnesses != np.inf)

                    worst_indices = np.argsort(fitness)[-num_rejuvenated:]  # Indices of worst individuals
                    population[worst_indices] = new_individuals
                    fitness[worst_indices] = new_fitnesses

                    # Update optimal solution
                    best_index = np.argmin(fitness)
                    if fitness[best_index] < self.f_opt:
                        self.f_opt = fitness[best_index]
                        self.x_opt = population[best_index]
                

            if self.budget <= 0:
                break
                
        return self.f_opt, self.x_opt