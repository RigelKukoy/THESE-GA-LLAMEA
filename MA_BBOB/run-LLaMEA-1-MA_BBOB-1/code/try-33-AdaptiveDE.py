import numpy as np

class AdaptiveDE:
    def __init__(self, budget=10000, dim=10, pop_size=50, F_base=0.5, CR_base=0.7, archive_size=100, F_range=0.3, CR_range=0.3, local_search_radius=0.1, local_search_frequency=50, diversity_threshold=1e-6, pop_size_reduction_factor = 0.9):
        self.budget = budget
        self.dim = dim
        self.pop_size = pop_size
        self.initial_pop_size = pop_size # Store the initial population size
        self.F_base = F_base
        self.CR_base = CR_base
        self.archive_size = archive_size
        self.F_range = F_range
        self.CR_range = CR_range
        self.archive = []
        self.archive_fitness = []
        self.local_search_radius = local_search_radius
        self.local_search_frequency = local_search_frequency
        self.diversity_threshold = diversity_threshold
        self.pop_size_reduction_factor = pop_size_reduction_factor

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
        while self.budget > 0:
            generation += 1
            new_population = np.copy(population)
            new_fitness = np.copy(fitness)

            # Calculate population diversity
            diversity = np.std(fitness)
            
            for i in range(self.pop_size):
                # Adaptive F and CR
                F = self.F_base + np.random.uniform(-self.F_range, self.F_range)
                CR = self.CR_base + np.random.uniform(-self.CR_range, self.CR_range)
                F = np.clip(F, 0.1, 1.0)
                CR = np.clip(CR, 0.1, 1.0)

                # Mutation: Diversity-guided strategy
                indices = np.random.choice(self.pop_size, 3, replace=False)
                x1, x2, x3 = population[indices]

                # If diversity is low, explore more using archive
                if diversity < self.diversity_threshold and len(self.archive) > 0:
                    archive_index = np.random.randint(len(self.archive))
                    mutant = x1 + F * (x2 - self.archive[archive_index])  # Exploration using archive
                else:
                    mutant = x1 + F * (x2 - x3)  # Standard DE mutation

                mutant = np.clip(mutant, func.bounds.lb, func.bounds.ub)

                # Crossover
                trial = np.copy(population[i])
                j_rand = np.random.randint(self.dim)
                for j in range(self.dim):
                    if np.random.rand() < CR or j == j_rand:
                        trial[j] = mutant[j]
                
                # Selection
                trial = np.clip(trial, func.bounds.lb, func.bounds.ub)
                f_trial = func(trial)
                self.budget -= 1
                if f_trial < fitness[i]:
                    new_population[i] = trial
                    new_fitness[i] = f_trial

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
            
            # Adaptive Local Search
            if generation % self.local_search_frequency == 0:
                 # Adjust frequency based on diversity
                if diversity < self.diversity_threshold:
                    self.local_search_frequency = max(10, self.local_search_frequency // 2) # Increase frequency if stagnating
                else:
                    self.local_search_frequency = min(50, self.local_search_frequency * 2)  # Reduce frequency if diverse
                
                x_local = np.copy(self.x_opt)
                local_search_evals = min(self.budget // 10, 100)
                for _ in range(local_search_evals):  # Limited local search budget
                    x_new = x_local + np.random.uniform(-self.local_search_radius, self.local_search_radius, size=self.dim)
                    x_new = np.clip(x_new, func.bounds.lb, func.bounds.ub)
                    f_new = func(x_new)
                    self.budget -= 1
                    if f_new < self.f_opt:
                        self.f_opt = f_new
                        self.x_opt = x_new
                        x_local = np.copy(x_new)  # Move center of local search
                        

            # Dynamic Population Sizing
            if generation % 100 == 0 and diversity < self.diversity_threshold and self.pop_size > 10:  # Reduce population if stagnating
                self.pop_size = int(self.pop_size * self.pop_size_reduction_factor)
                population = population[:self.pop_size]
                fitness = fitness[:self.pop_size]
                print(f"Reducing pop size to {self.pop_size}")
            elif diversity > 5 * self.diversity_threshold and self.pop_size < self.initial_pop_size:
                # Dynamically increase pop size based on diversity
                self.pop_size = min(self.initial_pop_size, self.pop_size + 10)
                print(f"Increasing pop size to {self.pop_size}")
                new_individuals = np.random.uniform(func.bounds.lb, func.bounds.ub, size=(10, self.dim))
                new_fitnesses = np.array([func(x) for x in new_individuals])
                self.budget -= 10

                population = np.vstack((population, new_individuals))
                fitness = np.concatenate((fitness, new_fitnesses))

            if self.budget <= 0:
                break
                
        return self.f_opt, self.x_opt