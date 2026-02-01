import numpy as np

class AdaptiveDE:
    def __init__(self, budget=10000, dim=10, pop_size=50, archive_size=100, F_init=0.5, CR_init=0.7, orthogonal_learning_freq=50):
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
        self.orthogonal_learning_freq = orthogonal_learning_freq

    def __call__(self, func):
        self.f_opt = np.Inf
        self.x_opt = None
        
        # Initialize population
        population = np.random.uniform(func.bounds.lb, func.bounds.ub, size=(self.pop_size, self.dim))
        fitness = np.array([func(x) for x in population])
        self.budget -= self.pop_size

        # Update optimal solution
        best_index = np.argmin(fitness)
        self.f_opt = fitness[best_index]
        self.x_opt = population[best_index]
        best_individual = population[best_index].copy()


        generation = 0
        stagnation_counter = 0
        prev_best_fitness = self.f_opt

        while self.budget > 0:
            generation += 1
            new_population = np.copy(population)
            new_fitness = np.copy(fitness)

            for i in range(self.pop_size):
                # Adaptive F and CR
                self.F_memory[i] = np.clip(np.random.normal(0.5, 0.3), 0.1, 1.0)
                self.CR_memory[i] = np.clip(np.random.normal(0.7, 0.1), 0.0, 1.0)

                # Mutation strategy: Self-adaptive current-to-best or current-to-archive
                if np.random.rand() < 0.5:
                    # Current-to-best
                    indices = np.random.choice(self.pop_size, 2, replace=False)
                    x1, x2 = population[indices]
                    mutant = population[i] + self.F_memory[i] * (best_individual - population[i]) + self.F_memory[i] * (x1 - x2)
                else:
                    # Current-to-archive
                    if len(self.archive) > 0:
                        archive_index = np.random.randint(len(self.archive))
                        x_archive = self.archive[archive_index]
                        indices = np.random.choice(self.pop_size, 1, replace=False)
                        x1 = population[indices[0]]
                        mutant = population[i] + self.F_memory[i] * (x_archive - population[i]) + self.F_memory[i] * (x1 - population[i])
                    else:
                        # If archive is empty, revert to current-to-best with random individuals
                        indices = np.random.choice(self.pop_size, 2, replace=False)
                        x1, x2 = population[indices]
                        mutant = population[i] + self.F_memory[i] * (best_individual - population[i]) + self.F_memory[i] * (x1 - x2)

                mutant = np.clip(mutant, func.bounds.lb, func.bounds.ub)

                # Crossover
                trial = np.copy(population[i])
                j_rand = np.random.randint(self.dim)
                for j in range(self.dim):
                    if np.random.rand() < self.CR_memory[i] or j == j_rand:
                        trial[j] = mutant[j]
                
                # Selection
                f_trial = func(trial)
                self.budget -= 1
                if f_trial < fitness[i]:
                    new_population[i] = trial
                    new_fitness[i] = f_trial

                    # Archive management: replace worst in archive or add if not full (Aging Archive)
                    if len(self.archive) < self.archive_size:
                        self.archive.append(population[i])
                        self.archive_fitness.append(fitness[i])
                    else:
                        #Aging mechanism: Give advantage to newer individuals
                        weights = np.linspace(1.0, 0.1, len(self.archive_fitness))
                        weighted_fitnesses = weights * np.array(self.archive_fitness)
                        worst_archive_index = np.argmax(weighted_fitnesses)
                        if f_trial < self.archive_fitness[worst_archive_index]:
                            self.archive[worst_archive_index] = population[i]
                            self.archive_fitness[worst_archive_index] = fitness[i]
                            
                    # Update optimal solution
                    if f_trial < self.f_opt:
                        self.f_opt = f_trial
                        self.x_opt = trial
                        best_individual = trial.copy()
                        stagnation_counter = 0 # Reset stagnation counter
                else:
                    # Archive management (trial vector): Replace worst in archive or add if not full
                    if len(self.archive) < self.archive_size:
                        self.archive.append(trial)
                        self.archive_fitness.append(f_trial)
                    else:
                         #Aging mechanism: Give advantage to newer individuals
                        weights = np.linspace(1.0, 0.1, len(self.archive_fitness))
                        weighted_fitnesses = weights * np.array(self.archive_fitness)
                        worst_archive_index = np.argmax(weighted_fitnesses)
                        if f_trial < self.archive_fitness[worst_archive_index]:
                            self.archive[worst_archive_index] = trial
                            self.archive_fitness[worst_archive_index] = f_trial
                        

            population = new_population
            fitness = new_fitness
            
            # Orthogonal Learning
            if generation % self.orthogonal_learning_freq == 0:
                self.perform_orthogonal_learning(func, population)
                fitness = np.array([func(x) for x in population])
                self.budget -= self.pop_size
                best_index = np.argmin(fitness)
                if fitness[best_index] < self.f_opt:
                    self.f_opt = fitness[best_index]
                    self.x_opt = population[best_index]
                    best_individual = population[best_index].copy()

            # Stagnation check and restart
            if abs(self.f_opt - prev_best_fitness) < 1e-8: #More robust stagnation detection
                stagnation_counter += 1
            else:
                stagnation_counter = 0
            prev_best_fitness = self.f_opt

            if stagnation_counter > 100: #Increased stagnation threshold
                population = np.random.uniform(func.bounds.lb, func.bounds.ub, size=(self.pop_size, self.dim))
                fitness = np.array([func(x) for x in population])
                self.budget -= self.pop_size
                best_index = np.argmin(fitness)
                self.f_opt = fitness[best_index]
                self.x_opt = population[best_index]
                best_individual = population[best_index].copy()
                stagnation_counter = 0  # Reset after restart
                self.archive = [] #Clear archive after restart
                self.archive_fitness = []

            if self.budget <= 0:
                break
                
        return self.f_opt, self.x_opt

    def perform_orthogonal_learning(self, func, population):
            # Select a subset of individuals for orthogonal learning
            num_ol_individuals = min(10, self.pop_size)  # Reduce the number for efficiency
            ol_indices = np.random.choice(self.pop_size, num_ol_individuals, replace=False)

            for i in ol_indices:
                # Generate orthogonal array (simplified 2-level OA)
                oa = np.array([[1, 1], [1, -1], [-1, 1], [-1, -1]])

                # Select a subset of dimensions for orthogonal learning
                num_ol_dimensions = min(5, self.dim) # Reduce the number for efficiency
                ol_dims = np.random.choice(self.dim, num_ol_dimensions, replace=False)

                # Create temporary solutions based on orthogonal array
                temp_solutions = np.zeros((len(oa), self.dim))
                for j in range(len(oa)):
                    temp_solution = population[i].copy()
                    for k, dim_index in enumerate(ol_dims):
                        # Map OA values (1, -1) to the upper and lower bounds of the selected dimensions
                        if oa[j, k % 2] == 1:
                            temp_solution[dim_index] = func.bounds.ub[dim_index]
                        else:
                            temp_solution[dim_index] = func.bounds.lb[dim_index]
                    temp_solutions[j] = temp_solution

                # Evaluate temporary solutions
                temp_fitnesses = np.array([func(x) for x in temp_solutions])
                self.budget -= len(temp_solutions)

                # Select the best solution among the temporary solutions
                best_temp_index = np.argmin(temp_fitnesses)

                # Replace the original individual with the best temporary solution
                population[i] = temp_solutions[best_temp_index]