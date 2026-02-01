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
        self.success_F = []
        self.success_CR = []
        self.ortho_group_size = ortho_group_size  # Number of individuals for orthogonal learning
        self.ortho_matrix = self._create_orthogonal_matrix(self.ortho_group_size)

    def _create_orthogonal_matrix(self, size):
        """Creates an orthogonal matrix for orthogonal learning."""
        if size == 2:
            return np.array([[1, 1], [1, -1]])
        elif size == 3:
            return np.array([[1, 1, 1], [1, -1, 0], [1, 1, -2]])
        elif size == 4:
             return np.array([[1, 1, 1, 1], [1, -1, 1, -1], [1, 1, -1, -1], [1, -1, -1, 1]])
        elif size == 5:
            return np.array([[1, 1, 1, 1, 1], [1, -1, 1, -1, 0], [1, 1, -2, 0, -1], [1, -1, -1, 1, 2], [1, 1, 1, 1, -4]])
        else:
            raise ValueError("Orthogonal matrix size must be 2, 3, 4 or 5.")


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

        while self.budget > 0:
            generation += 1
            new_population = np.copy(population)
            new_fitness = np.copy(fitness)

            for i in range(self.pop_size):
                # Adaptive F and CR
                if self.success_F:
                    self.F_memory[i] = np.clip(np.random.choice(self.success_F), 0.1, 1.0)
                else:
                    self.F_memory[i] = np.clip(np.random.normal(0.5, 0.3), 0.1, 1.0)

                if self.success_CR:
                    self.CR_memory[i] = np.clip(np.random.choice(self.success_CR), 0.0, 1.0)
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

                # Self-adaptive mutation strategy
                p = np.random.rand()
                if p < 0.33:
                    mutant = population[i] + self.F_memory[i] * (x1 - x2)  # DE/rand/1
                elif p < 0.66:
                    mutant = self.x_opt + self.F_memory[i] * (x1 - x2)  # DE/best/1
                else:
                    mutant = population[i] + self.F_memory[i] * (x1 - x2) + self.F_memory[i] * (x3 - population[i]) #DE/rand/2
                
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

                    # Update success memory for F and CR
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

            # Orthogonal learning
            if generation % 50 == 0:
                group_indices = np.random.choice(self.pop_size, self.ortho_group_size, replace=False)
                group = population[group_indices]
                group_fitness = fitness[group_indices]
                
                means = np.mean(group, axis=0)
                std = np.std(group, axis=0)
                
                levels = np.zeros_like(group)
                for k in range(self.ortho_group_size):
                  levels[k, :] = means + self.ortho_matrix[k, :] * std

                new_group = np.clip(levels, func.bounds.lb, func.bounds.ub)
                new_fitness_group = np.array([func(x) for x in new_group])
                self.budget -= self.ortho_group_size

                for k in range(self.ortho_group_size):
                  if new_fitness_group[k] < group_fitness[k]:
                    population[group_indices[k]] = new_group[k]
                    fitness[group_indices[k]] = new_fitness_group[k]
                    
                    if new_fitness_group[k] < self.f_opt:
                      self.f_opt = new_fitness_group[k]
                      self.x_opt = new_group[k]

            # Restart population if stagnating
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
                if fitness[best_index] < self.f_opt:
                    self.f_opt = fitness[best_index]
                    self.x_opt = population[best_index]
                stagnation_counter = 0  # Reset after restart
                self.archive = [] #Clear archive after restart
                self.archive_fitness = []
                self.success_F = []
                self.success_CR = []

            if self.budget <= 0:
                break
                
        return self.f_opt, self.x_opt