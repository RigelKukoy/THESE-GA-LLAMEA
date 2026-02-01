import numpy as np

class AdaptiveDE:
    def __init__(self, budget=10000, dim=10, pop_size=50, archive_size=100, F_init=0.5, CR_init=0.7):
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
        self.archive_age = [] # Track age of individuals in archive
        self.success_F = []
        self.success_CR = []


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
        archive_clear_interval = 200 # Clear the archive periodically
        
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

                # Weighted difference vector
                mutant = population[i] + self.F_memory[i] * (x1 - x2) + self.F_memory[i] * 0.5 * (x3 - population[i])  # Reduced pull to archive
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
                        self.archive_age.append(0)
                    else:
                        worst_archive_index = np.argmax(self.archive_fitness)
                        if fitness[i] < self.archive_fitness[worst_archive_index]:
                            self.archive[worst_archive_index] = population[i]
                            self.archive_fitness[worst_archive_index] = fitness[i]
                            self.archive_age[worst_archive_index] = 0

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
                        self.archive_age.append(0)
                    else:
                        worst_archive_index = np.argmax(self.archive_fitness)
                        if f_trial < self.archive_fitness[worst_archive_index]:
                            self.archive[worst_archive_index] = trial
                            self.archive_fitness[worst_archive_index] = f_trial
                            self.archive_age[worst_archive_index] = 0

            # Update age of archive members
            self.archive_age = [age + 1 for age in self.archive_age]


            population = new_population
            fitness = new_fitness

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
                self.archive_age = []
                self.success_F = []
                self.success_CR = []

            # Periodic archive clearing
            if generation % archive_clear_interval == 0:
                #Remove old or redundant entries to increase diversity
                if len(self.archive) > 0:
                    max_age = np.max(self.archive_age)
                    to_remove = []
                    for k in range(len(self.archive)):
                        if self.archive_age[k] >= max_age:
                            to_remove.append(k)

                    for k in sorted(to_remove, reverse=True):
                        del self.archive[k]
                        del self.archive_fitness[k]
                        del self.archive_age[k]



            if self.budget <= 0:
                break
                
        return self.f_opt, self.x_opt