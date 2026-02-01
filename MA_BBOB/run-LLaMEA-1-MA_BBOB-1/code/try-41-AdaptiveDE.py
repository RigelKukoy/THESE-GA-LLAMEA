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
        self.archive_age = []
        self.success_F = []
        self.success_CR = []
        self.min_F = 0.1
        self.max_F = 0.9
        self.archive_prob = 0.2  # Static archive probability


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
                # Adaptive F and CR using success history (Le Cun style update)
                if self.success_F:
                    F_mean = np.mean(self.success_F)
                    self.F_memory[i] = np.clip(np.random.normal(F_mean, 0.1), self.min_F, self.max_F)
                else:
                    self.F_memory[i] = np.clip(np.random.normal(0.5, 0.1), self.min_F, self.max_F)

                if self.success_CR:
                    CR_mean = np.mean(self.success_CR)
                    self.CR_memory[i] = np.clip(np.random.normal(CR_mean, 0.1), 0.0, 1.0)
                else:
                    self.CR_memory[i] = np.clip(np.random.normal(0.7, 0.1), 0.0, 1.0)

                # Mutation: Simplified DE/rand/1 strategy with archive
                indices = np.random.choice(self.pop_size, 3, replace=False)
                x1, x2, x3 = population[indices]

                if len(self.archive) > 0 and np.random.rand() < self.archive_prob:
                    archive_index = np.random.randint(len(self.archive))
                    x3 = self.archive[archive_index]

                mutant = x1 + self.F_memory[i] * (x2 - x3)
                mutant = np.clip(mutant, func.bounds.lb, func.bounds.ub)

                # Orthogonal Crossover
                trial = np.copy(population[i])
                num_changed_vars = 0
                for j in range(self.dim):
                    if np.random.rand() < self.CR_memory[i]:
                        trial[j] = mutant[j]
                        num_changed_vars += 1
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

                    # Archive management: replace worst in archive
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

                    if f_trial < self.f_opt:
                        self.f_opt = f_trial
                        self.x_opt = trial
                        stagnation_counter = 0
                else:
                    # Archive management: replace worst in archive (trial vector)
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
                            
            # Update age of archive members and clear out old ones
            self.archive_age = [age + 1 for age in self.archive_age]

            # Diversity Maintenance
            if generation % 50 == 0:
                distances = np.zeros((self.pop_size, self.pop_size))
                for k in range(self.pop_size):
                    for l in range(k + 1, self.pop_size):
                        distances[k, l] = np.linalg.norm(population[k] - population[l])
                        distances[l, k] = distances[k, l]

                min_dist = np.min(distances + np.eye(self.pop_size) * 1e9)
                if min_dist < 0.1:
                    # Introduce random individuals
                    for k in range(self.pop_size):
                        if np.random.rand() < 0.1:
                            population[k] = np.random.uniform(func.bounds.lb, func.bounds.ub)
                            fitness[k] = func(population[k])
                            self.budget -= 1
                            if fitness[k] < self.f_opt:
                                self.f_opt = fitness[k]
                                self.x_opt = population[k]

            population = new_population
            fitness = new_fitness

            if abs(self.f_opt - prev_best_fitness) < 1e-8:
                stagnation_counter += 1
            else:
                stagnation_counter = 0
            prev_best_fitness = self.f_opt

            if stagnation_counter > 100:
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
                self.archive_age = []
                self.success_F = []
                self.success_CR = []

            if generation % archive_clear_interval == 0:
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