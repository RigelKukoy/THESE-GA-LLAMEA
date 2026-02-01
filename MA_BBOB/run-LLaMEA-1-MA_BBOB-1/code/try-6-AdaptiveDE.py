import numpy as np

class AdaptiveDE:
    def __init__(self, budget=10000, dim=10, pop_size=50, archive_size=100, F_init=0.5, CR_init=0.7, local_search_prob=0.1):
        self.budget = budget
        self.dim = dim
        self.pop_size = pop_size
        self.archive_size = archive_size
        self.F_init = F_init
        self.CR_init = CR_init
        self.local_search_prob = local_search_prob

    def __call__(self, func):
        self.f_opt = np.Inf
        self.x_opt = None
        self.archive = []
        self.archive_fitness = []
        self.F = np.full(self.pop_size, self.F_init)
        self.CR = np.full(self.pop_size, self.CR_init)
        
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
        while self.budget > 0:
            generation += 1
            new_population = np.copy(population)
            new_fitness = np.copy(fitness)

            for i in range(self.pop_size):
                # Adaptive F and CR
                self.F[i] = np.clip(np.random.normal(self.F_init, 0.1), 0.1, 1.0)
                self.CR[i] = np.clip(np.random.normal(self.CR_init, 0.1), 0.1, 1.0)

                # Mutation
                indices = np.random.choice(self.pop_size, 3, replace=False)
                x1, x2, x3 = population[indices]
                mutant = x1 + self.F[i] * (x2 - x3)

                # Use archive
                if np.random.rand() < 0.1 and len(self.archive) > 0:
                    x4 = self.archive[np.random.randint(len(self.archive))]
                    mutant = x1 + self.F[i] * (x2 - x4)

                mutant = np.clip(mutant, func.bounds.lb, func.bounds.ub)

                # Crossover
                trial = np.copy(population[i])
                j_rand = np.random.randint(self.dim)
                for j in range(self.dim):
                    if np.random.rand() < self.CR[i] or j == j_rand:
                        trial[j] = mutant[j]
                
                # Local Search
                if np.random.rand() < self.local_search_prob:
                    trial_copy = np.copy(trial)
                    step_size = 0.01 * (func.bounds.ub - func.bounds.lb)
                    for j in range(self.dim):
                        trial_copy[j] = np.clip(trial[j] + np.random.uniform(-step_size, step_size), func.bounds.lb, func.bounds.ub)
                    f_trial = func(trial_copy)
                    self.budget -= 1
                    if f_trial < fitness[i]:
                        trial = trial_copy
                    else:
                         f_trial = func(trial)
                         self.budget -= 1
                else:
                   f_trial = func(trial)
                   self.budget -= 1
               
                # Selection
                if f_trial < fitness[i]:
                    new_population[i] = trial
                    new_fitness[i] = f_trial

                    # Add replaced vector to archive
                    if len(self.archive) < self.archive_size:
                        self.archive.append(population[i])
                        self.archive_fitness.append(fitness[i])
                    else:
                        #Replace oldest in archive
                        self.archive[generation % self.archive_size] = population[i]
                        self.archive_fitness[generation % self.archive_size] = fitness[i]

                    # Update optimal solution
                    if f_trial < self.f_opt:
                        self.f_opt = f_trial
                        self.x_opt = trial
                        
                else:
                    # Add trial vector to archive
                    if len(self.archive) < self.archive_size:
                        self.archive.append(trial)
                        self.archive_fitness.append(f_trial)
                    else:
                        #Replace oldest in archive
                        self.archive[generation % self.archive_size] = trial
                        self.archive_fitness[generation % self.archive_size] = f_trial
                        

            population = new_population
            fitness = new_fitness

            # Restart population if stagnating
            if generation % 50 == 0:
                if np.std(fitness) < 1e-6:  # Stagnation criterion
                    population = np.random.uniform(func.bounds.lb, func.bounds.ub, size=(self.pop_size, self.dim))
                    fitness = np.array([func(x) for x in population])
                    self.budget -= self.pop_size
                    best_index = np.argmin(fitness)
                    if fitness[best_index] < self.f_opt:
                        self.f_opt = fitness[best_index]
                        self.x_opt = population[best_index]

            if self.budget <= 0:
                break
                
        return self.f_opt, self.x_opt