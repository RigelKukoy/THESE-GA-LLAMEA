import numpy as np

class AdaptiveDE:
    def __init__(self, budget=10000, dim=10, pop_size=50, F=0.5, CR=0.7, archive_size=100):
        self.budget = budget
        self.dim = dim
        self.pop_size = pop_size
        self.F = F
        self.CR = CR
        self.archive_size = archive_size
        self.F_memory = np.ones(self.pop_size) * 0.5
        self.CR_memory = np.ones(self.pop_size) * 0.7

    def __call__(self, func):
        self.f_opt = np.Inf
        self.x_opt = None
        self.archive = []
        self.archive_fitness = []
        
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
                self.F_memory[i] = np.clip(np.random.normal(0.5, 0.3), 0.1, 1.0)
                self.CR_memory[i] = np.clip(np.random.normal(0.7, 0.1), 0.0, 1.0)

                # Mutation
                indices = np.random.choice(self.pop_size, 2, replace=False)
                x1, x2 = population[indices]

                # Utilize archive
                if len(self.archive) > 0 and np.random.rand() < 0.2:
                     archive_index = np.random.randint(len(self.archive))
                     x3 = self.archive[archive_index]
                else:
                    indices = np.random.choice(self.pop_size, 1, replace=False)
                    x3 = population[indices[0]]

                mutant = population[i] + self.F_memory[i] * (x1 - x2) + self.F_memory[i] * (x3 - population[i]) # Added extra term to pull to archive
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

                    # Add replaced vector to archive
                    if len(self.archive) < self.archive_size:
                        self.archive.append(population[i])
                    else:
                        #Replace oldest in archive
                        self.archive[generation % self.archive_size] = population[i]

                    # Update optimal solution
                    if f_trial < self.f_opt:
                        self.f_opt = f_trial
                        self.x_opt = trial
                        stagnation_counter = 0 # Reset stagnation counter
                else:
                    # Add trial vector to archive
                    if len(self.archive) < self.archive_size:
                        self.archive.append(trial)
                    else:
                        #Replace oldest in archive
                        self.archive[generation % self.archive_size] = trial
                        

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

            if self.budget <= 0:
                break
                
        return self.f_opt, self.x_opt