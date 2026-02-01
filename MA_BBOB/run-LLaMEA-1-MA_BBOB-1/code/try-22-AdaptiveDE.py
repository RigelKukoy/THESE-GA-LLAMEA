import numpy as np

class AdaptiveDE:
    def __init__(self, budget=10000, dim=10, pop_size=50, F_base=0.5, CR_base=0.7, archive_size=100, F_range=0.3, CR_range=0.3, age_limit=50):
        self.budget = budget
        self.dim = dim
        self.pop_size = pop_size
        self.F_base = F_base
        self.CR_base = CR_base
        self.archive_size = archive_size
        self.F_range = F_range
        self.CR_range = CR_range
        self.archive = []
        self.archive_fitness = []
        self.age_limit = age_limit
        self.population_ages = np.zeros(pop_size, dtype=int)

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
        while self.budget > 0:
            generation += 1
            new_population = np.copy(population)
            new_fitness = np.copy(fitness)

            for i in range(self.pop_size):
                # Adaptive F and CR
                F = self.F_base + np.random.uniform(-self.F_range, self.F_range)
                CR = self.CR_base + np.random.uniform(-self.CR_range, self.CR_range)
                F = np.clip(F, 0.1, 1.0)
                CR = np.clip(CR, 0.1, 1.0)

                # Mutation
                indices = np.random.choice(self.pop_size, 3, replace=False)
                x1, x2, x3 = population[indices]
                mutant = x1 + F * (x2 - x3)
                mutant = np.clip(mutant, func.bounds.lb, func.bounds.ub)

                # Crossover
                trial = np.copy(population[i])
                j_rand = np.random.randint(self.dim)
                for j in range(self.dim):
                    if np.random.rand() < CR or j == j_rand:
                        trial[j] = mutant[j]
                
                # Selection
                f_trial = func(trial)
                self.budget -= 1
                if f_trial < fitness[i]:
                    new_population[i] = trial
                    new_fitness[i] = f_trial
                    self.population_ages[i] = 0  # Reset age

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
                    self.population_ages[i] += 1  # Increment age
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

            # Replace old individuals
            for i in range(self.pop_size):
                if self.population_ages[i] > self.age_limit:
                    population[i] = np.random.uniform(func.bounds.lb, func.bounds.ub)
                    fitness[i] = func(population[i])
                    self.budget -= 1
                    self.population_ages[i] = 0  # Reset age
                    
                    if fitness[i] < self.f_opt:
                        self.f_opt = fitness[i]
                        self.x_opt = population[i]
            
            # Restart population if stagnating
            if generation % 50 == 0:
                if np.std(fitness) < 1e-6:  # Stagnation criterion
                    population = np.random.uniform(func.bounds.lb, func.bounds.ub, size=(self.pop_size, self.dim))
                    fitness = np.array([func(x) for x in population])
                    self.budget -= self.pop_size
                    self.population_ages = np.zeros(self.pop_size, dtype=int) # Reset all ages
                    best_index = np.argmin(fitness)
                    if fitness[best_index] < self.f_opt:
                        self.f_opt = fitness[best_index]
                        self.x_opt = population[best_index]

            if self.budget <= 0:
                break
                
        return self.f_opt, self.x_opt