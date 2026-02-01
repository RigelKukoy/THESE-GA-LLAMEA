import numpy as np
from scipy.optimize import fmin_l_bfgs_b

class AdaptiveDE:
    def __init__(self, budget=10000, dim=10, pop_size_base=50, F_base=0.5, CR_base=0.7, archive_size=100, F_range=0.3, CR_range=0.3, local_search_prob=0.05):
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
        self.local_search_prob = local_search_prob

    def __call__(self, func):
        self.f_opt = np.Inf
        self.x_opt = None
        bounds = func.bounds
        lb, ub = bounds.lb, bounds.ub
        
        # Initialize population
        population = np.random.uniform(lb, ub, size=(self.pop_size, self.dim))
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

                # Add archive vector with small probability
                if np.random.rand() < 0.1 and len(self.archive) > 0:
                    x4 = self.archive[np.random.randint(len(self.archive))]
                    mutant = x1 + F * (x2 - x3) + F * (x4 - population[i]) # Including archive
                else:
                    mutant = x1 + F * (x2 - x3)
                mutant = np.clip(mutant, lb, ub)

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

            # Local Search with L-BFGS-B
            if np.random.rand() < self.local_search_prob:
                index = np.random.randint(self.pop_size)
                x_local, f_local, d = fmin_l_bfgs_b(func, population[index], bounds=[(lb, ub)] * self.dim, approx_grad=True, maxfun=self.budget // 100) # Reduce maxfun
                self.budget -= d['funcalls']

                if f_local < fitness[index]:
                    population[index] = x_local
                    fitness[index] = f_local
                    if f_local < self.f_opt:
                        self.f_opt = f_local
                        self.x_opt = x_local

            # Adaptive Population Size
            if generation % 20 == 0:
                if np.std(fitness) < 1e-6:
                    self.pop_size = int(self.pop_size * 0.8)
                else:
                    self.pop_size = min(self.pop_size_base, self.budget // (2 * self.dim))
                self.pop_size = max(10, self.pop_size)

            # Restart population if stagnating
            if generation % 50 == 0:
                if np.std(fitness) < 1e-6:  # Stagnation criterion
                    population = np.random.uniform(lb, ub, size=(self.pop_size, self.dim))
                    fitness = np.array([func(x) for x in population])
                    self.budget -= self.pop_size
                    best_index = np.argmin(fitness)
                    if fitness[best_index] < self.f_opt:
                        self.f_opt = fitness[best_index]
                        self.x_opt = population[best_index]

            if self.budget <= 0:
                break
                
        return self.f_opt, self.x_opt