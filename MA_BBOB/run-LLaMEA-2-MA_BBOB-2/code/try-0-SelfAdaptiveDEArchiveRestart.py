import numpy as np

class SelfAdaptiveDEArchiveRestart:
    def __init__(self, budget=10000, dim=10, popsize=None, F=0.5, CR=0.7, archive_size=50, restart_threshold=1e-9):
        self.budget = budget
        self.dim = dim
        self.popsize = popsize if popsize is not None else 10 * self.dim
        self.F = F
        self.CR = CR
        self.archive_size = archive_size
        self.restart_threshold = restart_threshold
        self.archive = []
        self.archive_fitness = []

    def __call__(self, func):
        lb = func.bounds.lb
        ub = func.bounds.ub
        self.population = np.random.uniform(lb, ub, size=(self.popsize, self.dim))
        self.fitness = np.array([func(x) for x in self.population])
        self.eval_count = self.popsize
        self.f_opt = np.min(self.fitness)
        self.x_opt = self.population[np.argmin(self.fitness)]

        memory_F = np.ones(self.popsize) * self.F
        memory_CR = np.ones(self.popsize) * self.CR

        while self.eval_count < self.budget:
            for i in range(self.popsize):
                # Mutation
                idxs = np.random.choice(self.popsize, 3, replace=False)
                x1, x2, x3 = self.population[idxs]

                F = np.random.normal(memory_F[i], 0.1)
                F = np.clip(F, 0.1, 1.0)
                
                mutant = x1 + F * (x2 - x3)
                mutant = np.clip(mutant, lb, ub)

                # Crossover
                CR = np.random.normal(memory_CR[i], 0.1)
                CR = np.clip(CR, 0.1, 1.0)
                
                crossover_mask = np.random.rand(self.dim) < CR
                trial = np.where(crossover_mask, mutant, self.population[i])

                # Selection
                f_trial = func(trial)
                self.eval_count += 1

                if f_trial < self.fitness[i]:
                    # Update individual
                    self.population[i] = trial
                    self.fitness[i] = f_trial
                    
                    # Update memory
                    memory_F[i] = F
                    memory_CR[i] = CR
                    
                    # Update archive
                    if len(self.archive) < self.archive_size:
                        self.archive.append(self.population[i].copy())
                        self.archive_fitness.append(self.fitness[i])
                    else:
                        # Replace the worst element in the archive
                        max_archive_idx = np.argmax(self.archive_fitness) # Higher fitness is worse!
                        if self.fitness[i] < self.archive_fitness[max_archive_idx]:
                            self.archive[max_archive_idx] = self.population[i].copy()
                            self.archive_fitness[max_archive_idx] = self.fitness[i]
                            
                    # Update best
                    if self.fitness[i] < self.f_opt:
                        self.f_opt = self.fitness[i]
                        self.x_opt = self.population[i]
                else:
                    # Add parent to archive if trial is worse
                    if len(self.archive) < self.archive_size:
                        self.archive.append(self.population[i].copy())
                        self.archive_fitness.append(self.fitness[i])
                    else:
                        # Replace the worst element in the archive
                        max_archive_idx = np.argmax(self.archive_fitness) # Higher fitness is worse!
                        if self.fitness[i] < self.archive_fitness[max_archive_idx]:
                            self.archive[max_archive_idx] = self.population[i].copy()
                            self.archive_fitness[max_archive_idx] = self.fitness[i]

            # Restart mechanism based on fitness concentration
            if np.std(self.fitness) < self.restart_threshold:
                # Replace a portion of the population with individuals from the archive and some random individuals
                num_archive = int(0.5 * self.popsize)
                num_random = self.popsize - num_archive

                if self.archive:
                    archive_indices = np.random.choice(len(self.archive), min(num_archive, len(self.archive)), replace=False)
                    self.population[:min(num_archive, len(self.archive))] = np.array(self.archive)[archive_indices]
                
                self.population[min(num_archive, len(self.archive)):] = np.random.uniform(lb, ub, size=(num_random, self.dim))

                self.fitness = np.array([func(x) for x in self.population])
                self.eval_count += num_random
                best_idx = np.argmin(self.fitness)
                self.f_opt = self.fitness[best_idx]
                self.x_opt = self.population[best_idx]
                memory_F = np.ones(self.popsize) * self.F
                memory_CR = np.ones(self.popsize) * self.CR

        return self.f_opt, self.x_opt