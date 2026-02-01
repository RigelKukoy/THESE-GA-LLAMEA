import numpy as np

class ArchiveDE:
    def __init__(self, budget=10000, dim=10, popsize=None, CR=0.7, F=0.5, archive_size=10, archive_rate=0.1):
        self.budget = budget
        self.dim = dim
        self.popsize = popsize if popsize is not None else 10 * self.dim
        self.CR = CR
        self.F = F
        self.archive_size = archive_size
        self.archive_rate = archive_rate
        self.archive = []

    def __call__(self, func):
        lb = func.bounds.lb
        ub = func.bounds.ub
        self.population = np.random.uniform(lb, ub, size=(self.popsize, self.dim))
        self.fitness = np.array([func(x) for x in self.population])
        self.eval_count = self.popsize
        self.f_opt = np.min(self.fitness)
        self.x_opt = self.population[np.argmin(self.fitness)]

        while self.eval_count < self.budget:
            for i in range(self.popsize):
                # Mutation
                idxs = np.random.choice(self.popsize, 3, replace=False)
                x1, x2, x3 = self.population[idxs]

                # With probability archive_rate, use a vector from the archive
                if self.archive and np.random.rand() < self.archive_rate:
                    idx_archive = np.random.randint(len(self.archive))
                    x_archive = self.archive[idx_archive]
                    mutant = self.population[i] + self.F * (x1 - x2) + self.F * (x_archive - x3) # archive individual mixed with the population
                else:
                    mutant = self.population[i] + self.F * (x2 - x3)

                mutant = np.clip(mutant, lb, ub)

                # Crossover
                crossover_mask = np.random.rand(self.dim) < self.CR
                trial = np.where(crossover_mask, mutant, self.population[i])

                # Selection
                f_trial = func(trial)
                self.eval_count += 1

                if f_trial < self.fitness[i]:
                    # Replace individual
                    self.population[i] = trial
                    self.fitness[i] = f_trial

                    # Update optimal solution
                    if self.fitness[i] < self.f_opt:
                        self.f_opt = self.fitness[i]
                        self.x_opt = self.population[i]
                else:
                    # Add the rejected individual to the archive
                    if len(self.archive) < self.archive_size:
                        self.archive.append(self.population[i])
                    else:
                        # Replace a random element in the archive
                        idx_replace = np.random.randint(self.archive_size)
                        self.archive[idx_replace] = self.population[i]
                    

            # Potentially prune the archive (optional)
            if len(self.archive) > self.archive_size:
                self.archive = self.archive[:self.archive_size]


        return self.f_opt, self.x_opt