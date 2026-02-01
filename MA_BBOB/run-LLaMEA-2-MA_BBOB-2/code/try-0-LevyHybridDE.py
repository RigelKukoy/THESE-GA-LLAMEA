import numpy as np

class LevyHybridDE:
    def __init__(self, budget=10000, dim=10, popsize=None, F=0.5, CR=0.7, archive_size=10):
        self.budget = budget
        self.dim = dim
        self.popsize = popsize if popsize is not None else 10 * self.dim
        self.F = F
        self.CR = CR
        self.archive_size = archive_size
        self.archive = []
        self.archive_fitness = []

    def levy_flight(self, size, lam=1.5):
        sigma = (np.math.gamma(1 + lam) * np.sin(np.pi * lam / 2) / (np.math.gamma((1 + lam) / 2) * lam * (2 ** ((lam - 1) / 2)))) ** (1 / lam)
        u = np.random.normal(0, sigma, size=size)
        v = np.random.normal(0, 1, size=size)
        s = u / (np.abs(v) ** (1 / lam))
        return s

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
                # Mutation with Lévy Flight
                idxs = np.random.choice(self.popsize, 3, replace=False)
                x1, x2, x3 = self.population[idxs]
                levy_step = self.levy_flight(self.dim)
                mutant = x1 + self.F * (x2 - x3) + 0.01 * levy_step * (ub - lb) # Levy flight scaled to the search space
                mutant = np.clip(mutant, lb, ub)

                # Crossover
                crossover_mask = np.random.rand(self.dim) < self.CR
                trial = np.where(crossover_mask, mutant, self.population[i])

                # Selection
                f_trial = func(trial)
                self.eval_count += 1

                if f_trial < self.fitness[i]:
                    # Update population
                    self.population[i] = trial
                    self.fitness[i] = f_trial

                    # Update archive
                    if len(self.archive) < self.archive_size:
                        self.archive.append(trial)
                        self.archive_fitness.append(f_trial)
                    else:
                        max_archive_fitness_idx = np.argmax(self.archive_fitness)
                        if f_trial < self.archive_fitness[max_archive_fitness_idx]:
                            self.archive[max_archive_fitness_idx] = trial
                            self.archive_fitness[max_archive_fitness_idx] = f_trial
                            
                    if self.fitness[i] < self.f_opt:
                        self.f_opt = self.fitness[i]
                        self.x_opt = self.population[i]
                else:
                    # If trial is not better than current, consider adding parent to archive
                    if len(self.archive) < self.archive_size:
                        self.archive.append(self.population[i])
                        self.archive_fitness.append(self.fitness[i])
                    else:
                        max_archive_fitness_idx = np.argmax(self.archive_fitness)
                        if self.fitness[i] < self.archive_fitness[max_archive_fitness_idx]:
                            self.archive[max_archive_fitness_idx] = self.population[i]
                            self.archive_fitness[max_archive_fitness_idx] = self.fitness[i]

                # Elitism: Include best archive member in mutation if archive is sufficiently filled
                if len(self.archive) == self.archive_size and np.random.rand() < 0.1:
                    best_archive_idx = np.argmin(self.archive_fitness)
                    best_archive_member = self.archive[best_archive_idx]
                    idxs = np.random.choice(self.popsize, 2, replace=False)
                    x2, x3 = self.population[idxs]
                    mutant = best_archive_member + self.F * (x2 - x3)
                    mutant = np.clip(mutant, lb, ub)
                    
                    crossover_mask = np.random.rand(self.dim) < self.CR
                    trial = np.where(crossover_mask, mutant, self.population[i])

                    f_trial = func(trial)
                    self.eval_count += 1
                    
                    if f_trial < self.fitness[i]:
                        self.population[i] = trial
                        self.fitness[i] = f_trial
                        if self.fitness[i] < self.f_opt:
                            self.f_opt = self.fitness[i]
                            self.x_opt = self.population[i]


        return self.f_opt, self.x_opt