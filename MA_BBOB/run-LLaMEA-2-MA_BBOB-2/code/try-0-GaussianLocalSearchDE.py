import numpy as np

class GaussianLocalSearchDE:
    def __init__(self, budget=10000, dim=10, popsize=None, F=0.5, CR=0.7, archive_size=10):
        self.budget = budget
        self.dim = dim
        self.popsize = popsize if popsize is not None else 10 * self.dim
        self.F = F
        self.CR = CR
        self.archive_size = archive_size
        self.population = None
        self.fitness = None
        self.x_opt = None
        self.f_opt = np.inf
        self.eval_count = 0
        self.archive = []
        self.archive_fitness = []

    def initialize_population(self, func):
        lb = func.bounds.lb
        ub = func.bounds.ub
        self.population = np.random.uniform(lb, ub, size=(self.popsize, self.dim))
        self.fitness = np.array([func(x) for x in self.population])
        self.eval_count += self.popsize
        self.x_opt = self.population[np.argmin(self.fitness)]
        self.f_opt = np.min(self.fitness)

    def gaussian_local_search(self, x, func, sigma=0.1):
        x_new = x + np.random.normal(0, sigma, size=self.dim)
        x_new = np.clip(x_new, func.bounds.lb, func.bounds.ub)
        f_new = func(x_new)
        self.eval_count += 1
        return x_new, f_new

    def update_archive(self, x, f):
        if len(self.archive) < self.archive_size:
            self.archive.append(x)
            self.archive_fitness.append(f)
        else:
            if f < np.max(self.archive_fitness):
                worst_index = np.argmax(self.archive_fitness)
                self.archive[worst_index] = x
                self.archive_fitness[worst_index] = f

    def __call__(self, func):
        self.initialize_population(func)
        lb = func.bounds.lb
        ub = func.bounds.ub

        while self.eval_count < self.budget:
            for i in range(self.popsize):
                # Mutation
                donor_indices = np.random.choice(self.popsize, 3, replace=False)
                mutant = self.population[donor_indices[0]] + self.F * (self.population[donor_indices[1]] - self.population[donor_indices[2]])
                mutant = np.clip(mutant, lb, ub)

                # Crossover
                crossover_mask = np.random.rand(self.dim) < self.CR
                trial = np.where(crossover_mask, mutant, self.population[i])

                # Gaussian Local Search
                trial, f_trial = self.gaussian_local_search(trial, func)

                # Selection
                if f_trial < self.fitness[i]:
                    self.population[i] = trial
                    self.fitness[i] = f_trial

                    # Update archive
                    self.update_archive(trial, f_trial)

                    if f_trial < self.f_opt:
                        self.f_opt = f_trial
                        self.x_opt = trial
                else:
                    # If trial fails, also try local search on the original individual
                    x_local, f_local = self.gaussian_local_search(self.population[i], func)
                    if f_local < self.fitness[i]:
                        self.population[i] = x_local
                        self.fitness[i] = f_local
                        self.update_archive(x_local, f_local)
                        if f_local < self.f_opt:
                            self.f_opt = f_local
                            self.x_opt = x_local
                    else:
                        # Learn from archive
                        if self.archive:
                            archive_index = np.random.randint(len(self.archive))
                            archived_solution = self.archive[archive_index]
                            learning_rate = np.random.uniform(0, 1)
                            trial = self.population[i] + learning_rate * (archived_solution - self.population[i])
                            trial = np.clip(trial, lb, ub)
                            f_trial = func(trial)
                            self.eval_count += 1

                            if f_trial < self.fitness[i]:
                                self.population[i] = trial
                                self.fitness[i] = f_trial
                                self.update_archive(trial, f_trial)
                                if f_trial < self.f_opt:
                                    self.f_opt = f_trial
                                    self.x_opt = trial

            if self.eval_count >= self.budget:
                break

        return self.f_opt, self.x_opt