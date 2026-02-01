import numpy as np

class ArchivedDifferentialEvolution:
    def __init__(self, budget=10000, dim=10, pop_size=50, archive_size=25, F=0.5, CR=0.9):
        self.budget = budget
        self.dim = dim
        self.pop_size = pop_size
        self.archive_size = archive_size
        self.F = F
        self.CR = CR
        self.archive = []

    def __call__(self, func):
        self.f_opt = np.Inf
        self.x_opt = None
        self.population = np.random.uniform(func.bounds.lb, func.bounds.ub, size=(self.pop_size, self.dim))
        fitness = np.array([func(x) for x in self.population])
        self.budget -= self.pop_size

        for i in range(self.pop_size):
            if fitness[i] < self.f_opt:
                self.f_opt = fitness[i]
                self.x_opt = self.population[i]

        while self.budget > 0:
            for i in range(self.pop_size):
                # Mutation
                idxs = np.random.choice(self.pop_size, 3, replace=False)
                x_1, x_2, x_3 = self.population[idxs]

                # Add a random solution from the archive if the archive is not empty
                if self.archive:
                    x_4 = self.archive[np.random.choice(len(self.archive))]
                    mutant = x_1 + self.F * (x_2 - x_3) + self.F * (x_4 - self.population[i]) # Adding a component from the archive
                else:
                    mutant = x_1 + self.F * (x_2 - x_3)

                mutant = np.clip(mutant, func.bounds.lb, func.bounds.ub)

                # Crossover
                crossover = np.random.uniform(size=self.dim) < self.CR
                trial = np.where(crossover, mutant, self.population[i])

                # Selection
                f_trial = func(trial)
                self.budget -= 1

                if f_trial < self.f_opt:
                    self.f_opt = f_trial
                    self.x_opt = trial

                if f_trial < fitness[i]:
                    # Replace individual in population
                    fitness[i] = f_trial
                    self.population[i] = trial

                    # Update archive: Add trial to archive if it's not already present
                    if not any((trial == x).all() for x in self.archive):
                        self.archive.append(trial)
                        if len(self.archive) > self.archive_size:
                            self.archive.pop(np.random.randint(0, len(self.archive)))  # Remove a random element if archive is full
        return self.f_opt, self.x_opt