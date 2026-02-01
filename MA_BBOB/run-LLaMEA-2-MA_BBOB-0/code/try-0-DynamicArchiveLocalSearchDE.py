import numpy as np

class DynamicArchiveLocalSearchDE:
    def __init__(self, budget=10000, dim=10, pop_size=50, archive_size_init=10, F=0.5, CR=0.9, local_search_prob=0.1, local_search_radius=0.1):
        self.budget = budget
        self.dim = dim
        self.pop_size = pop_size
        self.archive_size_init = archive_size_init
        self.archive_size = archive_size_init
        self.F = F
        self.CR = CR
        self.archive = []
        self.local_search_prob = local_search_prob
        self.local_search_radius = local_search_radius

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

        generation = 0
        while self.budget > 0:
            generation += 1
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
            
            # Local Search: Apply local search to the best solution with a certain probability
            if np.random.rand() < self.local_search_prob:
                x_local_search = self.x_opt + np.random.uniform(-self.local_search_radius, self.local_search_radius, size=self.dim)
                x_local_search = np.clip(x_local_search, func.bounds.lb, func.bounds.ub)
                f_local_search = func(x_local_search)
                self.budget -= 1

                if f_local_search < self.f_opt:
                    self.f_opt = f_local_search
                    self.x_opt = x_local_search

            # Adjust archive size dynamically
            if generation % 10 == 0:
                if self.f_opt == np.min(fitness):
                     self.archive_size = min(self.archive_size + 1, self.pop_size)
                else:
                    self.archive_size = max(self.archive_size - 1, self.archive_size_init) #ensure archive size doesn't become zero.
                    
        return self.f_opt, self.x_opt