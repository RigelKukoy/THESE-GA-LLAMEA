import numpy as np

class CooperativeDEArchiveAging:
    def __init__(self, budget=10000, dim=10, popsize=None, num_populations=3, CR=0.7, F=0.5, archive_size=50, age_limit=50):
        self.budget = budget
        self.dim = dim
        self.popsize = popsize if popsize is not None else 10 * self.dim
        self.num_populations = num_populations
        self.CR = CR
        self.F = F
        self.archive_size = archive_size
        self.age_limit = age_limit
        self.populations = []
        self.fitnesses = []
        self.ages = []
        self.archive = []  # Store promising solutions
        self.archive_fitness = []
        self.archive_ages = []
        self.eval_count = 0
        self.f_opt = np.Inf
        self.x_opt = None

    def __call__(self, func):
        lb = func.bounds.lb
        ub = func.bounds.ub

        # Initialize multiple populations
        for _ in range(self.num_populations):
            population = np.random.uniform(lb, ub, size=(self.popsize, self.dim))
            self.populations.append(population)
            fitness = np.array([func(x) for x in population])
            self.fitnesses.append(fitness)
            self.ages.append(np.zeros(self.popsize, dtype=int))  # Initialize ages
            self.eval_count += self.popsize

            # Update global best
            local_best_idx = np.argmin(fitness)
            if fitness[local_best_idx] < self.f_opt:
                self.f_opt = fitness[local_best_idx]
                self.x_opt = population[local_best_idx]

        while self.eval_count < self.budget:
            for pop_idx in range(self.num_populations):
                for i in range(self.popsize):
                    # Mutation: Cooperative strategy - select from other populations and archive
                    pool = [p for idx, p in enumerate(self.populations) if idx != pop_idx]
                    if self.archive:
                         pool.append(np.array(self.archive))  # Add archive to the pool

                    if len(pool) > 0:
                        chosen_population = np.random.choice(len(pool))
                        if chosen_population < len(self.populations) -1:
                            idxs = np.random.choice(self.popsize, 3, replace=False)
                            x1, x2, x3 = self.populations[pop_idx][idxs]
                            xp = pool[chosen_population][np.random.randint(0,self.popsize)]
                            mutant = x1 + self.F * (x2 - x3) #+ self.F * (xp - self.populations[pop_idx][i])
                            #mutant = x1 + self.F * (x2 - x3) #+ np.random.normal(0,0.1,self.dim)
                        else:
                            idxs = np.random.choice(self.popsize, 2, replace=False)
                            x1, x2 = self.populations[pop_idx][idxs]
                            archive_idx = np.random.randint(0, len(self.archive))
                            mutant = x1 + self.F * (x2 - self.archive[archive_idx])
                    else: #Fallback if no other populations or archive exist
                        idxs = np.random.choice(self.popsize, 3, replace=False)
                        x1, x2, x3 = self.populations[pop_idx][idxs]
                        mutant = x1 + self.F * (x2 - x3)
                    mutant = np.clip(mutant, lb, ub)
    
                    # Crossover
                    crossover_mask = np.random.rand(self.dim) < self.CR
                    trial = np.where(crossover_mask, mutant, self.populations[pop_idx][i])
    
                    # Selection
                    f_trial = func(trial)
                    self.eval_count += 1
    
                    if f_trial < self.fitnesses[pop_idx][i]:
                        self.populations[pop_idx][i] = trial
                        self.fitnesses[pop_idx][i] = f_trial
                        self.ages[pop_idx][i] = 0  # Reset age
    
                        if f_trial < self.f_opt:
                            self.f_opt = f_trial
                            self.x_opt = trial

                        # Archive update: add if better than worst in archive
                        if len(self.archive) < self.archive_size:
                            self.archive.append(trial)
                            self.archive_fitness.append(f_trial)
                            self.archive_ages.append(0)
                        else:
                            worst_archive_idx = np.argmax(self.archive_fitness)
                            if f_trial < self.archive_fitness[worst_archive_idx]:
                                self.archive[worst_archive_idx] = trial
                                self.archive_fitness[worst_archive_idx] = f_trial
                                self.archive_ages[worst_archive_idx] = 0
                    else:
                        self.ages[pop_idx][i] += 1  # Increase age
    
                # Aging: Replace individuals that have reached the age limit
                for i in range(self.popsize):
                    if self.ages[pop_idx][i] > self.age_limit:
                        self.populations[pop_idx][i] = np.random.uniform(lb, ub, size=self.dim)
                        self.fitnesses[pop_idx][i] = func(self.populations[pop_idx][i])
                        self.eval_count += 1
                        self.ages[pop_idx][i] = 0
                        if self.fitnesses[pop_idx][i] < self.f_opt:
                            self.f_opt = self.fitnesses[pop_idx][i]
                            self.x_opt = self.populations[pop_idx][i]

                # Archive aging: Increase age of archive members
                for j in range(len(self.archive)):
                    self.archive_ages[j] += 1
                
                # Remove old archive members:
                to_remove = [j for j in range(len(self.archive)) if self.archive_ages[j] > self.age_limit]
                for j in sorted(to_remove, reverse=True):
                    del self.archive[j]
                    del self.archive_fitness[j]
                    del self.archive_ages[j]
        return self.f_opt, self.x_opt