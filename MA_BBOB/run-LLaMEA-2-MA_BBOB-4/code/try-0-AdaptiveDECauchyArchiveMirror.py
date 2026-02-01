import numpy as np

class AdaptiveDECauchyArchiveMirror:
    def __init__(self, budget=10000, dim=10, pop_size=50, CR_init=0.5, F_init=0.7, archive_size=10):
        self.budget = budget
        self.dim = dim
        self.pop_size = pop_size
        self.CR_init = CR_init
        self.F_init = F_init
        self.archive_size = archive_size
        self.population = None
        self.fitness = None
        self.CR = None
        self.F = None
        self.f_opt = np.Inf
        self.x_opt = None
        self.eval_count = 0
        self.archive = []  # Archive for storing solutions that are worse than their parents

    def initialize_population(self, func):
        self.population = np.random.uniform(func.bounds.lb, func.bounds.ub, size=(self.pop_size, self.dim))
        self.fitness = np.array([func(x) for x in self.population])
        self.eval_count += self.pop_size
        self.CR = np.full(self.pop_size, self.CR_init)
        self.F = np.full(self.pop_size, self.F_init)
        
        best_index = np.argmin(self.fitness)
        if self.fitness[best_index] < self.f_opt:
            self.f_opt = self.fitness[best_index]
            self.x_opt = self.population[best_index].copy()

    def evolve(self, func):
        for i in range(self.pop_size):
            # Mutation (Cauchy)
            idxs = np.random.choice(self.pop_size, 3, replace=False)
            x_r1, x_r2, x_r3 = self.population[idxs]
            
            # Adaptive F
            self.F[i] = np.clip(np.random.normal(self.F[i], 0.1), 0.1, 1.0)
            
            x_mutated = x_r1 + self.F[i] * (x_r2 - x_r3) + np.random.standard_cauchy(size=self.dim) # Adding Cauchy noise
            x_mutated = np.clip(x_mutated, func.bounds.lb, func.bounds.ub)

            # Crossover
            x_trial = self.population[i].copy()
            j_rand = np.random.randint(self.dim)
            CR = np.clip(np.random.normal(self.CR[i], 0.1), 0.1, 1.0)  # Self-adjusting CR
            for j in range(self.dim):
                if np.random.rand() < CR or j == j_rand:
                    x_trial[j] = x_mutated[j]

            # Mirrored Sampling
            best_index = np.argmin(self.fitness)
            x_best = self.population[best_index]
            x_mirrored = 2 * x_best - x_trial
            x_mirrored = np.clip(x_mirrored, func.bounds.lb, func.bounds.ub)

            # Selection
            f_trial = func(x_trial)
            f_mirrored = func(x_mirrored)
            self.eval_count += 2

            if f_trial < self.fitness[i] and f_trial <= f_mirrored:
                self.CR[i] = CR  # Update CR of individual
                self.population[i] = x_trial
                self.fitness[i] = f_trial
                if f_trial < self.f_opt:
                    self.f_opt = f_trial
                    self.x_opt = x_trial.copy()
            elif f_mirrored < self.fitness[i] and f_mirrored < f_trial:
                self.CR[i] = CR
                self.population[i] = x_mirrored
                self.fitness[i] = f_mirrored
                if f_mirrored < self.f_opt:
                    self.f_opt = f_mirrored
                    self.x_opt = x_mirrored.copy()
            else:
                # Archive the parent solution
                if len(self.archive) < self.archive_size:
                    self.archive.append(self.population[i].copy())
                else:
                    # Replace a random element in the archive
                    idx_to_replace = np.random.randint(self.archive_size)
                    self.archive[idx_to_replace] = self.population[i].copy()
                    
            if self.eval_count >= self.budget:
                break
            
            # Use information from archive: restart if necessary
            if len(self.archive) > 0 and np.random.rand() < 0.05:  # Small probability to use the archive
                archived_idx = np.random.randint(len(self.archive))
                self.population[i] = self.archive[archived_idx].copy()
                self.fitness[i] = func(self.population[i])
                self.eval_count += 1
                if self.fitness[i] < self.f_opt:
                    self.f_opt = self.fitness[i]
                    self.x_opt = self.population[i].copy()
                
    def __call__(self, func):
        self.initialize_population(func)
        while self.eval_count < self.budget:
            self.evolve(func)
        return self.f_opt, self.x_opt