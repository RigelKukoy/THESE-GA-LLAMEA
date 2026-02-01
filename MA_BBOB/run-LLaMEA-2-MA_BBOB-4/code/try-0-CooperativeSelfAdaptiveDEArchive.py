import numpy as np

class CooperativeSelfAdaptiveDEArchive:
    def __init__(self, budget=10000, dim=10, pop_size=50, archive_size=10, F_init=0.5, CR_init=0.9):
        self.budget = budget
        self.dim = dim
        self.pop_size = pop_size
        self.archive_size = archive_size
        self.F_init = F_init
        self.CR_init = CR_init
        self.population = None
        self.fitness = None
        self.F = None
        self.CR = None
        self.f_opt = np.Inf
        self.x_opt = None
        self.eval_count = 0
        self.archive = []  # Archive to store promising solutions

    def initialize_population(self, func):
        self.population = np.random.uniform(func.bounds.lb, func.bounds.ub, size=(self.pop_size, self.dim))
        self.fitness = np.array([func(x) for x in self.population])
        self.eval_count += self.pop_size
        self.F = np.full(self.pop_size, self.F_init)
        self.CR = np.full(self.pop_size, self.CR_init)
        
        best_index = np.argmin(self.fitness)
        if self.fitness[best_index] < self.f_opt:
            self.f_opt = self.fitness[best_index]
            self.x_opt = self.population[best_index].copy()

    def evolve(self, func):
        successful_F = []
        successful_CR = []

        for i in range(self.pop_size):
            # Parameter adaptation
            self.F[i] = np.clip(np.random.normal(self.F[i], 0.1), 0.1, 1.0)
            self.CR[i] = np.clip(np.random.normal(self.CR[i], 0.1), 0.1, 1.0)
            
            # Mutation - Cooperative strategy: use archive if available
            if len(self.archive) > 0 and np.random.rand() < 0.2:  # 20% chance to use archive
                x_r1 = self.archive[np.random.randint(len(self.archive))]
                idxs = np.random.choice(self.pop_size, 2, replace=False)
                x_r2, x_r3 = self.population[idxs]
            else:
                idxs = np.random.choice(self.pop_size, 3, replace=False)
                x_r1, x_r2, x_r3 = self.population[idxs]
            
            x_mutated = x_r1 + self.F[i] * (x_r2 - x_r3)
            x_mutated = np.clip(x_mutated, func.bounds.lb, func.bounds.ub)

            # Crossover
            x_trial = self.population[i].copy()
            j_rand = np.random.randint(self.dim)
            for j in range(self.dim):
                if np.random.rand() < self.CR[i] or j == j_rand:
                    x_trial[j] = x_mutated[j]

            # Selection
            f_trial = func(x_trial)
            self.eval_count += 1

            if f_trial < self.fitness[i]:
                successful_F.append(self.F[i])
                successful_CR.append(self.CR[i])
                
                # Update archive
                if len(self.archive) < self.archive_size:
                    self.archive.append(x_trial)
                else:
                    # Replace a random element in the archive
                    self.archive[np.random.randint(self.archive_size)] = x_trial
                    
                self.population[i] = x_trial
                self.fitness[i] = f_trial
                
                if f_trial < self.f_opt:
                    self.f_opt = f_trial
                    self.x_opt = x_trial.copy()

            if self.eval_count >= self.budget:
                break
        
        # Update F and CR based on successful values from the generation
        if successful_F:
            self.F = np.full(self.pop_size, np.mean(successful_F))
        if successful_CR:
            self.CR = np.full(self.pop_size, np.mean(successful_CR))
                
    def __call__(self, func):
        self.initialize_population(func)
        while self.eval_count < self.budget:
            self.evolve(func)
        return self.f_opt, self.x_opt