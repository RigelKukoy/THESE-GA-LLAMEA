import numpy as np

class AdaptiveDEMix:
    def __init__(self, budget=10000, dim=10, pop_size=50, archive_size=10):
        self.budget = budget
        self.dim = dim
        self.pop_size = pop_size
        self.archive_size = archive_size
        self.F = 0.5
        self.CR = 0.9
        self.archive = []
        self.success_F = []
        self.success_CR = []
        self.success_delta_f = []
        self.p = 0.1 # probability for stochastic ranking
        self.mutation_probs = np.array([0.3, 0.3, 0.4])  # Probabilities for each mutation operator
        self.mutation_options = ['current_to_rand_1', 'rand_1', 'current_to_best_1']

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
                # Mutation Strategy Selection
                mutation_type = np.random.choice(self.mutation_options, p=self.mutation_probs)

                if mutation_type == 'current_to_rand_1':
                    idxs = np.random.choice(self.pop_size, 3, replace=False)
                    x_r1, x_r2, x_r3 = self.population[idxs]
                    mutant = self.population[i] + self.F * (x_r1 - x_r2) + self.F * (self.population[i] - x_r3)

                elif mutation_type == 'rand_1':
                    idxs = np.random.choice(self.pop_size, 3, replace=False)
                    x_r1, x_r2, x_r3 = self.population[idxs]
                    mutant = x_r1 + self.F * (x_r2 - x_r3)

                elif mutation_type == 'current_to_best_1':
                    best_idx = np.argmin(fitness)
                    x_best = self.population[best_idx]
                    idxs = np.random.choice(self.pop_size, 2, replace=False)
                    x_r1, x_r2 = self.population[idxs]
                    mutant = self.population[i] + self.F * (x_best - self.population[i]) + self.F * (x_r1 - x_r2)

                # Cauchy Mutation
                mutant += 0.01 * np.random.standard_cauchy(size=self.dim)
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
                
                # Stochastic Ranking
                if (fitness[i] < 0 and f_trial < 0) or np.random.rand() < self.p:
                    if f_trial < fitness[i]:
                        self.success_F.append(self.F)
                        self.success_CR.append(self.CR)
                        self.success_delta_f.append(np.abs(f_trial - fitness[i]))
                        
                        fitness[i] = f_trial
                        self.population[i] = trial
                        
                        if len(self.archive) < self.archive_size:
                            self.archive.append(self.population[i].copy())
                        else:
                            idx_to_replace = np.random.randint(0, self.archive_size)
                            self.archive[idx_to_replace] = self.population[i].copy()
                else:
                     if f_trial < fitness[i]:
                        fitness[i] = f_trial
                        self.population[i] = trial
                        
                        if len(self.archive) < self.archive_size:
                            self.archive.append(self.population[i].copy())
                        else:
                            idx_to_replace = np.random.randint(0, self.archive_size)
                            self.archive[idx_to_replace] = self.population[i].copy()

            # Parameter Adaptation
            if self.success_F:
                self.F = np.mean(self.success_F)
                self.CR = np.mean(self.success_CR)

                # Reset success history
                self.success_F = []
                self.success_CR = []
                self.success_delta_f = []
            
            self.F = np.clip(self.F, 0.1, 1.0)
            self.CR = np.clip(self.CR, 0.1, 1.0)

        return self.f_opt, self.x_opt