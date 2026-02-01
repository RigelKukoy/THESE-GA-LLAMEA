import numpy as np

class AdaptiveDE_Orthogonal:
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
        self.levy_exponent = 1.5 # Parameter for Levy flight

    def levy_flight(self, beta):
        sigma = (np.math.gamma(1 + beta) * np.sin(np.pi * beta / 2) / (np.math.gamma((1 + beta) / 2) * beta * (2 ** ((beta - 1) / 2)))) ** (1 / beta)
        u = np.random.randn(self.dim) * sigma
        v = np.random.randn(self.dim)
        step = u / abs(v) ** (1 / beta)
        return step

    def orthogonal_learning(self, population, fitness, num_samples=5):
        """Performs orthogonal learning to generate new candidate solutions."""
        best_idx = np.argmin(fitness)
        best_individual = population[best_idx].copy()
        
        new_candidates = []
        for _ in range(num_samples):
            # Randomly select dimensions to modify
            dims_to_modify = np.random.choice(self.dim, size=int(self.dim/2), replace=False) 
            new_candidate = best_individual.copy()
            
            # Generate random values for the selected dimensions
            new_values = np.random.uniform(low=-5.0, high=5.0, size=len(dims_to_modify))
            new_candidate[dims_to_modify] = new_values
            new_candidates.append(new_candidate)
            
        return new_candidates
        

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
                # Mutation (Cauchy)
                idxs = np.random.choice(self.pop_size, 3, replace=False)
                x_1, x_2, x_3 = self.population[idxs]
                # Cauchy mutation
                mutant = x_1 + self.F * (x_2 - x_3) + 0.01 * np.random.standard_cauchy(size=self.dim)
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

            # Orthogonal Learning
            new_candidates = self.orthogonal_learning(self.population, fitness)
            for candidate in new_candidates:
                f_candidate = func(candidate)
                self.budget -= 1

                if f_candidate < self.f_opt:
                    self.f_opt = f_candidate
                    self.x_opt = candidate

                # Replace worst individual with the new candidate if it's better
                worst_idx = np.argmax(fitness)
                if f_candidate < fitness[worst_idx]:
                    fitness[worst_idx] = f_candidate
                    self.population[worst_idx] = candidate


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