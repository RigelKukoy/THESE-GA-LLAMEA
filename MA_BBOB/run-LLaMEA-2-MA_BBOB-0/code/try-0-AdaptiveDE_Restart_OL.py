import numpy as np

class AdaptiveDE_Restart_OL:
    def __init__(self, budget=10000, dim=10, pop_size=50, archive_size=10, explore_ratio=0.5, restart_patience=50):
        self.budget = budget
        self.dim = dim
        self.pop_size = pop_size
        self.archive_size = archive_size
        self.explore_ratio = explore_ratio
        self.restart_patience = restart_patience
        self.F = 0.5
        self.CR = 0.9
        self.archive = []
        self.success_F = []
        self.success_CR = []
        self.success_delta_f = []
        self.p = 0.1  # probability for stochastic ranking
        self.explore_pop_size = int(self.pop_size * self.explore_ratio)
        self.exploit_pop_size = self.pop_size - self.explore_pop_size
        self.restart_counter = 0
        self.best_fitness_history = []

    def orthogonal_design(self, n, k):
        # Generate an orthogonal array using a simple method.  Can be replaced with a more robust OA generator.
        H = np.zeros((n, k))
        for i in range(n):
            for j in range(k):
                H[i, j] = (i * j) % n
        return H

    def __call__(self, func):
        self.f_opt = np.Inf
        self.x_opt = None

        # Initialize exploration and exploitation populations
        self.explore_population = np.random.uniform(func.bounds.lb, func.bounds.ub, size=(self.explore_pop_size, self.dim))
        self.exploit_population = np.random.uniform(func.bounds.lb, func.bounds.ub, size=(self.exploit_pop_size, self.dim))
        self.population = np.concatenate((self.explore_population, self.exploit_population), axis=0)
        
        fitness = np.array([func(x) for x in self.population])
        self.budget -= self.pop_size

        for i in range(self.pop_size):
            if fitness[i] < self.f_opt:
                self.f_opt = fitness[i]
                self.x_opt = self.population[i]
                self.best_fitness_history.append(self.f_opt)

        while self.budget > 0:
            # Adaptive DE for Exploration Population
            for i in range(self.explore_pop_size):
                idxs = np.random.choice(self.explore_pop_size, 3, replace=False)
                x_1, x_2, x_3 = self.explore_population[idxs]
                mutant = x_1 + self.F * (x_2 - x_3)
                mutant = np.clip(mutant, func.bounds.lb, func.bounds.ub)

                crossover = np.random.uniform(size=self.dim) < self.CR
                trial = np.where(crossover, mutant, self.explore_population[i])

                f_trial = func(trial)
                self.budget -= 1

                if f_trial < self.f_opt:
                    self.f_opt = f_trial
                    self.x_opt = trial
                    self.best_fitness_history.append(self.f_opt)

                if (fitness[i] < 0 and f_trial < 0) or np.random.rand() < self.p:
                    if f_trial < fitness[i]:
                        self.success_F.append(self.F)
                        self.success_CR.append(self.CR)
                        self.success_delta_f.append(np.abs(f_trial - fitness[i]))

                        fitness[i] = f_trial
                        self.explore_population[i] = trial
                else:
                    if f_trial < fitness[i]:
                        fitness[i] = f_trial
                        self.explore_population[i] = trial

            # Adaptive DE for Exploitation Population
            for i in range(self.exploit_pop_size):
                idxs = np.random.choice(self.pop_size, 3, replace=False) #draw from whole pop
                x_1, x_2, x_3 = self.population[idxs]
                mutant = x_1 + self.F * (x_2 - x_3)
                mutant = np.clip(mutant, func.bounds.lb, func.bounds.ub)

                crossover = np.random.uniform(size=self.dim) < self.CR
                trial = np.where(crossover, mutant, self.exploit_population[i]) #exploit population

                f_trial = func(trial)
                self.budget -= 1

                if f_trial < self.f_opt:
                    self.f_opt = f_trial
                    self.x_opt = trial
                    self.best_fitness_history.append(self.f_opt)

                if (fitness[i + self.explore_pop_size] < 0 and f_trial < 0) or np.random.rand() < self.p:
                    if f_trial < fitness[i + self.explore_pop_size]:
                        self.success_F.append(self.F)
                        self.success_CR.append(self.CR)
                        self.success_delta_f.append(np.abs(f_trial - fitness[i + self.explore_pop_size]))

                        fitness[i + self.explore_pop_size] = f_trial
                        self.exploit_population[i] = trial
                else:
                    if f_trial < fitness[i + self.explore_pop_size]:
                        fitness[i + self.explore_pop_size] = f_trial
                        self.exploit_population[i] = trial
            
            self.population = np.concatenate((self.explore_population, self.exploit_population), axis=0)
            
            # Parameter Adaptation
            if self.success_F:
                self.F = np.mean(self.success_F)
                self.CR = np.mean(self.success_CR)
                self.success_F = []
                self.success_CR = []
                self.success_delta_f = []

            self.F = np.clip(self.F, 0.1, 1.0)
            self.CR = np.clip(self.CR, 0.1, 1.0)
            
            # Restart Mechanism
            if len(self.best_fitness_history) > self.restart_patience:
                if self.best_fitness_history[-1] >= self.best_fitness_history[-self.restart_patience]:
                    self.restart_counter += 1
                    if self.restart_counter >= self.restart_patience:
                        # Restart: Re-initialize exploration population
                        self.explore_population = np.random.uniform(func.bounds.lb, func.bounds.ub, size=(self.explore_pop_size, self.dim))
                        self.population = np.concatenate((self.explore_population, self.exploit_population), axis=0)
                        fitness = np.array([func(x) for x in self.population])
                        
                        for i in range(self.pop_size):
                            if fitness[i] < self.f_opt:
                                self.f_opt = fitness[i]
                                self.x_opt = self.population[i]
                                self.best_fitness_history.append(self.f_opt)

                        self.restart_counter = 0
                else:
                    self.restart_counter = 0
            
            # Orthogonal Learning:  Apply to the exploitation population.  Uses evaluations.
            if self.budget > self.dim * (self.exploit_pop_size + 1):
                oa = self.orthogonal_design(self.exploit_pop_size, self.dim)
                for j in range(self.dim):
                    vals = np.linspace(func.bounds.lb, func.bounds.ub, self.exploit_pop_size)
                    for i in range(self.exploit_pop_size):
                         trial = self.exploit_population[i].copy()
                         trial[j] = vals[int(oa[i,j])]
                         f_trial = func(trial)
                         self.budget -= 1
                         if f_trial < self.f_opt:
                            self.f_opt = f_trial
                            self.x_opt = trial
                            self.best_fitness_history.append(self.f_opt)
                         if f_trial < fitness[i + self.explore_pop_size]:
                             fitness[i + self.explore_pop_size] = f_trial
                             self.exploit_population[i] = trial
                                          
        return self.f_opt, self.x_opt