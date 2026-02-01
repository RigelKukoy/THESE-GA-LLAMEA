import numpy as np

class AdaptiveLocalSearchDE:
    def __init__(self, budget=10000, dim=10, pop_size=50, F=0.5, CR=0.9, local_search_prob=0.1, local_search_radius=0.1):
        self.budget = budget
        self.dim = dim
        self.pop_size = pop_size
        self.F = F
        self.CR = CR
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

        while self.budget > 0:
            # Calculate population diversity (standard deviation along each dimension)
            diversity = np.std(self.population, axis=0)
            # Adapt local search radius based on population diversity
            adaptive_radius = self.local_search_radius * np.mean(diversity)

            for i in range(self.pop_size):
                # Mutation
                idxs = np.random.choice(self.pop_size, 3, replace=False)
                x_1, x_2, x_3 = self.population[idxs]
                mutant = x_1 + self.F * (x_2 - x_3)
                mutant = np.clip(mutant, func.bounds.lb, func.bounds.ub)

                # Crossover
                crossover = np.random.uniform(size=self.dim) < self.CR
                trial = np.where(crossover, mutant, self.population[i])

                # Local Search
                if np.random.rand() < self.local_search_prob:
                    trial_local = trial + np.random.uniform(-adaptive_radius, adaptive_radius, size=self.dim)
                    trial_local = np.clip(trial_local, func.bounds.lb, func.bounds.ub)

                    f_trial_local = func(trial_local)
                    self.budget -= 1

                    if f_trial_local < self.f_opt:
                        self.f_opt = f_trial_local
                        self.x_opt = trial_local

                    if f_trial_local < fitness[i]:
                        fitness[i] = f_trial_local
                        self.population[i] = trial_local
                        trial = trial_local
                    else:
                        f_trial = func(trial)
                        self.budget -= 1
                        if f_trial < self.f_opt:
                            self.f_opt = f_trial
                            self.x_opt = trial

                        if f_trial < fitness[i]:
                            fitness[i] = f_trial
                            self.population[i] = trial


                else:

                    f_trial = func(trial)
                    self.budget -= 1

                    if f_trial < self.f_opt:
                        self.f_opt = f_trial
                        self.x_opt = trial

                    if f_trial < fitness[i]:
                        fitness[i] = f_trial
                        self.population[i] = trial



        return self.f_opt, self.x_opt