import numpy as np

class AdaptiveDEGaussianLocalSearch:
    def __init__(self, budget=10000, dim=10, pop_size=40, F=0.5, CR=0.7, local_search_prob=0.1, local_search_sigma=0.1):
        self.budget = budget
        self.dim = dim
        self.pop_size = pop_size
        self.F = F
        self.CR = CR
        self.local_search_prob = local_search_prob
        self.local_search_sigma = local_search_sigma
        self.population = None
        self.fitness = None
        self.x_opt = None
        self.f_opt = np.inf

    def __call__(self, func):
        self.population = np.random.uniform(func.bounds.lb, func.bounds.ub, size=(self.pop_size, self.dim))
        self.fitness = np.array([func(x) for x in self.population])
        self.budget -= self.pop_size

        best_index = np.argmin(self.fitness)
        self.x_opt = self.population[best_index]
        self.f_opt = self.fitness[best_index]

        while self.budget > 0:
            for i in range(self.pop_size):
                # Differential Evolution mutation
                idxs = np.random.choice(self.pop_size, 3, replace=False)
                x_r1, x_r2, x_r3 = self.population[idxs]
                
                # Mutation
                v_trial = self.population[i] + self.F * (x_r2 - x_r3)
                
                # Crossover
                u_trial = np.zeros(self.dim)
                for j in range(self.dim):
                    if np.random.rand() < self.CR or j == np.random.randint(self.dim):
                        u_trial[j] = v_trial[j]
                    else:
                        u_trial[j] = self.population[i][j]
                
                u_trial = np.clip(u_trial, func.bounds.lb, func.bounds.ub)

                # Local Search around best solution
                if np.random.rand() < self.local_search_prob:
                    mutation = np.random.normal(0, self.local_search_sigma, size=self.dim)
                    u_trial = self.x_opt + mutation
                    u_trial = np.clip(u_trial, func.bounds.lb, func.bounds.ub)

                f_new = func(u_trial)
                self.budget -= 1
                
                if f_new < self.f_opt:
                    self.f_opt = f_new
                    self.x_opt = u_trial

                if f_new < self.fitness[i]:
                    self.fitness[i] = f_new
                    self.population[i] = u_trial

            best_index = np.argmin(self.fitness)
            if self.fitness[best_index] < self.f_opt:
                self.f_opt = self.fitness[best_index]
                self.x_opt = self.population[best_index]

        return self.f_opt, self.x_opt