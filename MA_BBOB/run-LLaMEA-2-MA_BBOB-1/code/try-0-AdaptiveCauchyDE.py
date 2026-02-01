import numpy as np

class AdaptiveCauchyDE:
    def __init__(self, budget=10000, dim=10, pop_size=50, F=0.5, CR=0.7, lr_F=0.1, lr_CR=0.1, shrink_factor=0.99):
        self.budget = budget
        self.dim = dim
        self.pop_size = pop_size
        self.F = F
        self.CR = CR
        self.lr_F = lr_F
        self.lr_CR = lr_CR
        self.shrink_factor = shrink_factor
        self.population = None
        self.fitness = None
        self.best_index = None
        self.f_opt = np.inf
        self.x_opt = None
        self.success_F = []
        self.success_CR = []
        self.archive = []

    def initialize_population(self, func):
        self.population = np.random.uniform(func.bounds.lb, func.bounds.ub, size=(self.pop_size, self.dim))
        self.fitness = np.array([func(x) for x in self.population])
        self.budget -= self.pop_size
        self.best_index = np.argmin(self.fitness)
        self.f_opt = self.fitness[self.best_index]
        self.x_opt = self.population[self.best_index].copy()
        self.lb = func.bounds.lb
        self.ub = func.bounds.ub

    def cauchy_mutation(self, x_r1, x_r2, F):
        return x_r1 + F * np.random.standard_cauchy(size=x_r1.shape) * (x_r1 - x_r2)

    def __call__(self, func):
        self.initialize_population(func)

        while self.budget > 0:
            successful_mutations = 0
            temp_success_F = []
            temp_success_CR = []

            for i in range(self.pop_size):
                # Mutation
                indices = np.random.choice(self.pop_size, 3, replace=False)
                x_r1, x_r2, x_r3 = self.population[indices]

                # Generate trial F and CR
                trial_F = np.clip(np.random.normal(self.F, 0.1), 0.1, 1.0)
                trial_CR = np.clip(np.random.normal(self.CR, 0.1), 0.1, 1.0)

                # Cauchy mutation
                v = self.cauchy_mutation(x_r1, x_r2, trial_F)
                v = np.clip(v, self.lb, self.ub)


                # Crossover
                j_rand = np.random.randint(self.dim)
                u = self.population[i].copy()
                for j in range(self.dim):
                    if np.random.rand() < trial_CR or j == j_rand:
                        u[j] = v[j]

                # Evaluation
                f_u = func(u)
                self.budget -= 1

                # Selection
                if f_u < self.fitness[i]:
                    successful_mutations += 1
                    temp_success_F.append(trial_F)
                    temp_success_CR.append(trial_CR)

                    self.fitness[i] = f_u
                    self.population[i] = u
                    if f_u < self.f_opt:
                        self.f_opt = f_u
                        self.x_opt = u.copy()
                else:
                    self.archive.append(u)

            # Adapt F and CR
            if temp_success_F:
                self.F = (1 - self.lr_F) * self.F + self.lr_F * np.mean(temp_success_F)
            if temp_success_CR:
                self.CR = (1 - self.lr_CR) * self.CR + self.lr_CR * np.mean(temp_success_CR)
            
            # Shrink the search space
            self.lb = self.shrink_factor * (self.lb - self.x_opt) + self.x_opt
            self.ub = self.shrink_factor * (self.ub - self.x_opt) + self.x_opt
            self.population = np.clip(self.population, self.lb, self.ub)


        return self.f_opt, self.x_opt