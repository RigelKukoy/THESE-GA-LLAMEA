import numpy as np
from scipy.optimize import minimize

class SelfAdaptiveDE_NM:
    def __init__(self, budget=10000, dim=10, pop_size=50, CR_init=0.5, F_init=0.5, success_rate_memory=10, stagnation_limit=50):
        self.budget = budget
        self.dim = dim
        self.pop_size = pop_size
        self.CR = CR_init
        self.F = F_init
        self.success_rate_memory = success_rate_memory
        self.stagnation_limit = stagnation_limit
        self.population = None
        self.fitness = None
        self.best_fitness_history = []
        self.successful_CRs = []
        self.successful_Fs = []
        self.stagnation_counter = 0
        self.f_opt = np.inf
        self.x_opt = None

    def initialize_population(self, func):
        self.population = np.random.uniform(func.bounds.lb, func.bounds.ub, size=(self.pop_size, self.dim))
        self.fitness = np.array([func(x) for x in self.population])
        self.budget -= self.pop_size
        self.f_opt = np.min(self.fitness)
        self.x_opt = self.population[np.argmin(self.fitness)]
        self.best_fitness_history.append(self.f_opt)

    def mutate(self, x_i):
        indices = np.random.choice(self.pop_size, size=3, replace=False)
        x_r1 = self.population[indices[0]]
        x_r2 = self.population[indices[1]]
        x_r3 = self.population[indices[2]]
        return x_r1 + self.F * (x_r2 - x_r3)

    def crossover(self, x_i, v_i):
        u_i = np.copy(x_i)
        j_rand = np.random.randint(self.dim)
        for j in range(self.dim):
            if np.random.rand() <= self.CR or j == j_rand:
                u_i[j] = v_i[j]
        return u_i

    def repair(self, x, func):
        return np.clip(x, func.bounds.lb, func.bounds.ub)

    def adjust_parameters(self):
        if self.successful_CRs:
            self.CR = np.mean(self.successful_CRs)
            self.successful_CRs = []
        if self.successful_Fs:
            self.F = np.mean(self.successful_Fs)
            self.successful_Fs = []
        self.F = np.clip(self.F, 0.1, 1.0)
        self.CR = np.clip(self.CR, 0.1, 0.9)

    def check_stagnation(self):
        if len(self.best_fitness_history) > self.stagnation_limit:
            if np.abs(self.best_fitness_history[-1] - np.mean(self.best_fitness_history[-self.stagnation_limit:])) < 1e-6:
                self.stagnation_counter += 1
            else:
                self.stagnation_counter = 0

        if self.stagnation_counter >= self.stagnation_limit:
            return True
        else:
            return False

    def local_search(self, func, x_start):
        bounds = [(func.bounds.lb, func.bounds.ub)] * self.dim
        res = minimize(func, x_start, method='Nelder-Mead', bounds=bounds, options={'maxfev': self.budget // 100})  # Limit FE
        self.budget -= res.nfev
        return res.fun, res.x

    def __call__(self, func):
        self.initialize_population(func)

        while self.budget > 0:
            self.adjust_parameters()

            for i in range(self.pop_size):
                # Mutation
                v_i = self.mutate(self.population[i])

                # Crossover
                u_i = self.crossover(self.population[i], v_i)

                # Repair
                u_i = self.repair(u_i, func)

                # Selection
                f_u_i = func(u_i)
                self.budget -= 1

                if f_u_i < self.fitness[i]:
                    self.successful_CRs.append(self.CR)
                    self.successful_Fs.append(self.F)
                    self.population[i] = u_i
                    self.fitness[i] = f_u_i

                    if f_u_i < self.f_opt:
                        self.f_opt = f_u_i
                        self.x_opt = u_i

            best_fitness = np.min(self.fitness)
            self.best_fitness_history.append(best_fitness)

            if best_fitness < self.f_opt:
                self.f_opt = best_fitness
                self.x_opt = self.population[np.argmin(self.fitness)]

            if self.check_stagnation():
                # Perform local search on the best individual
                f_local, x_local = self.local_search(func, self.x_opt)
                if f_local < self.f_opt:
                    self.f_opt = f_local
                    self.x_opt = x_local
                self.stagnation_counter = 0

            if self.budget <= 0:
                break

        return self.f_opt, self.x_opt