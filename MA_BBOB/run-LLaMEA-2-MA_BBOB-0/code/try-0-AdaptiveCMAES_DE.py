import numpy as np

class AdaptiveCMAES_DE:
    def __init__(self, budget=10000, dim=10, pop_size=20, de_mutation=0.5, de_crossover=0.7, cmaes_sigma=0.5, adaptation_rate=0.1):
        self.budget = budget
        self.dim = dim
        self.pop_size = pop_size
        self.de_mutation = de_mutation
        self.de_crossover = de_crossover
        self.cmaes_sigma = cmaes_sigma
        self.adaptation_rate = adaptation_rate
        self.population = None
        self.fitness = None
        self.x_opt = None
        self.f_opt = np.inf
        self.use_cmaes = True  # Start with CMA-ES

        # CMA-ES related variables
        self.mean = None
        self.covariance = None
        self.pc = None
        self.ps = None
        self.chiN = None
        self.C = None
        self.eigenspace = None
        self.eigenvalues = None
        self.mu = self.pop_size // 2  # Number of individuals for recombination
        self.weights = np.log(self.mu + 1/2) - np.log(np.arange(1, self.mu + 1))
        self.weights = self.weights / np.sum(self.weights)
        self.cs = (self.budget/self.dim + 4) / (self.budget/self.dim + self.mu + 4)
        self.damps = 1 + 2*max(0, np.sqrt((self.mu-1)/(self.dim+1)) - 1) + self.cs
        self.cc = (4 + self.mu / self.dim) / (self.dim + 4 + 2 * self.mu / self.dim)
        self.mucov = self.mu/(self.dim*self.dim)
        self.c1 = self.adaptation_rate / ((self.dim + 1.3)**2 + self.mucov)
        self.cmu = min(1 - self.c1, self.adaptation_rate * self.mu**2 / ((self.dim + 2)**2 + self.mucov))

    def initialize_cmaes(self, func):
        self.mean = np.random.uniform(func.bounds.lb, func.bounds.ub, size=self.dim)
        self.covariance = np.eye(self.dim)
        self.pc = np.zeros(self.dim)
        self.ps = np.zeros(self.dim)
        self.chiN = self.dim**0.5 * (1 - 1/(4*self.dim) + 1/(21*self.dim*self.dim))
        self.C = self.covariance
        self.eigenspace = np.eye(self.dim)
        self.eigenvalues = np.ones(self.dim)

    def sample_population_cmaes(self, func):
        z = np.random.normal(0, 1, size=(self.pop_size, self.dim))
        y = self.eigenspace @ (self.eigenvalues**0.5 * z.T)
        x = self.mean + self.cmaes_sigma * y.T
        x = np.clip(x, func.bounds.lb, func.bounds.ub)
        return x

    def update_cmaes(self, population, fitness):
        sorted_indices = np.argsort(fitness)
        best_indices = sorted_indices[:self.mu]

        y = (population[best_indices] - self.mean) / self.cmaes_sigma
        self.mean = np.sum(self.weights[:, None] * population[best_indices], axis=0)

        ps_temp = (1-self.cs)*self.ps + (self.cs*(2-self.cs))**0.5 * self.eigenspace @ (y.T @ self.weights)
        norm_ps = np.linalg.norm(ps_temp)
        self.ps = ps_temp
        hsig = norm_ps / (1 - (1 - self.cs)**(self.budget/self.pop_size)) / self.chiN < 1 + 2/(self.dim+1)
        self.pc = (1-self.cc)*self.pc + hsig * (self.cc*(2-self.cc))**0.5 * (population[best_indices[0]] - self.mean) / self.cmaes_sigma

        dC = np.diag(self.weights @ (y**2))
        self.C = (1-self.c1-self.cmu) * self.C + self.c1 * (self.pc[:, None] * self.pc) + self.cmu * (y.T @ np.diag(self.weights) @ y)

        self.covariance = np.triu(self.C) + np.triu(self.C, 1).T
        self.eigenvalues, self.eigenspace = np.linalg.eig(self.covariance)
        self.eigenvalues = np.real(self.eigenvalues)
        self.eigenspace = np.real(self.eigenspace)


    def differential_evolution(self, func):
        for i in range(self.pop_size):
            idxs = [idx for idx in range(self.pop_size) if idx != i]
            a, b, c = self.population[np.random.choice(idxs, 3, replace=False)]
            mutant = a + self.de_mutation * (b - c)
            mutant = np.clip(mutant, func.bounds.lb, func.bounds.ub)

            cross_points = np.random.rand(self.dim) < self.de_crossover
            if not np.any(cross_points):
                cross_points[np.random.randint(0, self.dim)] = True

            trial = np.where(cross_points, mutant, self.population[i])
            f_trial = func(trial)
            self.budget -= 1

            if f_trial < self.fitness[i]:
                self.fitness[i] = f_trial
                self.population[i] = trial

                if f_trial < self.f_opt:
                    self.f_opt = f_trial
                    self.x_opt = trial


    def __call__(self, func):
        self.population = np.random.uniform(func.bounds.lb, func.bounds.ub, size=(self.pop_size, self.dim))
        self.fitness = np.array([func(x) for x in self.population])
        self.budget -= self.pop_size

        best_index = np.argmin(self.fitness)
        self.x_opt = self.population[best_index]
        self.f_opt = self.fitness[best_index]

        self.initialize_cmaes(func)

        cmaes_success = 0
        de_success = 0

        while self.budget > 0:
            if self.use_cmaes:
                # CMA-ES Step
                new_population = self.sample_population_cmaes(func)
                new_fitness = np.array([func(x) for x in new_population])
                self.budget -= self.pop_size

                best_index_cmaes = np.argmin(new_fitness)
                if new_fitness[best_index_cmaes] < np.min(self.fitness):
                    cmaes_success += 1
                else:
                    cmaes_success = max(0, cmaes_success -1)


                self.update_cmaes(new_population, new_fitness)
                combined_population = np.concatenate((self.population, new_population), axis=0)
                combined_fitness = np.concatenate((self.fitness, new_fitness))

                sorted_indices = np.argsort(combined_fitness)[:self.pop_size]
                self.population = combined_population[sorted_indices]
                self.fitness = combined_fitness[sorted_indices]
                
                best_index = np.argmin(self.fitness)
                if self.fitness[best_index] < self.f_opt:
                    self.f_opt = self.fitness[best_index]
                    self.x_opt = self.population[best_index]



            else:
                # Differential Evolution Step
                self.differential_evolution(func)

                best_index_de = np.argmin(self.fitness)
                if self.fitness[best_index_de] < self.f_opt:
                    de_success += 1
                else:
                     de_success = max(0, de_success -1)



            # Adapt strategy
            if cmaes_success > 5 and de_success < -5:
                self.use_cmaes = True
                cmaes_success = 0
                de_success = 0
            elif de_success > 5 and cmaes_success < -5:
                self.use_cmaes = False
                cmaes_success = 0
                de_success = 0



        return self.f_opt, self.x_opt