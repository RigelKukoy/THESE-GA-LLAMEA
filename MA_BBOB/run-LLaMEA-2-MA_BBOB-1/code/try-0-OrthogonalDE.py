import numpy as np

class OrthogonalDE:
    def __init__(self, budget=10000, dim=10, pop_size=None, F=0.5, CR=0.7, ortho_trials=3, pop_size_factor=2, adaptation_rate=0.1):
        self.budget = budget
        self.dim = dim
        self.pop_size = pop_size if pop_size is not None else int(4 + 3 * np.log(dim))
        self.F = F
        self.CR = CR
        self.ortho_trials = ortho_trials
        self.pop_size_factor = pop_size_factor
        self.adaptation_rate = adaptation_rate
        self.population = None
        self.fitness = None
        self.best_index = None
        self.f_opt = np.inf
        self.x_opt = None
        self.archive = []
        self.historical_best_rate = 0.1

    def initialize_population(self, func):
        self.population = np.random.uniform(func.bounds.lb, func.bounds.ub, size=(self.pop_size, self.dim))
        self.fitness = np.array([func(x) for x in self.population])
        self.budget -= self.pop_size
        self.best_index = np.argmin(self.fitness)
        self.f_opt = self.fitness[self.best_index]
        self.x_opt = self.population[self.best_index].copy()

    def orthogonal_learning(self, func, mutant):
        best_fitness = np.inf
        best_mutant = None
        for _ in range(self.ortho_trials):
            orthogonal_mutant = mutant.copy()
            j = np.random.randint(self.dim)
            orthogonal_mutant[j] = np.random.uniform(func.bounds.lb, func.bounds.ub)
            orthogonal_mutant = np.clip(orthogonal_mutant, func.bounds.lb, func.bounds.ub)
            fitness = func(orthogonal_mutant)
            self.budget -= 1
            if fitness < best_fitness:
                best_fitness = fitness
                best_mutant = orthogonal_mutant
        return best_mutant, best_fitness

    def adaptive_mutation_scaling(self):
        self.F = np.clip(np.random.normal(0.5, 0.3), 0.1, 1.0)
        self.CR = np.clip(np.random.normal(0.7, 0.2), 0.1, 1.0)

    def historical_best_mutation(self, x_target):
        if len(self.archive) > 0 and np.random.rand() < self.historical_best_rate:
            x_best_historical = self.archive[np.argmin([func(x) for x in self.archive])]
            return self.F * (x_best_historical - x_target)
        else:
            return 0.0

    def dynamic_population_size(self, improvement_rate):
        if improvement_rate > 0.1:
            self.pop_size = int(self.pop_size * (1 + self.adaptation_rate))
        elif improvement_rate < 0.01:
            self.pop_size = int(self.pop_size * (1 - self.adaptation_rate))
        self.pop_size = max(4, min(self.pop_size, self.dim * self.pop_size_factor))

    def __call__(self, func):
        self.initialize_population(func)
        generation = 0

        while self.budget > 0:
            generation += 1
            improvement_count = 0

            for i in range(self.pop_size):
                # Mutation
                indices = np.random.choice(self.pop_size, 3, replace=False)
                x_r1, x_r2, x_r3 = self.population[indices]

                mutation_vector = self.F * (x_r2 - x_r3) + self.historical_best_mutation(self.population[i])
                v = self.population[i] + mutation_vector
                v = np.clip(v, func.bounds.lb, func.bounds.ub)

                # Crossover
                j_rand = np.random.randint(self.dim)
                u = self.population[i].copy()
                for j in range(self.dim):
                    if np.random.rand() < self.CR or j == j_rand:
                        u[j] = v[j]

                # Orthogonal Learning
                u, f_u = self.orthogonal_learning(func, u)

                # Selection
                if f_u < self.fitness[i]:
                    improvement_count += 1
                    self.fitness[i] = f_u
                    self.population[i] = u
                    if f_u < self.f_opt:
                        self.f_opt = f_u
                        self.x_opt = u.copy()
                        if len(self.archive) < 100:
                            self.archive.append(u.copy())
                        else:
                            self.archive[np.random.randint(100)] = u.copy()

            # Adaptive F and CR
            self.adaptive_mutation_scaling()

            # Dynamic Population Size
            improvement_rate = improvement_count / self.pop_size
            self.dynamic_population_size(improvement_rate)
            self.pop_size = int(self.pop_size)

            if self.pop_size != self.population.shape[0]:
                # Resize the population (either by random initialization or truncation)
                if self.pop_size > self.population.shape[0]:
                    new_individuals = np.random.uniform(func.bounds.lb, func.bounds.ub, size=(self.pop_size - self.population.shape[0], self.dim))
                    new_fitness = np.array([func(x) for x in new_individuals])
                    self.budget -= new_individuals.shape[0]
                    self.population = np.vstack((self.population, new_individuals))
                    self.fitness = np.concatenate((self.fitness, new_fitness))
                else:
                    #Truncate population
                    indices_to_keep = np.argsort(self.fitness)[:self.pop_size]
                    self.population = self.population[indices_to_keep]
                    self.fitness = self.fitness[indices_to_keep]
        return self.f_opt, self.x_opt