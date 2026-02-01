import numpy as np

class AdaptiveDEArchive:
    def __init__(self, budget=10000, dim=10, pop_size=50, archive_size=20, CR_init=0.5, F=0.5, CR_adapt_rate=0.1):
        self.budget = budget
        self.dim = dim
        self.pop_size = pop_size
        self.archive_size = archive_size
        self.CR = CR_init
        self.F = F
        self.CR_adapt_rate = CR_adapt_rate
        self.population = None
        self.fitness = None
        self.archive = []
        self.archive_fitness = []
        self.f_opt = np.Inf
        self.x_opt = None

    def initialize_population(self, func):
        self.population = np.random.uniform(func.bounds.lb, func.bounds.ub, size=(self.pop_size, self.dim))
        self.fitness = np.array([func(x) for x in self.population])
        self.budget -= self.pop_size
        self.f_opt = np.min(self.fitness)
        self.x_opt = self.population[np.argmin(self.fitness)]

    def update_archive(self):
        """Update the archive with diverse and promising solutions."""
        for i in range(self.pop_size):
            if len(self.archive) < self.archive_size:
                self.archive.append(self.population[i])
                self.archive_fitness.append(self.fitness[i])
            else:
                worst_archive_index = np.argmax(self.archive_fitness)
                if self.fitness[i] < self.archive_fitness[worst_archive_index]:
                    self.archive[worst_archive_index] = self.population[i]
                    self.archive_fitness[worst_archive_index] = self.fitness[i]

    def mutate(self, x_i, best_x):
        """Modified mutation operator using population best and archive."""
        indices = np.random.choice(self.pop_size, size=2, replace=False)
        x_r1 = self.population[indices[0]]
        x_r2 = self.population[indices[1]]

        if len(self.archive) > 0:
            archive_index = np.random.randint(len(self.archive))
            x_r3 = self.archive[archive_index]
            v_i = x_i + self.F * (best_x - x_i) + self.F * (x_r1 - x_r2) + self.F * (x_r3 - x_i)
        else:
             v_i = x_i + self.F * (best_x - x_i) + self.F * (x_r1 - x_r2)

        return v_i

    def crossover(self, x_i, v_i):
        """Adaptive Crossover."""
        u_i = np.copy(x_i)
        j_rand = np.random.randint(self.dim)
        for j in range(self.dim):
            if np.random.rand() <= self.CR or j == j_rand:
                u_i[j] = v_i[j]
        return u_i

    def repair(self, x, func):
        return np.clip(x, func.bounds.lb, func.bounds.ub)

    def adjust_crossover_rate(self):
        """Adjust CR based on the success rate of previous generations."""
        self.CR = np.clip(self.CR + self.CR_adapt_rate * (np.random.rand() - 0.5), 0.1, 0.9)


    def __call__(self, func):
        self.initialize_population(func)

        while self.budget > 0:
            best_index = np.argmin(self.fitness)
            best_x = self.population[best_index]

            self.adjust_crossover_rate()

            for i in range(self.pop_size):
                # Mutation
                v_i = self.mutate(self.population[i], best_x)

                # Crossover
                u_i = self.crossover(self.population[i], v_i)

                # Repair
                u_i = self.repair(u_i, func)

                # Selection
                f_u_i = func(u_i)
                self.budget -= 1

                if f_u_i < self.fitness[i]:
                    self.population[i] = u_i
                    self.fitness[i] = f_u_i

                    if f_u_i < self.f_opt:
                        self.f_opt = f_u_i
                        self.x_opt = u_i

            self.update_archive()

            if self.budget <= 0:
                break

        return self.f_opt, self.x_opt