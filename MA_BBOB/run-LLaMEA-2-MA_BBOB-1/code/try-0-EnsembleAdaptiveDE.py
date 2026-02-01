import numpy as np

class EnsembleAdaptiveDE:
    def __init__(self, budget=10000, dim=10, popsize=50, archive_size=10):
        self.budget = budget
        self.dim = dim
        self.popsize = popsize
        self.archive_size = archive_size
        self.F = 0.5  # Initial scaling factor
        self.CR = 0.7  # Initial crossover rate
        self.population = None
        self.fitness = None
        self.archive = []
        self.archive_fitness = []
        self.best_fitness = np.inf
        self.best_solution = None

    def initialize_population(self, func):
        self.population = np.random.uniform(func.bounds.lb, func.bounds.ub, size=(self.popsize, self.dim))
        self.fitness = np.array([func(x) for x in self.population])
        self.best_fitness = np.min(self.fitness)
        self.best_solution = self.population[np.argmin(self.fitness)]

    def mutate(self, i):
        mutation_strategy = np.random.choice(['DE/rand/1', 'DE/best/1', 'DE/current-to-rand/1', 'DE/current-to-best/1'])

        if mutation_strategy == 'DE/rand/1':
            indices = np.random.choice(self.popsize, 3, replace=False)
            x_r1, x_r2, x_r3 = self.population[indices]
            v = x_r1 + self.F * (x_r2 - x_r3)
        elif mutation_strategy == 'DE/best/1':
            indices = np.random.choice(self.popsize, 2, replace=False)
            x_r1, x_r2 = self.population[indices]
            v = self.best_solution + self.F * (x_r1 - x_r2)
        elif mutation_strategy == 'DE/current-to-rand/1':
            indices = np.random.choice(self.popsize, 2, replace=False)
            x_r1, x_r2 = self.population[indices]
            v = self.population[i] + self.F * (x_r1 - x_r2)
        elif mutation_strategy == 'DE/current-to-best/1':
            v = self.population[i] + self.F * (self.best_solution - self.population[i]) + self.F * (self.population[np.random.randint(self.popsize)] - self.population[np.random.randint(self.popsize)])
        else:
            raise ValueError("Invalid mutation strategy")

        return v

    def crossover(self, i, mutant):
        trial_vector = np.copy(self.population[i])
        j_rand = np.random.randint(self.dim)
        for j in range(self.dim):
            if np.random.rand() < self.CR or j == j_rand:
                trial_vector[j] = mutant[j]
        return trial_vector

    def handle_boundary(self, trial_vector, func):
        return np.clip(trial_vector, func.bounds.lb, func.bounds.ub)
    
    def update_archive(self, x, f):
        if len(self.archive) < self.archive_size:
            self.archive.append(x)
            self.archive_fitness.append(f)
        else:
            if f < np.max(self.archive_fitness):
                worst_index = np.argmax(self.archive_fitness)
                self.archive[worst_index] = x
                self.archive_fitness[worst_index] = f
            
    def __call__(self, func):
        self.initialize_population(func)
        evals = self.popsize

        while evals < self.budget:
            for i in range(self.popsize):
                mutant = self.mutate(i)
                trial_vector = self.crossover(i, mutant)
                trial_vector = self.handle_boundary(trial_vector, func)
                
                f_trial = func(trial_vector)
                evals += 1

                if f_trial < self.fitness[i]:
                    self.update_archive(self.population[i], self.fitness[i])
                    self.population[i] = trial_vector
                    self.fitness[i] = f_trial

                    if f_trial < self.best_fitness:
                        self.best_fitness = f_trial
                        self.best_solution = trial_vector
                else:
                    self.update_archive(trial_vector, f_trial)

                if evals >= self.budget:
                    break

            # Adaptive F and CR using archive information (simplified)
            if len(self.archive) > 0:
                self.F = np.random.uniform(0.4, 0.9)
                self.CR = np.random.uniform(0.3, 1.0)


        return self.best_fitness, self.best_solution