import numpy as np

class CauchyCrossoverAdaptiveDE:
    def __init__(self, budget=10000, dim=10, pop_size=40, Cr_init=0.5, F=0.5, learning_rate=0.1):
        self.budget = budget
        self.dim = dim
        self.pop_size = pop_size
        self.Cr = Cr_init
        self.F = F
        self.learning_rate = learning_rate
        self.best_fitness_history = []
        self.population = None
        self.fitness = None
        self.success_Cr = []
        self.archive = []

    def __call__(self, func):
        # Initialization
        self.population = np.random.uniform(func.bounds.lb, func.bounds.ub, size=(self.pop_size, self.dim))
        self.fitness = np.array([func(x) for x in self.population])
        self.budget -= self.pop_size
        
        self.f_opt = np.min(self.fitness)
        self.x_opt = self.population[np.argmin(self.fitness)]
        self.best_fitness_history.append(self.f_opt)
        
        generation = 0

        while self.budget > self.pop_size:
            new_population = np.copy(self.population)
            new_fitness = np.zeros(self.pop_size)
            successful_offspring = 0
            Cr_sum = 0

            for i in range(self.pop_size):
                # Mutation using Cauchy distribution
                indices = [j for j in range(self.pop_size) if j != i]
                idxs = np.random.choice(indices, size=3, replace=False)
                x_r1, x_r2, x_r3 = self.population[idxs[0]], self.population[idxs[1]], self.population[idxs[2]]
                
                # Cauchy mutation
                cauchy_rand = np.random.standard_cauchy(size=self.dim)
                mutant = self.population[i] + self.F * (x_r1 - x_r2) + 0.01 * cauchy_rand  # Added Cauchy noise
                mutant = np.clip(mutant, func.bounds.lb, func.bounds.ub)
                
                # Crossover
                j_rand = np.random.randint(self.dim)
                for j in range(self.dim):
                    if np.random.rand() < self.Cr or j == j_rand:
                        new_population[i, j] = mutant[j]
                    
                new_population[i] = np.clip(new_population[i], func.bounds.lb, func.bounds.ub)
                new_fitness[i] = func(new_population[i])
                self.budget -= 1

                # Selection
                if new_fitness[i] < self.fitness[i]:
                    self.archive.append(self.population[i].copy())
                    self.population[i] = new_population[i]
                    self.fitness[i] = new_fitness[i]
                    successful_offspring += 1
                    Cr_sum += self.Cr
                    if new_fitness[i] < self.f_opt:
                        self.f_opt = new_fitness[i]
                        self.x_opt = new_population[i]


            # Adapt Cr
            if successful_offspring > 0:
                mean_Cr = Cr_sum / successful_offspring
                self.Cr = (1 - self.learning_rate) * self.Cr + self.learning_rate * mean_Cr
            else:
                self.Cr = (1 - self.learning_rate) * self.Cr + self.learning_rate * np.random.rand() # Randomize if no success
                
            self.Cr = np.clip(self.Cr, 0.1, 0.9)

            self.best_fitness_history.append(self.f_opt)
            generation += 1

        return self.f_opt, self.x_opt