import numpy as np

class SelfOrganizingDE:
    def __init__(self, budget=10000, dim=10, pop_size=40, F_initial=0.5, Cr_initial=0.9, lr_F=0.1, lr_Cr=0.1, niche_radius=0.5):
        self.budget = budget
        self.dim = dim
        self.pop_size = pop_size
        self.F_initial = F_initial
        self.Cr_initial = Cr_initial
        self.lr_F = lr_F # Learning rate for F
        self.lr_Cr = lr_Cr # Learning rate for Cr
        self.niche_radius = niche_radius
        self.population = None
        self.fitness = None
        self.F = None
        self.Cr = None
        self.success_F = None
        self.success_Cr = None
        self.archive = [] # Archive for storing successful solutions

    def initialize(self, func):
        self.population = np.random.uniform(func.bounds.lb, func.bounds.ub, size=(self.pop_size, self.dim))
        self.fitness = np.array([func(x) for x in self.population])
        self.budget -= self.pop_size
        self.F = np.full(self.pop_size, self.F_initial)
        self.Cr = np.full(self.pop_size, self.Cr_initial)
        self.success_F = np.zeros(self.pop_size)
        self.success_Cr = np.zeros(self.pop_size)
        self.f_opt = np.min(self.fitness)
        self.x_opt = self.population[np.argmin(self.fitness)]

    def niching(self):
        # Simplified niching: penalize individuals that are too close to each other
        for i in range(self.pop_size):
            for j in range(i + 1, self.pop_size):
                if np.linalg.norm(self.population[i] - self.population[j]) < self.niche_radius:
                    # Penalize the worse individual
                    if self.fitness[i] > self.fitness[j]:
                        self.fitness[i] += 0.01 * (self.fitness[i] - self.fitness[j])
                    else:
                        self.fitness[j] += 0.01 * (self.fitness[j] - self.fitness[i])

    def __call__(self, func):
        self.initialize(func)

        while self.budget > self.pop_size:
            new_population = np.copy(self.population)
            new_fitness = np.copy(self.fitness)

            for i in range(self.pop_size):
                # Mutation
                indices = [j for j in range(self.pop_size) if j != i]
                idxs = np.random.choice(indices, size=3, replace=False)
                x_r1, x_r2, x_r3 = self.population[idxs[0]], self.population[idxs[1]], self.population[idxs[2]]
                mutant = self.population[i] + self.F[i] * (x_r1 - x_r2)
                mutant = np.clip(mutant, func.bounds.lb, func.bounds.ub)

                # Crossover
                for j in range(self.dim):
                    if np.random.rand() > self.Cr[i]:
                        mutant[j] = self.population[i, j]

                # Evaluation
                f_mutant = func(mutant)
                self.budget -= 1

                # Selection
                if f_mutant < self.fitness[i]:
                    new_population[i] = mutant
                    new_fitness[i] = f_mutant
                    # Update success history
                    self.success_F[i] = 0.9 * self.success_F[i] + 0.1 # Exponential smoothing
                    self.success_Cr[i] = 0.9 * self.success_Cr[i] + 0.1
                    
                    # Archive successful solutions
                    self.archive.append(self.population[i].copy())  # Store the old solution
                    if len(self.archive) > 2 * self.pop_size:
                        self.archive.pop(0) # Maintain archive size
                        
                else:
                    self.success_F[i] = 0.9 * self.success_F[i]
                    self.success_Cr[i] = 0.9 * self.success_Cr[i]

            # Update population and fitness
            self.population = new_population
            self.fitness = new_fitness

            # Self-organizing adaptation of F and Cr
            self.F = np.clip(self.F + self.lr_F * (self.success_F - 0.5), 0.1, 1.0)
            self.Cr = np.clip(self.Cr + self.lr_Cr * (self.success_Cr - 0.5), 0.1, 1.0)
            
            # Niching strategy (optional but recommended)
            self.niching()
            
            # Update best solution
            if np.min(self.fitness) < self.f_opt:
                self.f_opt = np.min(self.fitness)
                self.x_opt = self.population[np.argmin(self.fitness)]

        return self.f_opt, self.x_opt