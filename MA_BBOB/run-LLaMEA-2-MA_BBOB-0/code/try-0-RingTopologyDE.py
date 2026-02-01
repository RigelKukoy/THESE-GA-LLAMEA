import numpy as np

class RingTopologyDE:
    def __init__(self, budget=10000, dim=10, pop_size=50, F=0.5, CR=0.9, exploration_decay=0.999):
        self.budget = budget
        self.dim = dim
        self.pop_size = pop_size
        self.F = F
        self.CR = CR
        self.exploration_decay = exploration_decay
        self.exploration_rate = 1.0
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
            new_population = np.copy(self.population)
            new_fitness = np.copy(self.fitness)

            for i in range(self.pop_size):
                # Ring Topology: Select neighbors
                left = (i - 1) % self.pop_size
                right = (i + 1) % self.pop_size

                # Create mutant vector
                idxs = np.random.choice(self.pop_size, 3, replace=False)
                x_r1, x_r2, x_r3 = self.population[idxs]

                if np.random.rand() < self.exploration_rate:
                    mutant = x_r1 + self.F * (x_r2 - x_r3)  # Global exploration
                else:
                    mutant = self.population[i] + self.F * (self.population[left] - self.population[right]) # Local exploitation via ring

                mutant = np.clip(mutant, func.bounds.lb, func.bounds.ub)

                # Crossover
                crossover = np.random.uniform(size=self.dim) < self.CR
                trial = np.where(crossover, mutant, self.population[i])

                # Evaluation
                f_trial = func(trial)
                self.budget -= 1

                if f_trial < self.f_opt:
                    self.f_opt = f_trial
                    self.x_opt = trial

                # Selection
                if f_trial < self.fitness[i]:
                    new_fitness[i] = f_trial
                    new_population[i] = trial

            self.population = new_population
            self.fitness = new_fitness

            # Update best
            best_index = np.argmin(self.fitness)
            if self.fitness[best_index] < self.f_opt:
                self.f_opt = self.fitness[best_index]
                self.x_opt = self.population[best_index]

            self.exploration_rate *= self.exploration_decay # Decay exploration rate

        return self.f_opt, self.x_opt