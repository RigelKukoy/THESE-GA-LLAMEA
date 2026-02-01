import numpy as np

class AdaptiveVelocityDE:
    def __init__(self, budget=10000, dim=10, popsize=None, F=0.5, CR=0.7, restart_prob=0.01):
        self.budget = budget
        self.dim = dim
        self.popsize = popsize if popsize is not None else 10 * self.dim
        self.F = F
        self.CR = CR
        self.restart_prob = restart_prob
        self.velocity = np.zeros((self.popsize, self.dim))

    def __call__(self, func):
        lb = func.bounds.lb
        ub = func.bounds.ub
        self.population = np.random.uniform(lb, ub, size=(self.popsize, self.dim))
        self.fitness = np.array([func(x) for x in self.population])
        self.eval_count = self.popsize
        self.f_opt = np.min(self.fitness)
        self.x_opt = self.population[np.argmin(self.fitness)]

        while self.eval_count < self.budget:
            for i in range(self.popsize):
                # Mutation with velocity update and best-so-far component
                random_index = np.random.randint(0, self.popsize)
                mutant = self.population[i] + self.velocity[i] + self.F * (self.x_opt - self.population[i]) + self.F * (self.population[random_index] - self.population[i])
                mutant = np.clip(mutant, lb, ub)

                # Crossover
                crossover_mask = np.random.rand(self.dim) < self.CR
                trial = np.where(crossover_mask, mutant, self.population[i])

                # Selection
                f_trial = func(trial)
                self.eval_count += 1

                if f_trial < self.fitness[i]:
                    # Update velocity
                    self.velocity[i] = trial - self.population[i]
                    self.population[i] = trial
                    self.fitness[i] = f_trial
                    if self.fitness[i] < self.f_opt:
                        self.f_opt = self.fitness[i]
                        self.x_opt = self.population[i]

                # Restart mechanism: Randomly re-initialize individuals to avoid stagnation
                if np.random.rand() < self.restart_prob:
                    self.population[i] = np.random.uniform(lb, ub, size=self.dim)
                    self.fitness[i] = func(self.population[i])
                    self.eval_count += 1
                    self.velocity[i] = np.zeros(self.dim)  # Reset velocity
                    if self.fitness[i] < self.f_opt:
                        self.f_opt = self.fitness[i]
                        self.x_opt = self.population[i]

            if self.eval_count >= self.budget:
                break

        return self.f_opt, self.x_opt