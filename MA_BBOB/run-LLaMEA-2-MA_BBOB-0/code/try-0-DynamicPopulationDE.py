import numpy as np

class DynamicPopulationDE:
    def __init__(self, budget=10000, dim=10, initial_pop_size=50, F=0.5, CR=0.9, pop_growth_rate=0.05, pop_shrink_rate=0.02, F_adapt_rate=0.1):
        self.budget = budget
        self.dim = dim
        self.pop_size = initial_pop_size
        self.initial_pop_size = initial_pop_size
        self.F = F
        self.CR = CR
        self.pop_growth_rate = pop_growth_rate
        self.pop_shrink_rate = pop_shrink_rate
        self.F_adapt_rate = F_adapt_rate
        self.population = None
        self.fitness = None
        self.x_opt = None
        self.f_opt = np.inf
        self.success_history = []


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
            improvements = 0  # Track number of improvements in this generation


            for i in range(self.pop_size):
                idxs = np.random.choice(self.pop_size, 3, replace=False)
                x_r1, x_r2, x_r3 = self.population[idxs]

                mutant = x_r1 + self.F * (x_r2 - x_r3)
                mutant = np.clip(mutant, func.bounds.lb, func.bounds.ub)

                crossover = np.random.uniform(size=self.dim) < self.CR
                trial = np.where(crossover, mutant, self.population[i])

                f_trial = func(trial)
                self.budget -= 1

                if f_trial < self.f_opt:
                    self.f_opt = f_trial
                    self.x_opt = trial
                    improvements += 1

                if f_trial < self.fitness[i]:
                    new_fitness[i] = f_trial
                    new_population[i] = trial

            self.population = new_population
            self.fitness = new_fitness

            best_index = np.argmin(self.fitness)
            if self.fitness[best_index] < self.f_opt:
                self.f_opt = self.fitness[best_index]
                self.x_opt = self.population[best_index]

            # Adapt population size based on improvement rate
            improvement_rate = improvements / self.pop_size

            if improvement_rate > self.pop_growth_rate and self.budget > self.dim:
                # Increase population size
                num_new = int(self.pop_size * self.pop_growth_rate)
                new_individuals = np.random.uniform(func.bounds.lb, func.bounds.ub, size=(num_new, self.dim))
                new_fitnesses = np.array([func(x) for x in new_individuals])
                self.budget -= num_new

                self.population = np.vstack((self.population, new_individuals))
                self.fitness = np.concatenate((self.fitness, new_fitnesses))
                self.pop_size += num_new

            elif improvement_rate < self.pop_shrink_rate and self.pop_size > self.initial_pop_size:
                # Decrease population size
                num_remove = int(self.pop_size * self.pop_shrink_rate)
                indices_to_remove = np.argsort(self.fitness)[-num_remove:]  # Remove worst individuals
                self.population = np.delete(self.population, indices_to_remove, axis=0)
                self.fitness = np.delete(self.fitness, indices_to_remove)
                self.pop_size -= num_remove


            # Adapt F based on improvement rate
            if improvement_rate > 0.1:
                self.F *= (1 + self.F_adapt_rate)
            else:
                self.F *= (1 - self.F_adapt_rate)
            self.F = np.clip(self.F, 0.1, 1.0)

        return self.f_opt, self.x_opt