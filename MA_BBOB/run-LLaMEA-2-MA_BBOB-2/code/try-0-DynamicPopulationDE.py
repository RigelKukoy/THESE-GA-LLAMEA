import numpy as np

class DynamicPopulationDE:
    def __init__(self, budget=10000, dim=10, initial_popsize=None, F=0.5, CR=0.7, stagnation_threshold=10, popsize_increase_factor=1.5, popsize_decrease_factor=0.5):
        self.budget = budget
        self.dim = dim
        self.initial_popsize = initial_popsize if initial_popsize is not None else 10 * self.dim
        self.popsize = self.initial_popsize
        self.F = F
        self.CR = CR
        self.stagnation_threshold = stagnation_threshold
        self.popsize_increase_factor = popsize_increase_factor
        self.popsize_decrease_factor = popsize_decrease_factor
        self.population = None
        self.fitness = None
        self.x_opt = None
        self.f_opt = np.inf
        self.eval_count = 0
        self.stagnation_counter = 0
        self.previous_best_fitness = np.inf
        self.velocities = None

    def initialize_population(self, func):
        lb = func.bounds.lb
        ub = func.bounds.ub
        self.population = np.random.uniform(lb, ub, size=(self.popsize, self.dim))
        self.fitness = np.array([func(x) for x in self.population])
        self.eval_count += self.popsize
        self.x_opt = self.population[np.argmin(self.fitness)]
        self.f_opt = np.min(self.fitness)
        self.velocities = np.zeros_like(self.population)

    def __call__(self, func):
        self.initialize_population(func)
        lb = func.bounds.lb
        ub = func.bounds.ub

        while self.eval_count < self.budget:
            for i in range(self.popsize):
                # Mutation: Velocity-based mutation
                donor_indices = np.random.choice(self.popsize, 3, replace=False)
                
                # Update velocity
                self.velocities[i] = self.F * (self.population[donor_indices[1]] - self.population[donor_indices[2]])
                
                mutant = self.population[i] + self.velocities[i]
                mutant = np.clip(mutant, lb, ub)

                # Crossover
                crossover_mask = np.random.rand(self.dim) < self.CR
                trial = np.where(crossover_mask, mutant, self.population[i])

                # Selection
                f_trial = func(trial)
                self.eval_count += 1

                if f_trial < self.fitness[i]:
                    self.population[i] = trial
                    self.fitness[i] = f_trial

                    if f_trial < self.f_opt:
                        self.f_opt = f_trial
                        self.x_opt = trial
            
            # Stagnation Check
            if self.f_opt >= self.previous_best_fitness:
                self.stagnation_counter += 1
            else:
                self.stagnation_counter = 0

            self.previous_best_fitness = self.f_opt

            # Adjust Population Size
            if self.stagnation_counter > self.stagnation_threshold:
                if self.popsize < 2 * self.initial_popsize:  # Avoid excessive population growth
                    self.popsize = int(self.popsize * self.popsize_increase_factor)
                    self.population = np.vstack((self.population, np.random.uniform(lb, ub, size=(self.popsize - len(self.population), self.dim))))
                    self.fitness = np.concatenate((self.fitness, np.array([func(x) for x in self.population[len(self.fitness):]])))
                    self.eval_count += self.popsize - len(self.fitness) + len(self.population[len(self.fitness):])
                    self.velocities = np.vstack((self.velocities, np.zeros((self.popsize - len(self.velocities), self.dim))))

                else:
                    # If popsize is already large, consider decreasing it.
                     self.popsize = int(self.popsize * self.popsize_decrease_factor)
                     self.population = self.population[:self.popsize]
                     self.fitness = self.fitness[:self.popsize]
                     self.velocities = self.velocities[:self.popsize]

                self.stagnation_counter = 0  # Reset stagnation counter
            
            if self.eval_count >= self.budget:
                break

        return self.f_opt, self.x_opt