import numpy as np

class AdaptiveMutationPoolDE:
    def __init__(self, budget=10000, dim=10, popsize=None, CR=0.7, mutation_pool=None, selection_pressure=0.2):
        self.budget = budget
        self.dim = dim
        self.popsize = popsize if popsize is not None else 10 * self.dim
        self.CR = CR
        self.selection_pressure = selection_pressure
        
        if mutation_pool is None:
            self.mutation_pool = [
                self._mutation_DE_rand1,
                self._mutation_DE_best1,
                self._mutation_DE_current_to_rand1,
                self._mutation_DE_current_to_best1,
            ]
        else:
            self.mutation_pool = mutation_pool
        
        self.mutation_success_rates = np.ones(len(self.mutation_pool)) / len(self.mutation_pool)  # Initialize with equal probabilities

    def __call__(self, func):
        lb = func.bounds.lb
        ub = func.bounds.ub
        self.population = np.random.uniform(lb, ub, size=(self.popsize, self.dim))
        self.fitness = np.array([func(x) for x in self.population])
        self.eval_count = self.popsize
        self.f_opt = np.min(self.fitness)
        self.x_opt = self.population[np.argmin(self.fitness)]
        self.mutation_history = []


        while self.eval_count < self.budget:
            for i in range(self.popsize):
                # Select mutation strategy based on success rates
                mutation_index = np.random.choice(len(self.mutation_pool), p=self.mutation_success_rates)
                mutation_function = self.mutation_pool[mutation_index]

                # Mutation
                mutant = mutation_function(i)
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
                    if self.fitness[i] < self.f_opt:
                        self.f_opt = self.fitness[i]
                        self.x_opt = self.population[i]
                    self.mutation_history.append((mutation_index, True))
                else:
                    self.mutation_history.append((mutation_index, False))

            self._update_mutation_success_rates()

        return self.f_opt, self.x_opt

    def _mutation_DE_rand1(self, i):
        idxs = np.random.choice(self.popsize, 3, replace=False)
        x1, x2, x3 = self.population[idxs]
        return self.population[i] + 0.5 * (x2 - x3)

    def _mutation_DE_best1(self, i):
        idxs = np.random.choice(self.popsize, 2, replace=False)
        x1, x2 = self.population[idxs]
        return self.population[i] + 0.5 * (self.x_opt - self.population[i]) + 0.5 * (x1 - x2)

    def _mutation_DE_current_to_rand1(self, i):
         idxs = np.random.choice(self.popsize, 3, replace=False)
         x1, x2, x3 = self.population[idxs]
         return self.population[i] + 0.5*(x1 - self.population[i]) + 0.5 * (x2 - x3)

    def _mutation_DE_current_to_best1(self, i):
        idxs = np.random.choice(self.popsize, 2, replace=False)
        x1, x2 = self.population[idxs]
        return self.population[i] + 0.5 * (self.x_opt - self.population[i]) + 0.5 * (x1 - x2)

    def _update_mutation_success_rates(self):
        for i in range(len(self.mutation_pool)):
            successes = [success for index, success in self.mutation_history if index == i]
            if successes:
                self.mutation_success_rates[i] = self.selection_pressure * np.mean(successes) + (1 - self.selection_pressure) * self.mutation_success_rates[i]
            else:
                self.mutation_success_rates[i] *= (1 - self.selection_pressure) #if no updates reduce probability
        self.mutation_success_rates /= np.sum(self.mutation_success_rates)