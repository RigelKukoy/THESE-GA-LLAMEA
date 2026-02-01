import numpy as np

class DecayingPopSizeDE:
    def __init__(self, budget=10000, dim=10, initial_popsize=None, F_initial=0.5, CR_initial=0.7, popsize_decay_rate=0.0001, mutation_strategies=None):
        self.budget = budget
        self.dim = dim
        self.initial_popsize = initial_popsize if initial_popsize is not None else 10 * self.dim
        self.popsize = self.initial_popsize
        self.F = F_initial
        self.CR = CR_initial
        self.popsize_decay_rate = popsize_decay_rate
        self.mutation_strategies = mutation_strategies if mutation_strategies is not None else [
            lambda x1, x2, x3, F: x1 + F * (x2 - x3),  # DE/rand/1
            lambda x1, x2, x3, x4, x5, F: x1 + F * (x2 - x3) + F * (x4-x5) #DE/rand/2
        ]
        self.strategy_probabilities = np.ones(len(self.mutation_strategies)) / len(self.mutation_strategies)  # Initially uniform probabilities
        self.f_opt = np.Inf
        self.x_opt = None
        self.eval_count = 0

    def __call__(self, func):
        lb = func.bounds.lb
        ub = func.bounds.ub
        self.population = np.random.uniform(lb, ub, size=(self.popsize, self.dim))
        self.fitness = np.array([func(x) for x in self.population])
        self.eval_count += self.popsize
        self.f_opt = np.min(self.fitness)
        self.x_opt = self.population[np.argmin(self.fitness)]


        while self.eval_count < self.budget:
            # Decay population size
            self.popsize = max(int(self.initial_popsize * np.exp(-self.popsize_decay_rate * self.eval_count)), 4) # Ensure popsize is at least 4
            if self.population.shape[0] > self.popsize:
                # Reduce population size if needed
                indices_to_keep = np.argsort(self.fitness)[:self.popsize]
                self.population = self.population[indices_to_keep]
                self.fitness = self.fitness[indices_to_keep]
            elif self.population.shape[0] < self.popsize:
                # Increase population size if needed (e.g., after a restart)
                num_to_add = self.popsize - self.population.shape[0]
                new_individuals = np.random.uniform(lb, ub, size=(num_to_add, self.dim))
                self.population = np.vstack((self.population, new_individuals))
                new_fitness = np.array([func(x) for x in new_individuals])
                self.fitness = np.concatenate((self.fitness, new_fitness))
                self.eval_count += num_to_add

            for i in range(self.popsize):
                # Strategy selection
                strategy_index = np.random.choice(len(self.mutation_strategies), p=self.strategy_probabilities)
                mutation_strategy = self.mutation_strategies[strategy_index]

                # Mutation based on selected strategy
                if len(mutation_strategy.__code__.co_varnames) == 5:  # DE/rand/1
                    idxs = np.random.choice(self.popsize, 3, replace=False)
                    x1, x2, x3 = self.population[idxs]
                    mutant = mutation_strategy(x1, x2, x3, self.F)
                elif len(mutation_strategy.__code__.co_varnames) == 7: #DE/rand/2
                    idxs = np.random.choice(self.popsize, 5, replace=False)
                    x1, x2, x3, x4, x5 = self.population[idxs]
                    mutant = mutation_strategy(x1, x2, x3, x4, x5, self.F)
                else:
                    raise ValueError("Unsupported mutation strategy arity.")


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
                        self.x_opt = self.population[i]
                

        return self.f_opt, self.x_opt