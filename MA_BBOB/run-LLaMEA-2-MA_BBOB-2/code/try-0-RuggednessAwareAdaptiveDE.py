import numpy as np

class RuggednessAwareAdaptiveDE:
    def __init__(self, budget=10000, dim=10, popsize=None, CR=0.7, F=0.5, selection_pressure=0.2, ruggedness_window=100):
        self.budget = budget
        self.dim = dim
        self.popsize = popsize if popsize is not None else 10 * self.dim
        self.CR = CR
        self.F = F
        self.selection_pressure = selection_pressure
        self.ruggedness_window = ruggedness_window  # Window size for ruggedness estimation

        self.mutation_strategies = {
            "rand1": self._mutation_DE_rand1,
            "best1": self._mutation_DE_best1,
            "current_to_rand1": self._mutation_DE_current_to_rand1,
            "current_to_best1": self._mutation_DE_current_to_best1
        }
        self.strategy_weights = {name: 1.0 / len(self.mutation_strategies) for name in self.mutation_strategies}
        self.success_rates = {name: 0.0 for name in self.mutation_strategies}
        self.strategy_usage_count = {name: 0 for name in self.mutation_strategies}
        self.fitness_history = []


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
                # Select mutation strategy based on weights
                strategy_name = self._select_strategy()
                mutation_function = self.mutation_strategies[strategy_name]
                self.strategy_usage_count[strategy_name] += 1

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
                    self.fitness_history.append(f_trial)  # Store for ruggedness calculation
                    self.population[i] = trial
                    self.fitness[i] = f_trial
                    if self.fitness[i] < self.f_opt:
                        self.f_opt = self.fitness[i]
                        self.x_opt = self.population[i]
                    self.success_rates[strategy_name] += 1
                else:
                    self.fitness_history.append(self.fitness[i]) #Store for ruggedness calculation

            self._update_strategy_weights()
            self._adjust_parameters(func)  # Adjust CR and F based on ruggedness
        return self.f_opt, self.x_opt

    def _select_strategy(self):
        names = list(self.strategy_weights.keys())
        weights = list(self.strategy_weights.values())
        return np.random.choice(names, p=weights)

    def _mutation_DE_rand1(self, i):
        idxs = np.random.choice(self.popsize, 3, replace=False)
        x1, x2, x3 = self.population[idxs]
        return self.population[i] + self.F * (x2 - x3)

    def _mutation_DE_best1(self, i):
        idxs = np.random.choice(self.popsize, 2, replace=False)
        x1, x2 = self.population[idxs]
        return self.population[i] + self.F * (self.x_opt - self.population[i]) + self.F * (x1 - x2)

    def _mutation_DE_current_to_rand1(self, i):
         idxs = np.random.choice(self.popsize, 3, replace=False)
         x1, x2, x3 = self.population[idxs]
         return self.population[i] + self.F*(x1 - self.population[i]) + self.F * (x2 - x3)

    def _mutation_DE_current_to_best1(self, i):
        idxs = np.random.choice(self.popsize, 2, replace=False)
        x1, x2 = self.population[idxs]
        return self.population[i] + self.F * (self.x_opt - self.population[i]) + self.F * (x1 - x2)

    def _update_strategy_weights(self):
        total_usage = sum(self.strategy_usage_count.values())
        if total_usage == 0:
             return #Avoid division by zero
        for name in self.strategy_weights:
            success_rate = self.success_rates[name] / self.strategy_usage_count[name] if self.strategy_usage_count[name] > 0 else 0.0
            self.strategy_weights[name] = (1 - self.selection_pressure) * self.strategy_weights[name] + self.selection_pressure * success_rate
            self.strategy_usage_count[name] = 0 #reset
            self.success_rates[name] = 0 #reset
        
        total_weight = sum(self.strategy_weights.values())
        for name in self.strategy_weights:
            self.strategy_weights[name] /= total_weight #normalize
            

    def _adjust_parameters(self, func):
        # Estimate landscape ruggedness based on recent fitness variance
        if len(self.fitness_history) > self.ruggedness_window:
            fitness_window = self.fitness_history[-self.ruggedness_window:]
            fitness_variance = np.var(fitness_window)
        else:
            fitness_variance = 0.0

        # Adjust CR and F based on ruggedness
        if fitness_variance > 1e-6:  # If landscape is rugged
            self.CR = min(1.0, self.CR + 0.1)  # Increase exploration
            self.F = max(0.1, self.F - 0.05)  # Reduce step size
        else:  # If landscape is smooth
            self.CR = max(0.2, self.CR - 0.05)  # Increase exploitation
            self.F = min(0.9, self.F + 0.1)  # Increase step size