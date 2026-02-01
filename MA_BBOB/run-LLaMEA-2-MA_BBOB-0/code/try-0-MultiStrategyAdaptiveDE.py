import numpy as np

class MultiStrategyAdaptiveDE:
    def __init__(self, budget=10000, dim=10, pop_size=50, archive_size=10,
                 mutation_strategies=[
                     lambda F, x1, x2, x3: x1 + F * (x2 - x3),  # DE/rand/1
                     lambda F, x1, x2, x3: x1 + F * (x2 - x3) + F * (np.random.rand(dim) - np.random.rand(dim)),  # DE/rand/1 with exploration
                     lambda F, x1, x2, x3, x_best: x_best + F * (x1 - x2),  # DE/best/1
                     lambda F, x1, x2, x3, x_best: x_best + F * (x1 - x2) + F * (np.random.rand(dim) - np.random.rand(dim)),  # DE/best/1 with exploration
                 ],
                 F_values=[0.1, 0.5, 0.9], CR_values=[0.1, 0.5, 0.9]):
        self.budget = budget
        self.dim = dim
        self.pop_size = pop_size
        self.archive_size = archive_size
        self.mutation_strategies = mutation_strategies
        self.F_values = F_values
        self.CR_values = CR_values
        self.strategy_rewards = [0.0] * len(mutation_strategies)
        self.strategy_counts = [0] * len(mutation_strategies)
        self.archive = []
        self.p = 0.1
        self.epsilon = 0.1  # Exploration rate for strategy selection

    def select_strategy(self):
        if np.random.rand() < self.epsilon:
            return np.random.randint(len(self.mutation_strategies))
        else:
            avg_rewards = [self.strategy_rewards[i] / (self.strategy_counts[i] + 1e-6) for i in range(len(self.mutation_strategies))]
            return np.argmax(avg_rewards)

    def __call__(self, func):
        self.f_opt = np.Inf
        self.x_opt = None
        self.population = np.random.uniform(func.bounds.lb, func.bounds.ub, size=(self.pop_size, self.dim))
        fitness = np.array([func(x) for x in self.population])
        self.budget -= self.pop_size

        for i in range(self.pop_size):
            if fitness[i] < self.f_opt:
                self.f_opt = fitness[i]
                self.x_opt = self.population[i]

        while self.budget > 0:
            for i in range(self.pop_size):
                # Strategy Selection
                strategy_index = self.select_strategy()
                mutation_strategy = self.mutation_strategies[strategy_index]

                # Parameter Selection
                F = np.random.choice(self.F_values)
                CR = np.random.choice(self.CR_values)

                # Mutation
                idxs = np.random.choice(self.pop_size, 3, replace=False)
                x_1, x_2, x_3 = self.population[idxs]
                x_best = self.population[np.argmin(fitness)] if len(self.population) > 0 else x_1 #fallback

                if len(self.mutation_strategies[strategy_index].__code__.co_varnames) > 3:
                    mutant = mutation_strategy(F, x_1, x_2, x_3, x_best)
                else:
                    mutant = mutation_strategy(F, x_1, x_2, x_3)
                mutant = np.clip(mutant, func.bounds.lb, func.bounds.ub)

                # Crossover
                crossover = np.random.uniform(size=self.dim) < CR
                trial = np.where(crossover, mutant, self.population[i])

                # Selection
                f_trial = func(trial)
                self.budget -= 1

                if f_trial < self.f_opt:
                    self.f_opt = f_trial
                    self.x_opt = trial

                if (fitness[i] < 0 and f_trial < 0) or np.random.rand() < self.p:
                    if f_trial < fitness[i]:
                        reward = fitness[i] - f_trial

                        # Update Strategy Rewards
                        self.strategy_rewards[strategy_index] += reward

                        # Update Strategy Counts
                        self.strategy_counts[strategy_index] += 1

                        fitness[i] = f_trial
                        self.population[i] = trial

                        if len(self.archive) < self.archive_size:
                            self.archive.append(self.population[i].copy())
                        else:
                            idx_to_replace = np.random.randint(0, self.archive_size)
                            self.archive[idx_to_replace] = self.population[i].copy()
                else:
                     if f_trial < fitness[i]:
                        fitness[i] = f_trial
                        self.population[i] = trial

                        if len(self.archive) < self.archive_size:
                            self.archive.append(self.population[i].copy())
                        else:
                            idx_to_replace = np.random.randint(0, self.archive_size)
                            self.archive[idx_to_replace] = self.population[i].copy()
        return self.f_opt, self.x_opt