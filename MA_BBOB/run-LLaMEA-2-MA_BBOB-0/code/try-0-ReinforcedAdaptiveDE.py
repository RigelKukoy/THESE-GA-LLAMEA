import numpy as np

class ReinforcedAdaptiveDE:
    def __init__(self, budget=10000, dim=10, pop_size=50, mutation_strategies=['DE/rand/1', 'DE/best/1', 'DE/current-to-rand/1'], learning_rate=0.1, exploration_rate=0.1):
        self.budget = budget
        self.dim = dim
        self.pop_size = pop_size
        self.mutation_strategies = mutation_strategies
        self.learning_rate = learning_rate
        self.exploration_rate = exploration_rate
        self.strategy_rewards = {strategy: 0.0 for strategy in mutation_strategies}
        self.strategy_counts = {strategy: 0 for strategy in mutation_strategies}
        self.F = 0.5
        self.CR = 0.7

    def apply_mutation(self, strategy, population, i, func_bounds):
        if strategy == 'DE/rand/1':
            idxs = np.random.choice(self.pop_size, 3, replace=False)
            x_1, x_2, x_3 = population[idxs]
            mutant = x_1 + self.F * (x_2 - x_3)
        elif strategy == 'DE/best/1':
            best_idx = np.argmin([func(x) for x in population])
            x_best = population[best_idx]
            idxs = np.random.choice(self.pop_size, 2, replace=False)
            x_1, x_2 = population[idxs]
            mutant = x_best + self.F * (x_1 - x_2)
        elif strategy == 'DE/current-to-rand/1':
            x_current = population[i]
            idxs = np.random.choice(self.pop_size, 2, replace=False)
            x_1, x_2 = population[idxs]
            mutant = x_current + self.F * (x_1 - x_2)
        else:
            raise ValueError(f"Unknown mutation strategy: {strategy}")

        mutant = np.clip(mutant, func_bounds.lb, func_bounds.ub)
        return mutant

    def choose_strategy(self):
        if np.random.rand() < self.exploration_rate:
            return np.random.choice(self.mutation_strategies)
        else:
            avg_rewards = {s: self.strategy_rewards[s] / (self.strategy_counts[s] + 1e-6) for s in self.mutation_strategies}
            return max(avg_rewards, key=avg_rewards.get)

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
                strategy = self.choose_strategy()

                # Mutation
                mutant = self.apply_mutation(strategy, self.population, i, func.bounds)

                # Crossover
                crossover = np.random.uniform(size=self.dim) < self.CR
                trial = np.where(crossover, mutant, self.population[i])

                # Selection
                f_trial = func(trial)
                self.budget -= 1

                if f_trial < self.f_opt:
                    self.f_opt = f_trial
                    self.x_opt = trial

                # Reinforcement Learning Update
                reward = fitness[i] - f_trial  # Immediate reward
                self.strategy_rewards[strategy] += self.learning_rate * reward
                self.strategy_counts[strategy] += 1
                
                if f_trial < fitness[i]:
                    fitness[i] = f_trial
                    self.population[i] = trial

        return self.f_opt, self.x_opt