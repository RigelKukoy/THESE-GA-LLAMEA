import numpy as np

class AdaptiveDEStrategyPool:
    def __init__(self, budget=10000, dim=10, pop_size=50, strategy_pool_size=4, CR_init=0.5):
        self.budget = budget
        self.dim = dim
        self.pop_size = pop_size
        self.strategy_pool_size = strategy_pool_size
        self.CR_init = CR_init
        self.population = None
        self.fitness = None
        self.CR = None
        self.strategies = [self.mutation_strategy_1, self.mutation_strategy_2, self.mutation_strategy_3, self.mutation_strategy_4]  # Pool of mutation strategies
        self.strategy_selection_probs = np.ones(self.strategy_pool_size) / self.strategy_pool_size
        self.success_counts = np.zeros(self.strategy_pool_size)
        self.f_opt = np.Inf
        self.x_opt = None
        self.eval_count = 0

    def initialize_population(self, func):
        self.population = np.random.uniform(func.bounds.lb, func.bounds.ub, size=(self.pop_size, self.dim))
        self.fitness = np.array([func(x) for x in self.population])
        self.eval_count += self.pop_size
        self.CR = np.full(self.pop_size, self.CR_init)
        
        best_index = np.argmin(self.fitness)
        if self.fitness[best_index] < self.f_opt:
            self.f_opt = self.fitness[best_index]
            self.x_opt = self.population[best_index].copy()

    def mutation_strategy_1(self, x, population): # DE/rand/1
        idxs = np.random.choice(self.pop_size, 3, replace=False)
        x_r1, x_r2, x_r3 = population[idxs]
        return x_r1 + 0.8 * (x_r2 - x_r3)

    def mutation_strategy_2(self, x, population): # DE/current-to-rand/1
         idxs = np.random.choice(self.pop_size, 2, replace=False)
         x_r1, x_r2 = population[idxs]
         return x + 0.5 * (x_r1 - x_r2)
    
    def mutation_strategy_3(self, x, population): # DE/best/1
        best_index = np.argmin(self.fitness)
        x_best = population[best_index]
        idxs = np.random.choice(self.pop_size, 2, replace=False)
        x_r1, x_r2 = population[idxs]
        return x_best + 0.8 * (x_r1 - x_r2)
    
    def mutation_strategy_4(self, x, population): # DE/current-to-best/1
        best_index = np.argmin(self.fitness)
        x_best = population[best_index]
        idxs = np.random.choice(self.pop_size, 1, replace=False)
        x_r1 = population[idxs[0]]
        return x + 0.5 * (x_best - x) + 0.5 * (x_r1 - x)

    def evolve(self, func):
        for i in range(self.pop_size):
            # Strategy selection
            strategy_index = np.random.choice(self.strategy_pool_size, p=self.strategy_selection_probs)
            mutation_strategy = self.strategies[strategy_index]

            # Mutation
            x_mutated = mutation_strategy(self.population[i], self.population)
            x_mutated = np.clip(x_mutated, func.bounds.lb, func.bounds.ub)

            # Crossover
            x_trial = self.population[i].copy()
            j_rand = np.random.randint(self.dim)
            CR = np.clip(np.random.normal(self.CR[i], 0.1), 0.1, 1.0)  # Self-adjusting CR
            for j in range(self.dim):
                if np.random.rand() < CR or j == j_rand:
                    x_trial[j] = x_mutated[j]

            # Selection
            f_trial = func(x_trial)
            self.eval_count += 1

            if f_trial < self.fitness[i]:
                self.success_counts[strategy_index] += 1
                self.CR[i] = CR  # Update CR of individual

                self.population[i] = x_trial
                self.fitness[i] = f_trial
                
                if f_trial < self.f_opt:
                    self.f_opt = f_trial
                    self.x_opt = x_trial.copy()
            
            if self.eval_count >= self.budget:
                break

        # Update strategy selection probabilities
        self.strategy_selection_probs = (1 - 0.1) * self.strategy_selection_probs + 0.1 * (self.success_counts / np.sum(self.success_counts) if np.sum(self.success_counts) > 0 else np.ones(self.strategy_pool_size) / self.strategy_pool_size)
        self.strategy_selection_probs /= np.sum(self.strategy_selection_probs) # Normalize
        self.success_counts = np.zeros(self.strategy_pool_size)  # Reset success counts
                
    def __call__(self, func):
        self.initialize_population(func)
        while self.eval_count < self.budget:
            self.evolve(func)
        return self.f_opt, self.x_opt