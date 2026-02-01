import numpy as np

class EvolvingSearchStrategies:
    def __init__(self, budget=10000, dim=10, pop_size=50, num_strategies=5, strategy_mutation_rate=0.1):
        self.budget = budget
        self.dim = dim
        self.pop_size = pop_size
        self.num_strategies = num_strategies
        self.strategy_mutation_rate = strategy_mutation_rate
        self.strategies = self.initialize_strategies()
        self.strategy_performance = np.zeros(num_strategies)
        self.population = None
        self.fitness = None
        self.x_opt = None
        self.f_opt = np.inf

    def initialize_strategies(self):
        strategies = []
        for _ in range(self.num_strategies):
            strategy = {
                'F': np.random.uniform(0.1, 1.0),
                'CR': np.random.uniform(0.1, 1.0),
                'local_search_prob': np.random.uniform(0.0, 0.2),
                'local_search_step_size': np.random.uniform(0.01, 0.1)
            }
            strategies.append(strategy)
        return strategies

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

            # Select and apply strategies
            for i in range(self.pop_size):
                strategy_id = self.select_strategy()
                strategy = self.strategies[strategy_id]

                mutant = self.create_mutant(i, strategy, func.bounds.lb, func.bounds.ub)
                trial = self.crossover(self.population[i], mutant, strategy['CR'])
                trial = self.local_search(trial, func.bounds.lb, func.bounds.ub, strategy['local_search_prob'], strategy['local_search_step_size'])
                
                f_trial = func(trial)
                self.budget -= 1
                
                if f_trial < self.f_opt:
                    self.f_opt = f_trial
                    self.x_opt = trial
                
                if f_trial < self.fitness[i]:
                    new_fitness[i] = f_trial
                    new_population[i] = trial
                    self.strategy_performance[strategy_id] += (self.fitness[i] - f_trial) # Reward the strategy

            self.population = new_population
            self.fitness = new_fitness

            # Update strategies
            self.evolve_strategies()
            
            best_index = np.argmin(self.fitness)
            if self.fitness[best_index] < self.f_opt:
                self.f_opt = self.fitness[best_index]
                self.x_opt = self.population[best_index]


        return self.f_opt, self.x_opt

    def select_strategy(self):
        probabilities = np.exp(self.strategy_performance - np.max(self.strategy_performance))
        probabilities /= np.sum(probabilities)
        return np.random.choice(range(self.num_strategies), p=probabilities)

    def create_mutant(self, i, strategy, lb, ub):
        idxs = np.random.choice(self.pop_size, 3, replace=False)
        x_r1, x_r2, x_r3 = self.population[idxs]
        mutant = x_r1 + strategy['F'] * (x_r2 - x_r3)
        return np.clip(mutant, lb, ub)

    def crossover(self, individual, mutant, CR):
        crossover = np.random.uniform(size=self.dim) < CR
        return np.where(crossover, mutant, individual)

    def local_search(self, x, lb, ub, prob, step_size):
        if np.random.rand() < prob:
            x_new = x + np.random.uniform(-step_size, step_size, size=self.dim)
            return np.clip(x_new, lb, ub)
        return x

    def evolve_strategies(self):
        for i in range(self.num_strategies):
            if np.random.rand() < self.strategy_mutation_rate:
                param_to_mutate = np.random.choice(['F', 'CR', 'local_search_prob', 'local_search_step_size'])
                if param_to_mutate == 'F':
                    self.strategies[i]['F'] = np.clip(self.strategies[i]['F'] + np.random.normal(0, 0.1), 0.1, 1.0)
                elif param_to_mutate == 'CR':
                    self.strategies[i]['CR'] = np.clip(self.strategies[i]['CR'] + np.random.normal(0, 0.1), 0.1, 1.0)
                elif param_to_mutate == 'local_search_prob':
                    self.strategies[i]['local_search_prob'] = np.clip(self.strategies[i]['local_search_prob'] + np.random.normal(0, 0.02), 0.0, 0.2)
                elif param_to_mutate == 'local_search_step_size':
                    self.strategies[i]['local_search_step_size'] = np.clip(self.strategies[i]['local_search_step_size'] + np.random.normal(0, 0.005), 0.01, 0.1)

            self.strategy_performance[i] *= 0.9  # Decay performance to favor recent strategies