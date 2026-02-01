import numpy as np

class AdaptiveDELocalSearch:
    def __init__(self, budget=10000, dim=10, pop_size=50, F=0.5, CR=0.9, local_search_prob=0.1):
        self.budget = budget
        self.dim = dim
        self.pop_size = pop_size
        self.F = F
        self.CR = CR
        self.local_search_prob = local_search_prob
        self.mutation_strategies = [self._mutation_rand1, self._mutation_current_to_best_1, self._mutation_best1]
        self.success_counts = [1] * len(self.mutation_strategies)
        self.total_counts = [1] * len(self.mutation_strategies)
        self.epsilon = 1e-6
        self.archive = []
        self.archive_fitness = []

    def _mutation_rand1(self, population, best_idx):
        idxs = np.random.choice(self.pop_size, 3, replace=False)
        return population[idxs[0]] + self.F * (population[idxs[1]] - population[idxs[2]])

    def _mutation_current_to_best_1(self, population, best_idx, i):
        idxs = np.random.choice(self.pop_size, 2, replace=False)
        return population[i] + self.F * (population[best_idx] - population[i]) + self.F * (population[idxs[0]] - population[idxs[1]])

    def _mutation_best1(self, population, best_idx):
        idxs = np.random.choice(self.pop_size, 2, replace=False)
        return population[best_idx] + self.F * (population[idxs[0]] - population[idxs[1]])

    def _local_search(self, x, func):
        # Simple random walk local search
        step_size = 0.1 * (func.bounds.ub - func.bounds.lb)
        new_x = x + np.random.uniform(-step_size, step_size, size=self.dim)
        new_x = np.clip(new_x, func.bounds.lb, func.bounds.ub)
        return new_x

    def __call__(self, func):
        self.f_opt = np.Inf
        self.x_opt = None
        
        # Initialize population
        population = np.random.uniform(func.bounds.lb, func.bounds.ub, size=(self.pop_size, self.dim))
        fitness = np.array([func(x) for x in population])
        used_budget = self.pop_size
        
        best_idx = np.argmin(fitness)
        self.f_opt = fitness[best_idx]
        self.x_opt = population[best_idx]
        
        while used_budget < self.budget:
            for i in range(self.pop_size):
                # Adaptive mutation strategy selection
                probabilities = np.array(self.success_counts) / np.array(self.total_counts)
                probabilities /= np.sum(probabilities)
                mutation_idx = np.random.choice(len(self.mutation_strategies), p=probabilities)
                
                if mutation_idx == 0:
                    mutant = self._mutation_rand1(population, best_idx)
                elif mutation_idx == 1:
                    mutant = self._mutation_current_to_best_1(population, best_idx, i)
                else:
                    mutant = self._mutation_best1(population, best_idx)
                    
                mutant = np.clip(mutant, func.bounds.lb, func.bounds.ub)
                
                # Crossover
                crossover_mask = np.random.rand(self.dim) < self.CR
                trial_vector = np.where(crossover_mask, mutant, population[i])
                
                # Local Search
                if np.random.rand() < self.local_search_prob:
                    trial_vector = self._local_search(trial_vector, func)
                
                # Evaluation
                f = func(trial_vector)
                used_budget += 1
                
                # Selection
                if f < fitness[i]:
                    fitness[i] = f
                    population[i] = trial_vector
                    self.success_counts[mutation_idx] += 1
                    
                    if f < self.f_opt:
                        self.f_opt = f
                        self.x_opt = trial_vector
                
                self.total_counts[mutation_idx] += 1
            
            best_idx = np.argmin(fitness)
                
        return self.f_opt, self.x_opt