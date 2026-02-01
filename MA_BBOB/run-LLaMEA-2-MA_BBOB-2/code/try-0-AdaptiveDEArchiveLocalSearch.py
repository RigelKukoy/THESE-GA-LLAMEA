import numpy as np

class AdaptiveDEArchiveLocalSearch:
    def __init__(self, budget=10000, dim=10, pop_size=50, F=0.5, CR=0.9, archive_size=10, local_search_prob=0.1, local_search_decay=0.99):
        self.budget = budget
        self.dim = dim
        self.pop_size = pop_size
        self.F = F
        self.CR = CR
        self.archive_size = archive_size
        self.local_search_prob = local_search_prob
        self.local_search_decay = local_search_decay # Decay rate for local search probability
        self.mutation_strategies = [self._mutation_rand1, self._mutation_current_to_best_1, self._mutation_best1, self._mutation_rand_archive]
        self.success_counts = [1] * len(self.mutation_strategies)
        self.total_counts = [1] * len(self.mutation_strategies)
        self.epsilon = 1e-6
        self.archive = []
        self.archive_fitness = []
        self.archive_full = False


    def _mutation_rand1(self, population, best_idx):
        idxs = np.random.choice(self.pop_size, 3, replace=False)
        return population[idxs[0]] + self.F * (population[idxs[1]] - population[idxs[2]])

    def _mutation_current_to_best_1(self, population, best_idx, i):
        idxs = np.random.choice(self.pop_size, 2, replace=False)
        return population[i] + self.F * (population[best_idx] - population[i]) + self.F * (population[idxs[0]] - population[idxs[1]])

    def _mutation_best1(self, population, best_idx):
        idxs = np.random.choice(self.pop_size, 2, replace=False)
        return population[best_idx] + self.F * (population[idxs[0]] - population[idxs[1]])

    def _mutation_rand_archive(self, population):
        if not self.archive:  # Ensure archive is not empty
            return self._mutation_rand1(population, np.random.randint(self.pop_size))
        
        idx1 = np.random.choice(self.pop_size, 1, replace=False)[0]
        idx_archive = np.random.choice(len(self.archive), 1, replace=False)[0]
        idxs = np.random.choice(self.pop_size, 1, replace=False)[0]  # choosing only one index
        return population[idx1] + self.F * (self.archive[idx_archive] - population[idxs])


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
                elif mutation_idx == 2:
                    mutant = self._mutation_best1(population, best_idx)
                else:
                    mutant = self._mutation_rand_archive(population)
                    
                mutant = np.clip(mutant, func.bounds.lb, func.bounds.ub)
                
                # Crossover
                crossover_mask = np.random.rand(self.dim) < self.CR
                trial_vector = np.where(crossover_mask, mutant, population[i])
                
                # Local Search
                if np.random.rand() < self.local_search_prob:
                    trial_vector = self._local_search(trial_vector, func)
                    self.local_search_prob *= self.local_search_decay # Reduce local search intensity
                
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

                    # Update archive
                    if self.archive_full:
                        if f < np.max(self.archive_fitness):
                            worst_idx = np.argmax(self.archive_fitness)
                            self.archive[worst_idx] = trial_vector
                            self.archive_fitness[worst_idx] = f
                    else:
                        self.archive.append(trial_vector)
                        self.archive_fitness.append(f)
                        if len(self.archive) == self.archive_size:
                            self.archive_full = True
                
                self.total_counts[mutation_idx] += 1
            
            best_idx = np.argmin(fitness)
                
        return self.f_opt, self.x_opt