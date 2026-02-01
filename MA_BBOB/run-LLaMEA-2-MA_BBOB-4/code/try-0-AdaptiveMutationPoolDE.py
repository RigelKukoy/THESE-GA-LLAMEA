import numpy as np

class AdaptiveMutationPoolDE:
    def __init__(self, budget=10000, dim=10, pop_size=40, Cr=0.9, ls_prob=0.1, mutation_pool_size=5, success_memory=10):
        self.budget = budget
        self.dim = dim
        self.pop_size = pop_size
        self.Cr = Cr
        self.ls_prob = ls_prob  # Probability of applying local search
        self.mutation_pool_size = mutation_pool_size
        self.success_memory = success_memory # Number of generations to track mutation success
        self.best_fitness_history = []
        self.population = None
        self.fitness = None
        self.mutation_strategies = [
            self._mutation_rand1,
            self._mutation_best1,
            self._mutation_current_to_rand1,
            self._mutation_current_to_best1,
            self._mutation_rand2,
        ]
        self.mutation_success_counts = np.zeros(len(self.mutation_strategies))
        self.mutation_usage_counts = np.ones(len(self.mutation_strategies)) #Initialize to 1 to avoid division by zero
        self.mutation_success_history = [[] for _ in range(len(self.mutation_strategies))]

    def __call__(self, func):
        # Initialization
        self.population = np.random.uniform(func.bounds.lb, func.bounds.ub, size=(self.pop_size, self.dim))
        self.fitness = np.array([func(x) for x in self.population])
        self.budget -= self.pop_size
        
        self.f_opt = np.min(self.fitness)
        self.x_opt = self.population[np.argmin(self.fitness)]
        self.best_fitness_history.append(self.f_opt)

        while self.budget > self.pop_size:
            new_population = np.copy(self.population)
            new_fitness = np.copy(self.fitness)
            
            mutation_probs = self.mutation_success_counts / self.mutation_usage_counts
            mutation_probs /= np.sum(mutation_probs)
            
            for i in range(self.pop_size):
                # Adaptive Mutation Strategy Selection
                mutation_index = np.random.choice(len(self.mutation_strategies), p=mutation_probs)
                self.mutation_usage_counts[mutation_index] +=1
                mutant = self.mutation_strategies[mutation_index](i)

                # Crossover
                for j in range(self.dim):
                    if np.random.rand() < self.Cr:
                        new_population[i, j] = mutant[j]
                    else:
                         new_population[i, j] = self.population[i, j]
                
                new_population[i] = np.clip(new_population[i], func.bounds.lb, func.bounds.ub)

            # Evaluation
            new_fitness = np.array([func(x) for x in new_population])
            self.budget -= self.pop_size

            # Selection
            for i in range(self.pop_size):
                if new_fitness[i] < self.fitness[i]:
                    if (new_fitness[i] - self.fitness[i]) / abs(self.fitness[i]) < 0.01: #Success criteria
                        self.mutation_success_counts[mutation_index] += 1
                    self.population[i] = new_population[i]
                    self.fitness[i] = new_fitness[i]

                    if new_fitness[i] < self.f_opt:
                        self.f_opt = new_fitness[i]
                        self.x_opt = new_population[i]
                
            # Local Search
            for i in range(self.pop_size):
                if np.random.rand() < self.ls_prob:
                    x_ls = self._local_search(self.population[i], func)
                    f_ls = func(x_ls)
                    self.budget -=1
                    if f_ls < self.fitness[i]:
                        self.population[i] = x_ls
                        self.fitness[i] = f_ls
                        if f_ls < self.f_opt:
                            self.f_opt = f_ls
                            self.x_opt = x_ls

            self.best_fitness_history.append(self.f_opt)
            self.mutation_success_counts *= 0.95 #Discount success
            self.mutation_success_counts += 0.05


        return self.f_opt, self.x_opt

    def _mutation_rand1(self, i):
        indices = [j for j in range(self.pop_size) if j != i]
        idxs = np.random.choice(indices, size=3, replace=False)
        x_r1, x_r2, x_r3 = self.population[idxs[0]], self.population[idxs[1]], self.population[idxs[2]]
        F = np.random.uniform(0.5, 1.0)
        return self.population[i] + F * (x_r1 - x_r2 + x_r3 - self.population[i])
    
    def _mutation_best1(self, i):
        indices = [j for j in range(self.pop_size) if j != i]
        idxs = np.random.choice(indices, size=2, replace=False)
        x_r1, x_r2 = self.population[idxs[0]], self.population[idxs[1]]
        F = np.random.uniform(0.5, 1.0)
        return self.x_opt + F * (x_r1 - x_r2)

    def _mutation_current_to_rand1(self, i):
        indices = [j for j in range(self.pop_size) if j != i]
        idxs = np.random.choice(indices, size=3, replace=False)
        x_r1, x_r2, x_r3 = self.population[idxs[0]], self.population[idxs[1]], self.population[idxs[2]]
        F = np.random.uniform(0.5, 1.0)
        return self.population[i] + F * (x_r1 - self.population[i] + x_r2 - x_r3)

    def _mutation_current_to_best1(self, i):
         F = np.random.uniform(0.5, 1.0)
         return self.population[i] + F * (self.x_opt - self.population[i])

    def _mutation_rand2(self, i):
        indices = [j for j in range(self.pop_size) if j != i]
        idxs = np.random.choice(indices, size=5, replace=False)
        x_r1, x_r2, x_r3, x_r4, x_r5 = self.population[idxs[0]], self.population[idxs[1]], self.population[idxs[2]], self.population[idxs[3]], self.population[idxs[4]]
        F = np.random.uniform(0.5, 1.0)
        return self.population[i] + F * (x_r1 - x_r2 + x_r3 - x_r4)

    def _local_search(self, x, func, radius=0.1, num_samples=5):
        best_x = x
        best_f = func(x)
        for _ in range(num_samples):
            x_new = x + np.random.uniform(-radius, radius, size=self.dim)
            x_new = np.clip(x_new, func.bounds.lb, func.bounds.ub)
            f_new = func(x_new)
            if f_new < best_f:
                best_f = f_new
                best_x = x_new
        return best_x