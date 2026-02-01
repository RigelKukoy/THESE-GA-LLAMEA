import numpy as np

class AdaptiveDE:
    def __init__(self, budget=10000, dim=10, pop_size=50, F_init=0.5, CR_init=0.7, restart_prob=0.05, F_adapt_rate=0.1, CR_adapt_rate=0.1, stagnation_threshold=1000, archive_size=10):
        self.budget = budget
        self.dim = dim
        self.pop_size = pop_size
        self.F = np.full(pop_size, F_init)
        self.CR = np.full(pop_size, CR_init)
        self.restart_prob = restart_prob
        self.F_adapt_rate = F_adapt_rate
        self.CR_adapt_rate = CR_adapt_rate
        self.pop = None
        self.fitness = None
        self.f_opt = np.Inf
        self.x_opt = None
        self.eval_count = 0
        self.stagnation_counter = 0
        self.previous_best_fitness = np.Inf
        self.stagnation_threshold = stagnation_threshold
        self.archive_size = archive_size
        self.archive = []  # Archive to store past solutions


    def initialize_population(self, func):
        self.pop = np.random.uniform(func.bounds.lb, func.bounds.ub, size=(self.pop_size, self.dim))
        self.fitness = np.array([func(x) for x in self.pop])
        self.eval_count += self.pop_size
        
        best_idx = np.argmin(self.fitness)
        if self.fitness[best_idx] < self.f_opt:
            self.f_opt = self.fitness[best_idx]
            self.x_opt = self.pop[best_idx]

    def evolve(self, func):
        for i in range(self.pop_size):
            if self.eval_count >= self.budget:
                break

            # Mutation: Tournament selection to increase diversity
            idxs = np.random.choice(self.pop_size, 2, replace=False) # Select 2 individuals
            if self.fitness[idxs[0]] < self.fitness[idxs[1]]:
                x_r1 = self.pop[idxs[0]]
            else:
                x_r1 = self.pop[idxs[1]]
            
            # Choose the other two indices from population and archive
            indices = list(range(self.pop_size))
            indices.remove(i)
            
            if len(self.archive) > 0:
              combined_pop = np.vstack((self.pop[indices], self.archive))
              rand_idx = np.random.choice(len(combined_pop), 2, replace=False)

              x_r2 = combined_pop[rand_idx[0]]
              x_r3 = combined_pop[rand_idx[1]]
            else:
              idxs = np.random.choice(indices, 2, replace=False)
              x_r2 = self.pop[idxs[0]]
              x_r3 = self.pop[idxs[1]]

            x_mutated = x_r1 + self.F[i] * (x_r2 - x_r3)
            x_mutated = np.clip(x_mutated, func.bounds.lb, func.bounds.ub)

            # Crossover
            x_trial = self.pop[i].copy()
            j_rand = np.random.randint(self.dim)
            for j in range(self.dim):
                if np.random.rand() < self.CR[i] or j == j_rand:
                    x_trial[j] = x_mutated[j]

            # Selection
            f_trial = func(x_trial)
            self.eval_count += 1

            if f_trial < self.fitness[i]:
                # Successful adaptation of F and CR
                self.F[i] = np.clip(self.F[i] + self.F_adapt_rate * np.random.normal(0, 0.3), 0.1, 1.0)
                self.CR[i] = np.clip(self.CR[i] + self.CR_adapt_rate * np.random.normal(0, 0.3), 0.1, 1.0)

                self.pop[i] = x_trial
                self.fitness[i] = f_trial
                
                if f_trial < self.f_opt:
                    self.f_opt = f_trial
                    self.x_opt = x_trial

                # Update archive
                if len(self.archive) < self.archive_size:
                    self.archive.append(x_trial)
                else:
                    # Replace a random element in the archive
                    idx_to_replace = np.random.randint(self.archive_size)
                    self.archive[idx_to_replace] = x_trial

            else:
                # Unsuccessful adaptation of F and CR
                self.F[i] = np.clip(self.F[i] - self.F_adapt_rate * np.random.normal(0, 0.3), 0.1, 1.0)
                self.CR[i] = np.clip(self.CR[i] - self.CR_adapt_rate * np.random.normal(0, 0.3), 0.1, 1.0)

    def restart_population(self, func):
        # Restart all the population except the best individual
        best_idx = np.argmin(self.fitness)
        
        new_pop = np.random.uniform(func.bounds.lb, func.bounds.ub, size=(self.pop_size -1 , self.dim))
        new_fitness = np.array([func(x) for x in new_pop])
        self.eval_count += self.pop_size -1

        # Insert the best individual from the previous population
        temp_pop = np.vstack((self.pop[best_idx], new_pop))
        temp_fitness = np.hstack((self.fitness[best_idx], new_fitness))

        self.pop = temp_pop
        self.fitness = temp_fitness
        
        best_idx = np.argmin(self.fitness)
        if self.fitness[best_idx] < self.f_opt:
            self.f_opt = self.fitness[best_idx]
            self.x_opt = self.pop[best_idx]

    def __call__(self, func):
        self.f_opt = np.Inf
        self.x_opt = None
        self.eval_count = 0
        self.stagnation_counter = 0
        self.previous_best_fitness = np.Inf
        self.archive = []  # Reset archive

        self.initialize_population(func)

        while self.eval_count < self.budget:
            self.evolve(func)

            # Stagnation check
            if self.f_opt < self.previous_best_fitness:
                self.stagnation_counter = 0
                self.previous_best_fitness = self.f_opt
            else:
                self.stagnation_counter += self.pop_size

            if self.stagnation_counter > self.stagnation_threshold:
                self.restart_population(func)
                self.stagnation_counter = 0


        return self.f_opt, self.x_opt