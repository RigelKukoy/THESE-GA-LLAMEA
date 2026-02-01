import numpy as np

class AdaptiveDE_Archive:
    def __init__(self, budget=10000, dim=10, pop_size=50, archive_size=50, F_init=0.5, CR_init=0.7, restart_prob=0.05, F_adapt_rate=0.1, CR_adapt_rate=0.1, stagnation_threshold=1000):
        self.budget = budget
        self.dim = dim
        self.pop_size = pop_size
        self.archive_size = archive_size
        self.F = np.full(pop_size, F_init)  # Mutation factor for each individual
        self.CR = np.full(pop_size, CR_init)  # Crossover rate for each individual
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
        self.archive = []
        self.archive_fitness = []


    def initialize_population(self, func):
        self.pop = np.random.uniform(func.bounds.lb, func.bounds.ub, size=(self.pop_size, self.dim))
        self.fitness = np.array([func(x) for x in self.pop])
        self.eval_count += self.pop_size
        
        best_idx = np.argmin(self.fitness)
        if self.fitness[best_idx] < self.f_opt:
            self.f_opt = self.fitness[best_idx]
            self.x_opt = self.pop[best_idx]

    def update_archive(self, x, f):
        if len(self.archive) < self.archive_size:
            self.archive.append(x)
            self.archive_fitness.append(f)
        else:
            # Replace a random element in the archive
            idx = np.random.randint(self.archive_size)
            self.archive[idx] = x
            self.archive_fitness[idx] = f
            
    def distance_based_F(self, x):
        # Calculate distances to other population members and archive members
        distances = np.linalg.norm(self.pop - x, axis=1)
        if len(self.archive) > 0:
            archive_distances = np.linalg.norm(np.array(self.archive) - x, axis=1)
            distances = np.concatenate((distances, archive_distances))

        # Use inverse distance to weight F values (closer individuals have more influence)
        weights = 1.0 / (distances + 1e-6)  # Add a small constant to avoid division by zero
        
        # Sample F value based on weights (example: weighted average)
        if len(self.archive) > 0:
            all_F = np.concatenate((self.F, np.full(len(self.archive), np.mean(self.F))))
        else:
            all_F = self.F
            
        F = np.average(all_F, weights=weights)
        return np.clip(F, 0.1, 1.0)
    
    def orthogonal_crossover(self, x_mutated, x_target):
        # Perform orthogonal crossover
        x_trial = x_target.copy()
        
        # Select two random indices
        idx1, idx2 = np.random.choice(self.dim, 2, replace=False)
        
        # Perform orthogonal array based crossover
        x_trial[idx1] = x_mutated[idx1]
        x_trial[idx2] = x_mutated[idx2]

        return x_trial


    def evolve(self, func):
        for i in range(self.pop_size):
            if self.eval_count >= self.budget:
                break

            # Mutation
            idxs = np.random.choice(self.pop_size, 3, replace=False)
            x_r1, x_r2, x_r3 = self.pop[idxs]
            
            # Adaptive F based on distance
            F = self.distance_based_F(self.pop[i])
            x_mutated = x_r1 + F * (x_r2 - x_r3)
            x_mutated = np.clip(x_mutated, func.bounds.lb, func.bounds.ub)

            # Crossover
            #x_trial = self.pop[i].copy()
            #j_rand = np.random.randint(self.dim)
            #for j in range(self.dim):
            #    if np.random.rand() < self.CR[i] or j == j_rand:
            #        x_trial[j] = x_mutated[j]
            
            x_trial = self.orthogonal_crossover(x_mutated, self.pop[i])


            # Selection
            f_trial = func(x_trial)
            self.eval_count += 1

            if f_trial < self.fitness[i]:
                # Successful adaptation of CR
                self.CR[i] = np.clip(self.CR[i] + self.CR_adapt_rate * np.random.normal(0, 0.3), 0.1, 1.0) # Small adaptation

                self.pop[i] = x_trial
                self.fitness[i] = f_trial
                
                if f_trial < self.f_opt:
                    self.f_opt = f_trial
                    self.x_opt = x_trial
                
                self.update_archive(x_trial, f_trial)
            else:
                # Unsuccessful adaptation of CR
                self.CR[i] = np.clip(self.CR[i] - self.CR_adapt_rate * np.random.normal(0, 0.3), 0.1, 1.0)  # Reduce CR
                self.update_archive(self.pop[i], self.fitness[i])

                    
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
        self.archive = []
        self.archive_fitness = []

        self.initialize_population(func)

        while self.eval_count < self.budget:
            self.evolve(func)

            # Stagnation check
            if self.f_opt < self.previous_best_fitness:
                self.stagnation_counter = 0
                self.previous_best_fitness = self.f_opt
            else:
                self.stagnation_counter += self.pop_size # Increment by population size each generation

            if self.stagnation_counter > self.stagnation_threshold:
                self.restart_population(func)
                self.stagnation_counter = 0 # Reset stagnation counter


        return self.f_opt, self.x_opt