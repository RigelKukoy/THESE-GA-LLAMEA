import numpy as np

class EnhancedAdaptiveDE:
    def __init__(self, budget=10000, dim=10, pop_size=50, F_init=0.5, CR_init=0.7, restart_prob=0.05, F_adapt_rate=0.1, CR_adapt_rate=0.1, stagnation_threshold=1000, pop_reduce_factor=0.9):
        self.budget = budget
        self.dim = dim
        self.pop_size = pop_size
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
        self.pop_reduce_factor = pop_reduce_factor  # Factor to reduce population size during stagnation
        self.min_pop_size = 10  # Minimum population size

    def initialize_population(self, func):
        self.pop = np.random.uniform(func.bounds.lb, func.bounds.ub, size=(self.pop_size, self.dim))
        self.fitness = np.array([func(x) for x in self.pop])
        self.eval_count += self.pop_size
        
        best_idx = np.argmin(self.fitness)
        if self.fitness[best_idx] < self.f_opt:
            self.f_opt = self.fitness[best_idx]
            self.x_opt = self.pop[best_idx]

    def mirrored_boundary_handling(self, x, lb, ub):
        """Handles boundaries using a mirrored strategy."""
        x_corrected = x.copy()
        for i in range(len(x)):
            if x[i] < lb:
                x_corrected[i] = lb + (lb - x[i])
            elif x[i] > ub:
                x_corrected[i] = ub - (x[i] - ub)
        return x_corrected
    
    def orthogonal_crossover(self, x_target, x_mutated):
        """Performs orthogonal crossover."""
        num_groups = 3 # increased groups for potentially better exploration.
        group_size = self.dim // num_groups
        x_trial = x_target.copy()

        for g in range(num_groups):
            start_idx = g * group_size
            end_idx = (g + 1) * group_size if g < num_groups - 1 else self.dim

            if np.random.rand() < self.CR[i]: #Apply with certain prob, might also adapt this.
               for j in range(start_idx, end_idx):
                   x_trial[j] = x_mutated[j]
        return x_trial

    def evolve(self, func):
        for i in range(self.pop_size):
            if self.eval_count >= self.budget:
                break

            # Mutation
            idxs = np.random.choice(self.pop_size, 3, replace=False)
            x_r1, x_r2, x_r3 = self.pop[idxs]
            x_mutated = x_r1 + self.F[i] * (x_r2 - x_r3)
            x_mutated = self.mirrored_boundary_handling(x_mutated, func.bounds.lb, func.bounds.ub)

            # Crossover
            x_trial = self.orthogonal_crossover(self.pop[i], x_mutated) #Use orthogonal crossover

            # Selection
            f_trial = func(x_trial)
            self.eval_count += 1

            if f_trial < self.fitness[i]:
                # Successful adaptation of F and CR
                self.F[i] = np.clip(self.F[i] + self.F_adapt_rate * np.random.normal(0, 0.3), 0.1, 1.0) # Small adaptation
                self.CR[i] = np.clip(self.CR[i] + self.CR_adapt_rate * np.random.normal(0, 0.3), 0.1, 1.0) # Small adaptation

                self.pop[i] = x_trial
                self.fitness[i] = f_trial
                
                if f_trial < self.f_opt:
                    self.f_opt = f_trial
                    self.x_opt = x_trial
            else:
                # Unsuccessful adaptation of F and CR
                self.F[i] = np.clip(self.F[i] - self.F_adapt_rate * np.random.normal(0, 0.3), 0.1, 1.0)  # Reduce F
                self.CR[i] = np.clip(self.CR[i] - self.CR_adapt_rate * np.random.normal(0, 0.3), 0.1, 1.0)  # Reduce CR

    def reduce_population_size(self):
        """Reduces the population size."""
        if self.pop_size > self.min_pop_size:
            new_pop_size = int(self.pop_size * self.pop_reduce_factor)
            new_pop_size = max(new_pop_size, self.min_pop_size)  # Ensure it doesn't go below minimum

            # Select the best individuals to keep
            best_indices = np.argsort(self.fitness)[:new_pop_size]
            self.pop = self.pop[best_indices]
            self.fitness = self.fitness[best_indices]
            self.pop_size = new_pop_size
            self.F = self.F[best_indices]
            self.CR = self.CR[best_indices]

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
                self.reduce_population_size()
                self.restart_population(func) #restart after reduction
                self.stagnation_counter = 0 # Reset stagnation counter


        return self.f_opt, self.x_opt