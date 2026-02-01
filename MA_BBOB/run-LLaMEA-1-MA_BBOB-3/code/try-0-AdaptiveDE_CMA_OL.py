import numpy as np

class AdaptiveDE_CMA_OL:
    def __init__(self, budget=10000, dim=10, pop_size_init=50, pop_size_min=10, pop_size_max=100, F_init=0.5, CR_init=0.7, restart_prob=0.05, F_adapt_rate=0.1, CR_adapt_rate=0.1, stagnation_threshold=1000, archive_size=10, cma_learning_rate=0.1, orthogonal_learning_rate=0.1):
        self.budget = budget
        self.dim = dim
        self.pop_size_init = pop_size_init
        self.pop_size_min = pop_size_min
        self.pop_size_max = pop_size_max
        self.pop_size = pop_size_init
        self.F = np.full(pop_size_init, F_init)  # Mutation factor for each individual
        self.CR = np.full(pop_size_init, CR_init)  # Crossover rate for each individual
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
        self.archive = []
        self.archive_fitness = []
        self.cma_learning_rate = cma_learning_rate
        self.C = np.eye(dim) # Covariance matrix for CMA-ES-like adaptation
        self.orthogonal_learning_rate = orthogonal_learning_rate

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
            # Replace the worst member in the archive
            worst_idx = np.argmax(self.archive_fitness)
            if f < self.archive_fitness[worst_idx]:
                self.archive[worst_idx] = x
                self.archive_fitness[worst_idx] = f

    def orthogonal_learning(self, func, x_current):
        # Generate orthogonal array
        orthogonal_matrix = self.generate_orthogonal_array(self.dim)
        
        trial_points = []
        for row in orthogonal_matrix:
            x_trial = x_current.copy()
            for j in range(self.dim):
                # Perturb each dimension based on the orthogonal array
                perturbation = (row[j] - 0.5) * self.orthogonal_learning_rate  # Scale perturbation
                x_trial[j] = x_current[j] + perturbation
                x_trial[j] = np.clip(x_trial[j], func.bounds.lb, func.bounds.ub)  # Clip to bounds
            trial_points.append(x_trial)
            
        # Evaluate trial points
        fitness_values = [func(x) for x in trial_points]
        self.eval_count += len(trial_points)
        
        # Select the best point
        best_idx = np.argmin(fitness_values)
        x_best_orthogonal = trial_points[best_idx]
        f_best_orthogonal = fitness_values[best_idx]
        
        return x_best_orthogonal, f_best_orthogonal

    def generate_orthogonal_array(self, dim):
        # A simple method to generate an orthogonal array.  Can be improved.
        return np.random.randint(0, 2, size=(dim + 1, dim))

    def evolve(self, func):
        ranked_indices = np.argsort(self.fitness)
        
        for i in range(self.pop_size):
            if self.eval_count >= self.budget:
                break

            # Mutation
            idxs = np.random.choice(self.pop_size, 3, replace=False)
            x_r1, x_r2, x_r3 = self.pop[idxs]
            
            # Rank-based selection of x_r1: Prefer better individuals
            rank_index = ranked_indices[np.random.randint(0, self.pop_size // 2)] # Choose from the top half
            x_r1 = self.pop[rank_index]

            # Use CMA-ES-like sampling
            z = np.random.multivariate_normal(np.zeros(self.dim), self.C)
            x_mutated = x_r1 + self.F[i] * (x_r2 - x_r3) + np.sqrt(self.F[i]) * z # Add CMA-ES-like exploration

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

            # Orthogonal Learning
            x_orthogonal, f_orthogonal = self.orthogonal_learning(func, x_trial)

            if f_orthogonal < f_trial:
                x_trial = x_orthogonal
                f_trial = f_orthogonal

            if f_trial < self.fitness[i]:
                # Successful adaptation of F and CR
                self.F[i] = np.clip(self.F[i] + self.F_adapt_rate * np.random.normal(0, 0.3), 0.1, 1.0) # Small adaptation
                self.CR[i] = np.clip(self.CR[i] + self.CR_adapt_rate * np.random.normal(0, 0.3), 0.1, 1.0) # Small adaptation

                self.pop[i] = x_trial
                self.fitness[i] = f_trial
                
                if f_trial < self.f_opt:
                    self.f_opt = f_trial
                    self.x_opt = x_trial

                # Update CMA-ES covariance matrix
                diff = x_trial - self.pop[i]
                self.C = (1 - self.cma_learning_rate) * self.C + self.cma_learning_rate * np.outer(diff, diff)

            else:
                # Unsuccessful adaptation of F and CR
                self.F[i] = np.clip(self.F[i] - self.F_adapt_rate * np.random.normal(0, 0.3), 0.1, 1.0)  # Reduce F
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
            
        # Reset CMA-ES covariance matrix upon restart
        self.C = np.eye(self.dim)

    def adapt_population_size(self):
        # Adapt population size based on stagnation
        if self.stagnation_counter > self.stagnation_threshold / 2:
            self.pop_size = max(self.pop_size - 5, self.pop_size_min)  # Reduce pop size
        else:
            self.pop_size = min(self.pop_size + 2, self.pop_size_max)  # Increase pop size

        # Ensure F and CR are properly sized
        if len(self.F) != self.pop_size:
            self.F = np.full(self.pop_size, np.mean(self.F))
        if len(self.CR) != self.pop_size:
            self.CR = np.full(self.pop_size, np.mean(self.CR))

    def __call__(self, func):
        self.f_opt = np.Inf
        self.x_opt = None
        self.eval_count = 0
        self.stagnation_counter = 0
        self.previous_best_fitness = np.Inf
        self.archive = []
        self.archive_fitness = []
        self.C = np.eye(self.dim) # Reset CMA-ES covariance matrix
        self.pop_size = self.pop_size_init # Reset population size

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

            self.adapt_population_size()


        return self.f_opt, self.x_opt