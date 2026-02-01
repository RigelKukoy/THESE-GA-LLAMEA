import numpy as np

class AdaptiveDE_CMA_Niching:
    def __init__(self, budget=10000, dim=10, pop_size_init=50, pop_size_min=20, pop_size_max=100, F_init=0.5, CR_init=0.7, restart_prob=0.05, F_adapt_rate=0.1, CR_adapt_rate=0.1, stagnation_threshold=1000, archive_size=10, cma_learning_rate=0.1, niche_radius=0.5):
        self.budget = budget
        self.dim = dim
        self.pop_size_init = pop_size_init
        self.pop_size_min = pop_size_min
        self.pop_size_max = pop_size_max
        self.pop_size = pop_size_init
        self.F = np.full(self.pop_size, F_init)  # Mutation factor for each individual
        self.CR = np.full(self.pop_size, CR_init)  # Crossover rate for each individual
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
        self.niche_radius = niche_radius

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

    def calculate_crowding_distance(self):
        distances = np.zeros(self.pop_size)
        for i in range(self.pop_size):
            for j in range(self.pop_size):
                if i != j:
                    distance = np.linalg.norm(self.pop[i] - self.pop[j])
                    if distance < self.niche_radius:
                        distances[i] += 1  # Count neighbors within the niche radius
        return distances

    def evolve(self, func):
        crowding_distances = self.calculate_crowding_distance()
        ranked_indices = np.argsort(self.fitness + crowding_distances * 0.001) # Promote diversity

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
        crowding_distances = self.calculate_crowding_distance()
        avg_crowding = np.mean(crowding_distances)

        if avg_crowding > 2:  # High crowding, reduce population
            self.pop_size = max(self.pop_size - 5, self.pop_size_min)
        elif avg_crowding < 1:  # Low crowding, increase population
            self.pop_size = min(self.pop_size + 5, self.pop_size_max)
        
        #Ensure population size doesn't drop to 0
        self.pop_size = max(1, self.pop_size)
        
        # Resize F and CR arrays
        self.F = np.resize(self.F, self.pop_size)
        self.CR = np.resize(self.CR, self.pop_size)

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
            self.adapt_population_size()

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