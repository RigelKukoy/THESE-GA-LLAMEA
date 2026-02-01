import numpy as np

class AdaptiveDE_OL:
    def __init__(self, budget=10000, dim=10, pop_size=50, F_init=0.5, CR_init=0.7, restart_prob=0.05, F_adapt_rate=0.1, CR_adapt_rate=0.1, stagnation_threshold=1000, aging_rate=0.01):
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
        self.aging_rate = aging_rate
        self.age = np.zeros(pop_size)


    def initialize_population(self, func):
        self.pop = np.random.uniform(func.bounds.lb, func.bounds.ub, size=(self.pop_size, self.dim))
        self.fitness = np.array([func(x) for x in self.pop])
        self.eval_count += self.pop_size
        
        best_idx = np.argmin(self.fitness)
        if self.fitness[best_idx] < self.f_opt:
            self.f_opt = self.fitness[best_idx]
            self.x_opt = self.pop[best_idx]

    def orthogonal_learning(self, func, x_current):
        # Generate an orthogonal array design
        oa = self.create_orthogonal_array(self.dim)
        
        # Generate test points based on orthogonal array and current solution
        test_points = self.generate_test_points(x_current, oa, func.bounds.lb, func.bounds.ub)
        
        # Evaluate the test points
        fitness_values = np.array([func(x) for x in test_points])
        self.eval_count += len(test_points)
        
        # Find the best test point
        best_idx = np.argmin(fitness_values)
        best_point = test_points[best_idx]
        best_fitness = fitness_values[best_idx]

        return best_point, best_fitness

    def create_orthogonal_array(self, dim):
        # A simplified orthogonal array creation (example with L9 array for up to 4 factors at 3 levels)
        # For higher dimensions/levels, use a proper orthogonal array library like pyDOE
        if dim <= 4:  # Example for dimensions up to 4
            oa = np.array([
                [0, 0, 0],
                [0, 1, 1],
                [0, 2, 2],
                [1, 0, 1],
                [1, 1, 2],
                [1, 2, 0],
                [2, 0, 2],
                [2, 1, 0],
                [2, 2, 1]
            ])
            if dim < 3:
              oa = oa[:,:dim]
            return oa
        else:
            # Return a random array if dim > 4 (replace with more proper OA)
            return np.random.randint(0, 3, size=(9, dim))

    def generate_test_points(self, x_current, oa, lb, ub):
        num_points, dim = oa.shape
        test_points = np.zeros((num_points, dim))
        
        for i in range(num_points):
            for j in range(dim):
                level = oa[i, j]
                test_points[i, j] = x_current[j] + (level - 1) * (ub - lb) / 4 # Create three levels near x_current

                test_points[i,j] = np.clip(test_points[i,j], lb, ub)
        return test_points


    def evolve(self, func):
        for i in range(self.pop_size):
            if self.eval_count >= self.budget:
                break

            # Mutation
            idxs = np.random.choice(self.pop_size, 3, replace=False)
            x_r1, x_r2, x_r3 = self.pop[idxs]
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

            # Orthogonal learning
            x_ol, f_ol = self.orthogonal_learning(func, x_trial)

            if f_ol < f_trial:
                f_trial = f_ol
                x_trial = x_ol

            if f_trial < self.fitness[i]:
                # Successful adaptation of F and CR
                self.F[i] = np.clip(self.F[i] + self.F_adapt_rate * np.random.normal(0, 0.3), 0.1, 1.0) # Small adaptation
                self.CR[i] = np.clip(self.CR[i] + self.CR_adapt_rate * np.random.normal(0, 0.3), 0.1, 1.0) # Small adaptation

                self.pop[i] = x_trial
                self.fitness[i] = f_trial
                self.age[i] = 0 # Reset age

                if f_trial < self.f_opt:
                    self.f_opt = f_trial
                    self.x_opt = x_trial
            else:
                # Unsuccessful adaptation of F and CR
                self.F[i] = np.clip(self.F[i] - self.F_adapt_rate * np.random.normal(0, 0.3), 0.1, 1.0)  # Reduce F
                self.CR[i] = np.clip(self.CR[i] - self.CR_adapt_rate * np.random.normal(0, 0.3), 0.1, 1.0)  # Reduce CR
                self.age[i] += 1 # Increase age
            

    def restart_population(self, func):
        # Restart all the population except the best individual and some young individuals
        best_idx = np.argmin(self.fitness)

        # Select individuals to keep based on age (favor younger ones)
        num_to_keep = int(self.pop_size * 0.2) # Keep top 20% youngest
        age_sorted_indices = np.argsort(self.age)
        indices_to_keep = age_sorted_indices[:num_to_keep]

        new_pop = np.random.uniform(func.bounds.lb, func.bounds.ub, size=(self.pop_size - len(indices_to_keep) - 1, self.dim))
        new_fitness = np.array([func(x) for x in new_pop])
        self.eval_count += self.pop_size - len(indices_to_keep) - 1

        # Insert the best individual from the previous population and the younger individuals
        temp_pop = np.vstack((self.pop[best_idx], self.pop[indices_to_keep], new_pop))
        temp_fitness = np.hstack((self.fitness[best_idx], self.fitness[indices_to_keep], new_fitness))
        temp_age = np.hstack((self.age[best_idx], self.age[indices_to_keep], np.zeros(len(new_fitness))))

        self.pop = temp_pop
        self.fitness = temp_fitness
        self.age = temp_age
        
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
        self.age = np.zeros(self.pop_size)

        self.initialize_population(func)

        while self.eval_count < self.budget:
            self.evolve(func)

            # Stagnation check
            if self.f_opt < self.previous_best_fitness:
                self.stagnation_counter = 0
                self.previous_best_fitness = self.f_opt
            else:
                self.stagnation_counter += self.pop_size # Increment by population size each generation

            # Aging mechanism: increment age of all individuals
            self.age += self.aging_rate

            if self.stagnation_counter > self.stagnation_threshold:
                self.restart_population(func)
                self.stagnation_counter = 0 # Reset stagnation counter

        return self.f_opt, self.x_opt