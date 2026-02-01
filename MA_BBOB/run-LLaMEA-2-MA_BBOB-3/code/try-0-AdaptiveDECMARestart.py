import numpy as np

class AdaptiveDECMARestart:
    def __init__(self, budget=10000, dim=10, pop_size=20, initial_mutation_factor=0.5, crossover_rate=0.7, local_search_iterations=5, restart_trigger=0.01):
        self.budget = budget
        self.dim = dim
        self.pop_size = pop_size
        self.mutation_factor = initial_mutation_factor
        self.crossover_rate = crossover_rate
        self.local_search_iterations = local_search_iterations
        self.restart_trigger = restart_trigger
        self.population = np.random.uniform(-5, 5, size=(pop_size, dim))
        self.fitness = np.zeros(pop_size)
        self.best_position = None
        self.best_fitness = np.inf
        self.eval_count = 0
        self.initial_mutation_factor = initial_mutation_factor

    def covariance_matrix_adaptation(self, func, x_mean, sigma, num_iterations):
        """
        Local search using Covariance Matrix Adaptation.
        """
        dim = self.dim
        C = np.eye(dim)  # Initialize covariance matrix
        path_c = np.zeros(dim)
        path_sigma = np.zeros(dim)
        mu_eff = self.pop_size / 4  # Effective population size

        c_sigma = (mu_eff + 2) / (dim + mu_eff + 5)
        c_c = (4 + mu_eff / dim) / (dim + 4 + 2 * mu_eff / dim)
        c_1 = 2 / ((dim + 1.3)**2 + mu_eff)
        c_mu = min(1 - c_1, 2 * (mu_eff - 2 + 1 / mu_eff) / ((dim + 2)**2 + mu_eff))
        d_sigma = 1 + 2 * max(0, np.sqrt((mu_eff - 1) / (dim + 1)) - 1) + c_sigma

        best_x = x_mean
        best_f = func(x_mean) if self.eval_count < self.budget else np.inf
        if self.eval_count < self.budget:
            self.eval_count += 1

        for _ in range(num_iterations):
            # Generate samples
            z = np.random.multivariate_normal(np.zeros(dim), C, self.pop_size)
            x = x_mean + sigma * z
            x = np.clip(x, func.bounds.lb, func.bounds.ub)
            f = np.array([func(xi) if self.eval_count < self.budget else np.inf for xi in x])
            self.eval_count += np.sum(f != np.inf) # correct eval count

            if self.eval_count >= self.budget:
                break

            # Selection and update
            idx = np.argsort(f)
            x_mean_new = np.mean(x[idx[:int(mu_eff)]], axis=0)
            
            if f[idx[0]] < best_f:
                best_f = f[idx[0]]
                best_x = x[idx[0]]
                
            # Update evolution path
            z_mean = np.mean(z[idx[:int(mu_eff)]], axis=0)
            path_sigma = (1 - c_sigma) * path_sigma + np.sqrt(c_sigma * (2 - c_sigma)) * z_mean
            path_c = (1 - c_c) * path_c + np.sqrt(c_c * (2 - c_c)) * (np.sqrt(mu_eff) / sigma) * (x_mean_new - x_mean)

            # Update covariance matrix
            C = (1 - c_1 - c_mu) * C + c_1 * np.outer(path_c, path_c) + c_mu * np.sum([np.outer(z[idx[i]], z[idx[i]]) for i in range(int(mu_eff))], axis=0)
            sigma *= np.exp((c_sigma / d_sigma) * (np.linalg.norm(path_sigma) / np.sqrt(dim) - 1))
            x_mean = x_mean_new
        
        return best_f, best_x

    def __call__(self, func):
        self.eval_count = 0
        self.mutation_factor = self.initial_mutation_factor

        # Initialize fitness values
        for i in range(self.pop_size):
            if self.eval_count < self.budget:
                f = func(self.population[i])
                self.eval_count += 1
                self.fitness[i] = f
                if f < self.best_fitness:
                    self.best_fitness = f
                    self.best_position = self.population[i].copy()

        last_improvement = 0
        while self.eval_count < self.budget:
            for i in range(self.pop_size):
                # Differential Evolution
                r1, r2, r3 = np.random.choice(self.pop_size, 3, replace=False)
                mutant_vector = self.population[r1] + self.mutation_factor * (self.population[r2] - self.population[r3])
                mutant_vector = np.clip(mutant_vector, func.bounds.lb, func.bounds.ub)

                # Crossover
                trial_vector = np.zeros(self.dim)
                for j in range(self.dim):
                    if np.random.rand() < self.crossover_rate or j == np.random.randint(self.dim):
                        trial_vector[j] = mutant_vector[j]
                    else:
                        trial_vector[j] = self.population[i][j]

                # Evaluate trial vector
                f_trial = func(trial_vector) if self.eval_count < self.budget else np.inf
                if self.eval_count < self.budget:
                    self.eval_count += 1
                    if f_trial < self.fitness[i]:
                        self.population[i] = trial_vector
                        self.fitness[i] = f_trial
                        if f_trial < self.best_fitness:
                            self.best_fitness = f_trial
                            self.best_position = self.population[i].copy()
                            last_improvement = self.eval_count

            # Covariance Matrix Adaptation Local Search
            local_search_fitness, local_search_position = self.covariance_matrix_adaptation(func, self.best_position, 0.1, self.local_search_iterations)

            if local_search_fitness < self.best_fitness:
                    self.best_fitness = local_search_fitness
                    self.best_position = local_search_position.copy()
                    last_improvement = self.eval_count

            # Restart mechanism
            if (self.eval_count - last_improvement) > self.restart_trigger * self.budget:
                self.population = np.random.uniform(-5, 5, size=(self.pop_size, self.dim))
                for i in range(self.pop_size):
                    if self.eval_count < self.budget:
                        f = func(self.population[i])
                        self.eval_count += 1
                        self.fitness[i] = f
                        if f < self.best_fitness:
                            self.best_fitness = f
                            self.best_position = self.population[i].copy()
                last_improvement = self.eval_count
                self.mutation_factor = self.initial_mutation_factor  # Reset mutation factor

            # Adaptive Mutation Factor (simple heuristic)
            if np.random.rand() < 0.1:
                self.mutation_factor = np.clip(self.mutation_factor + np.random.normal(0, 0.1), 0.1, 1.0)

        return self.best_fitness, self.best_position