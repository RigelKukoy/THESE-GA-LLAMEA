import numpy as np

class SOSpDE:
    def __init__(self, budget=10000, dim=10, pop_size=50, num_species=5):
        self.budget = budget
        self.dim = dim
        self.pop_size = pop_size
        self.num_species = num_species
        self.species = [[] for _ in range(self.num_species)]
        self.centroids = None
        self.mutation_strategies = [self.mutation_rand1, self.mutation_current_to_best, self.mutation_best2]
        self.mutation_probs = np.ones(len(self.mutation_strategies)) / len(self.mutation_strategies)
        self.success_counts = np.zeros(len(self.mutation_strategies))
        self.strategy_usage = np.zeros(len(self.mutation_strategies))
        self.learning_rate = 0.1
        self.F = 0.5
        self.CR = 0.7
        self.local_search_prob = 0.1

    def initialize_species(self, population):
        distances = np.linalg.norm(population[:, None, :] - self.centroids[None, :, :], axis=2)
        species_ids = np.argmin(distances, axis=1)
        for i in range(self.pop_size):
            self.species[species_ids[i]].append(i)

    def update_centroids(self, population):
        for i in range(self.num_species):
            if self.species[i]:
                self.centroids[i] = np.mean(population[self.species[i]], axis=0)
            else:
                self.centroids[i] = np.random.uniform(-5, 5, size=self.dim) #reinitialize if species is empty

    def mutation_rand1(self, population, i):
        indices = np.random.choice(self.pop_size, 3, replace=False)
        x_r1, x_r2, x_r3 = population[indices]
        return population[i] + self.F * (x_r1 - x_r2) + self.F * (x_r3 - population[i])

    def mutation_current_to_best(self, population, i, best_x):
        indices = np.random.choice(self.pop_size, 2, replace=False)
        x_r1, x_r2 = population[indices]
        return population[i] + self.F * (best_x - population[i]) + self.F * (x_r1 - x_r2)

    def mutation_best2(self, population, i, best_x):
        indices = np.random.choice(self.pop_size, 2, replace=False)
        x_r1, x_r2 = population[indices]
        return best_x + self.F * (population[i] - x_r1) + self.F * (x_r2 - population[i])

    def local_search(self, x, func):
        # Simple random local search around x
        x_new = x + np.random.normal(0, 0.1, size=self.dim)
        x_new = np.clip(x_new, func.bounds.lb, func.bounds.ub)
        f_new = func(x_new)
        self.budget -= 1
        if f_new < func(x):
            return x_new, f_new
        else:
            return x, func(x)

    def __call__(self, func):
        # Initialization
        population = np.random.uniform(func.bounds.lb, func.bounds.ub, size=(self.pop_size, self.dim))
        fitness = np.array([func(x) for x in population])
        self.budget -= self.pop_size
        best_index = np.argmin(fitness)
        self.f_opt = fitness[best_index]
        self.x_opt = population[best_index]
        best_x = self.x_opt.copy()

        # Initialize centroids randomly
        self.centroids = np.random.uniform(func.bounds.lb, func.bounds.ub, size=(self.num_species, self.dim))
        self.initialize_species(population)

        # Evolution loop
        while self.budget > 0:
            # Clear species
            self.species = [[] for _ in range(self.num_species)]
            
            for i in range(self.pop_size):
                # Strategy selection
                strategy_index = np.random.choice(len(self.mutation_strategies), p=self.mutation_probs)
                self.strategy_usage[strategy_index] += 1
                mutation_strategy = self.mutation_strategies[strategy_index]

                # Mutation
                if mutation_strategy in [self.mutation_current_to_best, self.mutation_best2]:
                    v = mutation_strategy(population, i, best_x)
                else:
                    v = mutation_strategy(population, i)
                v = np.clip(v, func.bounds.lb, func.bounds.ub)

                # Crossover
                j_rand = np.random.randint(self.dim)
                u = np.copy(population[i])
                for j in range(self.dim):
                    if np.random.rand() < self.CR or j == j_rand:
                        u[j] = v[j]

                # Evaluation
                f_u = func(u)
                self.budget -= 1

                if f_u < fitness[i]:
                    # Replacement
                    fitness[i] = f_u
                    population[i] = u

                    # Local Search
                    if np.random.rand() < self.local_search_prob and self.budget > 0:
                        population[i], fitness[i] = self.local_search(population[i], func)
                        if fitness[i] < f_u:
                            f_u = fitness[i]
                    

                    # Update best solution
                    if f_u < self.f_opt:
                        self.f_opt = f_u
                        self.x_opt = u.copy()
                        best_x = self.x_opt.copy()
                    self.success_counts[strategy_index] += 1

            # Update mutation probabilities
            total_usage = np.sum(self.strategy_usage)
            if total_usage > 0:
                success_rates = self.success_counts / self.strategy_usage
                for k in range(len(self.mutation_strategies)):
                    self.mutation_probs[k] += self.learning_rate * (success_rates[k] - self.mutation_probs[k])
                self.mutation_probs = np.maximum(self.mutation_probs, 0.01)
                self.mutation_probs /= np.sum(self.mutation_probs)
            self.success_counts[:] = 0
            self.strategy_usage[:] = 0

            self.update_centroids(population)
            self.initialize_species(population)

        return self.f_opt, self.x_opt