import numpy as np

class MirroredSamplingAdaptiveDE:
    def __init__(self, budget=10000, dim=10, pop_size=50, F=0.5, CR=0.7, mirror_rate=0.2):
        self.budget = budget
        self.dim = dim
        self.pop_size = pop_size
        self.F = F  # Initial mutation factor
        self.CR = CR  # Initial crossover rate
        self.mirror_rate = mirror_rate # probability of mirroring a solution.
        self.mutation_strategies = [self.mutation_strategy_1, self.mutation_strategy_2, self.mutation_strategy_3]
        self.mutation_probs = np.ones(len(self.mutation_strategies)) / len(self.mutation_strategies)  # Initial probabilities for each strategy
        self.success_counts = np.zeros(len(self.mutation_strategies))
        self.strategy_usage = np.zeros(len(self.mutation_strategies))
        self.learning_rate = 0.1
        self.diversity_threshold = 0.01 # Threshold for diversity check
        self.diversity_weight = 0.1 # weight for adjusting F and CR based on diversity.


    def mutation_strategy_1(self, population, i):
        # DE/rand/1
        indices = np.random.choice(self.pop_size, 3, replace=False)
        x_r1, x_r2, x_r3 = population[indices]
        return population[i] + self.F * (x_r1 - population[i]) + self.F * (x_r2 - x_r3)

    def mutation_strategy_2(self, population, i):
        # DE/current-to-rand/1
        indices = np.random.choice(self.pop_size, 3, replace=False)
        x_r1, x_r2, x_r3 = population[indices]
        return population[i] + self.F * (x_r1 - population[i]) + self.F * (x_r2 - x_r3)

    def mutation_strategy_3(self, population, i, best_x):
         # DE/best/1
        indices = np.random.choice(self.pop_size, 2, replace=False)
        x_r1, x_r2 = population[indices]
        return best_x + self.F * (x_r1 - x_r2)

    def __call__(self, func):
        # Initialization
        lb = func.bounds.lb
        ub = func.bounds.ub
        population = np.random.uniform(lb, ub, size=(self.pop_size, self.dim))
        fitness = np.array([func(x) for x in population])
        self.budget -= self.pop_size

        best_index = np.argmin(fitness)
        self.f_opt = fitness[best_index]
        self.x_opt = population[best_index]
        best_x = self.x_opt.copy()

        # Evolution loop
        while self.budget > 0:
            # Calculate population diversity
            diversity = np.std(population)

            for i in range(self.pop_size):
                # Strategy selection
                strategy_index = np.random.choice(len(self.mutation_strategies), p=self.mutation_probs)
                self.strategy_usage[strategy_index] += 1
                mutation_strategy = self.mutation_strategies[strategy_index]
                
                # Mutation
                if mutation_strategy == self.mutation_strategy_3:
                    v = mutation_strategy(population, i, best_x)
                else:
                    v = mutation_strategy(population, i)
                v = np.clip(v, lb, ub)

                # Crossover
                j_rand = np.random.randint(self.dim)
                u = np.copy(population[i])
                for j in range(self.dim):
                    if np.random.rand() < self.CR or j == j_rand:
                        u[j] = v[j]

                # Mirrored sampling
                if np.random.rand() < self.mirror_rate:
                    mirror_point = np.random.uniform(lb, ub, self.dim)  # Generate a random "midpoint" for mirroring.
                    u = 2 * mirror_point - u # mirror u around mirror_point

                    u = np.clip(u, lb, ub) # Keep solution within bounds

                # Evaluation
                f_u = func(u)
                self.budget -= 1

                if f_u < fitness[i]:
                    # Replacement
                    fitness[i] = f_u
                    population[i] = u

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
                self.mutation_probs = np.maximum(self.mutation_probs, 0.01)  # Avoid zero probabilities
                self.mutation_probs /= np.sum(self.mutation_probs)

            self.success_counts[:] = 0
            self.strategy_usage[:] = 0

             # Dynamic parameter adaptation based on diversity
            if diversity < self.diversity_threshold:
                self.F *= (1 + self.diversity_weight)  # Increase F to enhance exploration
                self.CR *= (1 - self.diversity_weight)  # Decrease CR to focus exploitation
            else:
                self.F *= (1 - self.diversity_weight/2)  # Reduce F to exploit better
                self.CR *= (1 + self.diversity_weight/2)   # Increase CR for more exploration

            self.F = np.clip(self.F, 0.1, 1.0) # bound F
            self.CR = np.clip(self.CR, 0.1, 0.9) # bound CR


        return self.f_opt, self.x_opt