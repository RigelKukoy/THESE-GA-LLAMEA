import numpy as np

class CrowdingDE:
    def __init__(self, budget=10000, dim=10, pop_size=50, F=0.5, CR=0.7, F_lr=0.1, CR_lr=0.1, current_to_best_prob=0.5):
        self.budget = budget
        self.dim = dim
        self.pop_size = pop_size
        self.F = F
        self.CR = CR
        self.F_lr = F_lr
        self.CR_lr = CR_lr
        self.current_to_best_prob = current_to_best_prob
        self.population = None
        self.fitness = None
        self.best_index = None
        self.f_opt = np.inf
        self.x_opt = None

    def initialize_population(self, func):
        self.population = np.random.uniform(func.bounds.lb, func.bounds.ub, size=(self.pop_size, self.dim))
        self.fitness = np.array([func(x) for x in self.population])
        self.budget -= self.pop_size
        self.best_index = np.argmin(self.fitness)
        self.f_opt = self.fitness[self.best_index]
        self.x_opt = self.population[self.best_index].copy()

    def current_to_best_mutation(self, x, x_best, x_r1, F):
        return x + F * (x_best - x) + F * (x_r1 - x)

    def cauchy_mutation(self, x_r1, x_r2, F):
        return x_r1 + F * np.random.standard_cauchy(size=x_r1.shape) * (x_r1 - x_r2)

    def calculate_crowding_distance(self):
        distances = np.zeros(self.pop_size)
        for m in range(self.dim):
            # Sort population based on the m-th dimension
            sorted_indices = np.argsort(self.population[:, m])
            
            # Boundary individuals get maximum distance
            distances[sorted_indices[0]] = np.inf
            distances[sorted_indices[-1]] = np.inf
            
            # Calculate distance for intermediate individuals
            for i in range(1, self.pop_size - 1):
                distances[sorted_indices[i]] += (self.population[sorted_indices[i+1], m] - self.population[sorted_indices[i-1], m])

        return distances

    def __call__(self, func):
        self.initialize_population(func)

        while self.budget > 0:
            crowding_distances = self.calculate_crowding_distance()

            for i in range(self.pop_size):
                # Mutation
                indices = np.random.choice(self.pop_size, 2, replace=False)
                x_r1, x_r2 = self.population[indices]

                if np.random.rand() < self.current_to_best_prob:
                    # Current-to-best mutation
                    v = self.current_to_best_mutation(self.population[i], self.population[self.best_index], x_r1, self.F)
                else:
                    # Cauchy mutation
                    v = self.cauchy_mutation(x_r1, x_r2, self.F)

                v = np.clip(v, func.bounds.lb, func.bounds.ub)

                # Crossover
                j_rand = np.random.randint(self.dim)
                u = self.population[i].copy()
                for j in range(self.dim):
                    if np.random.rand() < self.CR or j == j_rand:
                        u[j] = v[j]

                # Evaluation
                f_u = func(u)
                self.budget -= 1

                # Selection based on crowding distance
                if f_u < self.fitness[i] or (f_u == self.fitness[i] and crowding_distances[i] < crowding_distances[i]):
                    self.fitness[i] = f_u
                    self.population[i] = u

                    if f_u < self.f_opt:
                        self.f_opt = f_u
                        self.x_opt = u.copy()

                        #Update F and CR learning rate using a simple rule
                        if f_u < self.fitness[i]:
                            self.F = max(0.1, min(0.9, self.F + self.F_lr * (np.random.rand() - 0.5)))
                            self.CR = max(0.1, min(0.9, self.CR + self.CR_lr * (np.random.rand() - 0.5)))
                

        return self.f_opt, self.x_opt