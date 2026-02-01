import numpy as np

class SOMAdaptiveDE:
    def __init__(self, budget=10000, dim=10, popsize=None, CR=0.7, som_grid_size=5, learning_rate=0.1, sigma=1.0):
        self.budget = budget
        self.dim = dim
        self.popsize = popsize if popsize is not None else 10 * self.dim
        self.CR = CR
        self.som_grid_size = som_grid_size
        self.learning_rate = learning_rate
        self.sigma = sigma
        self.som = np.random.rand(som_grid_size, som_grid_size, dim)
        self.mutation_strategies = [
            self._mutation_DE_rand1,
            self._mutation_DE_best1,
            self._mutation_DE_current_to_rand1,
            self._mutation_DE_current_to_best1
        ]

    def __call__(self, func):
        lb = func.bounds.lb
        ub = func.bounds.ub
        self.population = np.random.uniform(lb, ub, size=(self.popsize, self.dim))
        self.fitness = np.array([func(x) for x in self.population])
        self.eval_count = self.popsize
        self.f_opt = np.min(self.fitness)
        self.x_opt = self.population[np.argmin(self.fitness)]

        while self.eval_count < self.budget:
            # SOM training
            self._train_som(self.population)

            # Assign individuals to SOM nodes
            node_assignments = self._assign_to_nodes(self.population)

            for i in range(self.popsize):
                # Select mutation strategy based on SOM node assignment
                node_row, node_col = node_assignments[i]
                mutation_index = (node_row * self.som_grid_size + node_col) % len(self.mutation_strategies)  # Ensure valid index
                mutation_function = self.mutation_strategies[mutation_index]

                # Mutation
                mutant = mutation_function(i)
                mutant = np.clip(mutant, lb, ub)

                # Crossover
                crossover_mask = np.random.rand(self.dim) < self.CR
                trial = np.where(crossover_mask, mutant, self.population[i])

                # Selection
                f_trial = func(trial)
                self.eval_count += 1

                if f_trial < self.fitness[i]:
                    self.population[i] = trial
                    self.fitness[i] = f_trial
                    if self.fitness[i] < self.f_opt:
                        self.f_opt = self.fitness[i]
                        self.x_opt = self.population[i]

        return self.f_opt, self.x_opt

    def _train_som(self, data):
        for x in data:
            best_node = self._find_best_matching_unit(x)
            self._update_som_nodes(x, best_node)

    def _find_best_matching_unit(self, x):
        distances = np.sum((self.som - x)**2, axis=2)
        return np.unravel_index(np.argmin(distances), distances.shape)

    def _update_som_nodes(self, x, best_node):
        for i in range(self.som_grid_size):
            for j in range(self.som_grid_size):
                distance = np.sqrt((i - best_node[0])**2 + (j - best_node[1])**2)
                influence = np.exp(-distance**2 / (2 * self.sigma**2))
                self.som[i, j] += self.learning_rate * influence * (x - self.som[i, j])

    def _assign_to_nodes(self, data):
        assignments = []
        for x in data:
            assignments.append(self._find_best_matching_unit(x))
        return assignments

    def _mutation_DE_rand1(self, i):
        idxs = np.random.choice(self.popsize, 3, replace=False)
        x1, x2, x3 = self.population[idxs]
        return self.population[i] + 0.5 * (x2 - x3)

    def _mutation_DE_best1(self, i):
        idxs = np.random.choice(self.popsize, 2, replace=False)
        x1, x2 = self.population[idxs]
        return self.population[i] + 0.5 * (self.x_opt - self.population[i]) + 0.5 * (x1 - x2)

    def _mutation_DE_current_to_rand1(self, i):
         idxs = np.random.choice(self.popsize, 3, replace=False)
         x1, x2, x3 = self.population[idxs]
         return self.population[i] + 0.5*(x1 - self.population[i]) + 0.5 * (x2 - x3)

    def _mutation_DE_current_to_best1(self, i):
        idxs = np.random.choice(self.popsize, 2, replace=False)
        x1, x2 = self.population[idxs]
        return self.population[i] + 0.5 * (self.x_opt - self.population[i]) + 0.5 * (x1 - x2)