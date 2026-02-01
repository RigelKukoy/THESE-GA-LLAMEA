import numpy as np

class DistanceAdaptiveDENiche:
    def __init__(self, budget=10000, dim=10, popsize=None, F_initial=0.5, CR_initial=0.7, F_adapt_rate=0.1, CR_adapt_rate=0.1, niche_radius=0.5):
        self.budget = budget
        self.dim = dim
        self.popsize = popsize if popsize is not None else 10 * self.dim
        self.F = F_initial
        self.CR = CR_initial
        self.F_adapt_rate = F_adapt_rate
        self.CR_adapt_rate = CR_adapt_rate
        self.niche_radius = niche_radius  # Radius for niching
        self.population = None
        self.fitness = None
        self.eval_count = 0
        self.f_opt = np.Inf
        self.x_opt = None

    def calculate_distances(self):
        """Calculates pairwise Euclidean distances between all individuals in the population."""
        distances = np.zeros((self.popsize, self.popsize))
        for i in range(self.popsize):
            for j in range(i + 1, self.popsize):
                distances[i, j] = np.linalg.norm(self.population[i] - self.population[j])
                distances[j, i] = distances[i, j]
        return distances

    def distance_based_mutation(self, i, distances, lb, ub):
        """Performs mutation favoring individuals further away."""
        farthest_idx = np.argmax(distances[i])  # Find the individual farthest from i
        idxs = np.random.choice(self.popsize, 2, replace=False)  # Select two random individuals
        x1, x2 = self.population[idxs]

        mutant = self.population[i] + self.F * (self.population[farthest_idx] - x1) + self.F * (x2 - self.population[i])
        mutant = np.clip(mutant, lb, ub)
        return mutant

    def niching(self):
        """Applies niching to maintain diversity. Penalizes fitness within niches."""
        distances = self.calculate_distances()
        for i in range(self.popsize):
            for j in range(i + 1, self.popsize):
                if distances[i, j] < self.niche_radius:
                    # Penalize fitness of individuals within the same niche
                    penalty = 0.1 * (self.niche_radius - distances[i, j])  # Example penalty
                    self.fitness[i] += penalty
                    self.fitness[j] += penalty


    def __call__(self, func):
        lb = func.bounds.lb
        ub = func.bounds.ub

        # Initialize population
        self.population = np.random.uniform(lb, ub, size=(self.popsize, self.dim))
        self.fitness = np.array([func(x) for x in self.population])
        self.eval_count = self.popsize
        self.f_opt = np.min(self.fitness)
        self.x_opt = self.population[np.argmin(self.fitness)]


        while self.eval_count < self.budget:
            distances = self.calculate_distances()
            for i in range(self.popsize):
                # Mutation
                mutant = self.distance_based_mutation(i, distances, lb, ub)

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

            # Adapt F and CR based on population fitness variance
            fitness_std = np.std(self.fitness)
            self.F = np.clip(self.F + self.F_adapt_rate * (fitness_std - 0.1), 0.1, 0.9)
            self.CR = np.clip(self.CR + self.CR_adapt_rate * (fitness_std - 0.1), 0.1, 0.9)


            self.niching() # Apply niching after each generation

        return self.f_opt, self.x_opt