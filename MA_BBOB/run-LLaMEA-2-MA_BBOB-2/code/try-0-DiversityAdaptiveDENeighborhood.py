import numpy as np

class DiversityAdaptiveDENeighborhood:
    def __init__(self, budget=10000, dim=10, popsize=None, CR_init=0.5, F=0.5, neighborhood_size=5, CR_adapt_speed=0.1, step_size_init=0.1, step_size_min=0.001):
        self.budget = budget
        self.dim = dim
        self.popsize = popsize if popsize is not None else 10 * self.dim
        self.CR_init = CR_init
        self.F = F
        self.neighborhood_size = neighborhood_size
        self.CR_adapt_speed = CR_adapt_speed
        self.step_size_init = step_size_init
        self.step_size_min = step_size_min
        self.step_size = self.step_size_init

    def __call__(self, func):
        lb = func.bounds.lb
        ub = func.bounds.ub
        self.population = np.random.uniform(lb, ub, size=(self.popsize, self.dim))
        self.fitness = np.array([func(x) for x in self.population])
        self.eval_count = self.popsize
        self.f_opt = np.min(self.fitness)
        self.x_opt = self.population[np.argmin(self.fitness)]
        self.CR = np.full(self.popsize, self.CR_init)

        while self.eval_count < self.budget:
            # Calculate population diversity
            diversity = np.std(self.population)

            # Adjust crossover rate based on diversity
            self.CR = self.CR_init + self.CR_adapt_speed * diversity
            self.CR = np.clip(self.CR, 0.1, 0.9) # Ensure CR remains within reasonable bounds

            # Adjust step size
            self.step_size = max(self.step_size_min, self.step_size_init * (1 - self.eval_count / self.budget))

            for i in range(self.popsize):
                # Mutation: Neighborhood-based
                neighbors_indices = np.random.choice(self.popsize, self.neighborhood_size, replace=False)
                neighbors = self.population[neighbors_indices]
                center = np.mean(neighbors, axis=0)
                mutant = self.population[i] + self.step_size * (center - self.population[i]) # dynamically changing step size
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