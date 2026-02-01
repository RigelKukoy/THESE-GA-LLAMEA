import numpy as np

class OrthogonalDifferentialEvolution:
    def __init__(self, budget=10000, dim=10, pop_size=20, orthogonal_components=5):
        self.budget = budget
        self.dim = dim
        self.pop_size = pop_size
        self.orthogonal_components = orthogonal_components
        self.F = 0.7  # Differential evolution scaling factor
        self.CR = 0.7  # Crossover rate

    def __call__(self, func):
        self.f_opt = np.Inf
        self.x_opt = None

        # Initialize population
        self.population = np.random.uniform(func.bounds.lb, func.bounds.ub, size=(self.pop_size, self.dim))
        self.fitness = np.array([func(x) for x in self.population])
        self.budget -= self.pop_size

        while self.budget > 0:
            for i in range(self.pop_size):
                # Differential Evolution Mutation
                idxs = [idx for idx in range(self.pop_size) if idx != i]
                a, b, c = np.random.choice(idxs, 3, replace=False)
                mutant = self.population[a] + self.F * (self.population[b] - self.population[c])
                mutant = np.clip(mutant, func.bounds.lb, func.bounds.ub)

                # Crossover
                trial = np.copy(self.population[i])
                mask = np.random.rand(self.dim) < self.CR
                trial[mask] = mutant[mask]

                # Orthogonal Learning: Sample around the trial vector
                orthogonal_samples = np.random.normal(trial, 0.05, size=(self.orthogonal_components, self.dim)) # Reduced std
                orthogonal_samples = np.clip(orthogonal_samples, func.bounds.lb, func.bounds.ub)
                orthogonal_fitness = np.array([func(x) for x in orthogonal_samples])
                self.budget -= self.orthogonal_components

                best_orthogonal_index = np.argmin(orthogonal_fitness)
                if orthogonal_fitness[best_orthogonal_index] < func(trial):
                    trial = orthogonal_samples[best_orthogonal_index]

                # Selection
                f_trial = func(trial)
                self.budget -= 1  # Increment the budget counter after the evaluation

                if f_trial < self.fitness[i]:
                    self.population[i] = trial
                    self.fitness[i] = f_trial

                    if f_trial < self.f_opt:
                        self.f_opt = f_trial
                        self.x_opt = self.population[i]

                if self.budget <= 0:
                    break

        return self.f_opt, self.x_opt