import numpy as np

class DiversityAdaptiveDE:
    def __init__(self, budget=10000, dim=10, popsize=None, F_base=0.5, CR_base=0.7, F_range=0.2, CR_range=0.2, diversity_threshold=0.1):
        self.budget = budget
        self.dim = dim
        self.popsize = popsize if popsize is not None else 10 * self.dim
        self.F_base = F_base
        self.CR_base = CR_base
        self.F_range = F_range
        self.CR_range = CR_range
        self.diversity_threshold = diversity_threshold

    def __call__(self, func):
        lb = func.bounds.lb
        ub = func.bounds.ub
        self.population = np.random.uniform(lb, ub, size=(self.popsize, self.dim))
        self.fitness = np.array([func(x) for x in self.population])
        self.eval_count = self.popsize
        self.f_opt = np.min(self.fitness)
        self.x_opt = self.population[np.argmin(self.fitness)]

        while self.eval_count < self.budget:
            for i in range(self.popsize):
                # Parameter adaptation
                F = self.F_base + self.F_range * np.random.normal(0, 1)
                CR = self.CR_base + self.CR_range * np.random.normal(0, 1)
                F = np.clip(F, 0.1, 1.0)
                CR = np.clip(CR, 0.1, 1.0)

                # Mutation: rand/1
                donor_indices = np.random.choice(self.popsize, 3, replace=False)
                mutant = self.population[donor_indices[0]] + F * (self.population[donor_indices[1]] - self.population[donor_indices[2]])
                mutant = np.clip(mutant, lb, ub)

                # Crossover
                crossover_mask = np.random.rand(self.dim) < CR
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

            # Diversity maintenance:
            distances = self.calculate_distances()
            min_distance = np.min(distances)
            if min_distance < self.diversity_threshold:
                # Introduce new random solutions to increase diversity
                num_new = int(self.popsize * 0.1)
                indices_to_replace = np.random.choice(self.popsize, num_new, replace=False)
                self.population[indices_to_replace] = np.random.uniform(lb, ub, size=(num_new, self.dim))
                self.fitness[indices_to_replace] = np.array([func(x) for x in self.population[indices_to_replace]])
                self.eval_count += num_new
                
                #Update best solution, as fitness values have changed.
                self.f_opt = np.min(self.fitness)
                self.x_opt = self.population[np.argmin(self.fitness)]

            if self.eval_count > self.budget:
                break
                
        return self.f_opt, self.x_opt

    def calculate_distances(self):
        distances = []
        for i in range(self.popsize):
            for j in range(i + 1, self.popsize):
                distance = np.linalg.norm(self.population[i] - self.population[j])
                distances.append(distance)
        return np.array(distances)