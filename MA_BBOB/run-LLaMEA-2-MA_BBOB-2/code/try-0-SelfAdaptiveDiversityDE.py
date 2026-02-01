import numpy as np

class SelfAdaptiveDiversityDE:
    def __init__(self, budget=10000, dim=10, popsize=None, F_mu=0.5, F_sigma=0.1, CR_mu=0.7, CR_sigma=0.1, restart_threshold=5000, crowding_distance_epsilon=1e-6):
        self.budget = budget
        self.dim = dim
        self.popsize = popsize if popsize is not None else 10 * self.dim
        self.F_mu = F_mu
        self.F_sigma = F_sigma
        self.CR_mu = CR_mu
        self.CR_sigma = CR_sigma
        self.restart_threshold = restart_threshold
        self.crowding_distance_epsilon = crowding_distance_epsilon
        self.eval_count = 0
        self.best_fitness_history = []

    def __call__(self, func):
        lb = func.bounds.lb
        ub = func.bounds.ub
        self.population = np.random.uniform(lb, ub, size=(self.popsize, self.dim))
        self.fitness = np.array([func(x) for x in self.population])
        self.eval_count += self.popsize
        self.f_opt = np.min(self.fitness)
        self.x_opt = self.population[np.argmin(self.fitness)]
        self.best_fitness_history.append(self.f_opt)

        self.F = np.random.normal(self.F_mu, self.F_sigma, self.popsize)
        self.F = np.clip(self.F, 0.0, 1.0)
        self.CR = np.random.normal(self.CR_mu, self.CR_sigma, self.popsize)
        self.CR = np.clip(self.CR, 0.0, 1.0)
        
        while self.eval_count < self.budget:
            for i in range(self.popsize):
                # Mutation
                donor_indices = np.random.choice(self.popsize, 3, replace=False)
                mutant = self.population[donor_indices[0]] + self.F[i] * (self.population[donor_indices[1]] - self.population[donor_indices[2]])
                mutant = np.clip(mutant, lb, ub)

                # Crossover
                crossover_mask = np.random.rand(self.dim) < self.CR[i]
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

                    # Update F and CR
                    self.F[i] = 0.9 * self.F[i] + 0.1 * np.random.normal(self.F_mu, self.F_sigma)
                    self.F[i] = np.clip(self.F[i], 0.0, 1.0)
                    self.CR[i] = 0.9 * self.CR[i] + 0.1 * np.random.normal(self.CR_mu, self.CR_sigma)
                    self.CR[i] = np.clip(self.CR[i], 0.0, 1.0)
                else:
                     #Update F and CR (opposite direction)
                    self.F[i] = 0.9 * self.F[i] - 0.1 * np.random.normal(self.F_mu, self.F_sigma)
                    self.F[i] = np.clip(self.F[i], 0.0, 1.0)
                    self.CR[i] = 0.9 * self.CR[i] - 0.1 * np.random.normal(self.CR_mu, self.CR_sigma)
                    self.CR[i] = np.clip(self.CR[i], 0.0, 1.0)


            # Diversity Maintenance (Crowding Distance)
            distances = np.zeros(self.popsize)
            for k in range(self.dim):
                sorted_indices = np.argsort(self.population[:, k])
                distances[sorted_indices[0]] = np.inf
                distances[sorted_indices[-1]] = np.inf
                for j in range(1, self.popsize - 1):
                    distances[sorted_indices[j]] += (self.population[sorted_indices[j+1], k] - self.population[sorted_indices[j-1], k]) / (ub - lb + self.crowding_distance_epsilon)

            min_dist = np.min(distances)
            if min_dist < 1e-6:
                worst_index = np.argmax(distances)
                self.population[worst_index] = np.random.uniform(lb, ub, self.dim)
                self.fitness[worst_index] = func(self.population[worst_index])
                self.eval_count += 1
                if self.fitness[worst_index] < self.f_opt:
                    self.f_opt = self.fitness[worst_index]
                    self.x_opt = self.population[worst_index]

            self.best_fitness_history.append(self.f_opt)

            # Restart Strategy
            if self.eval_count > self.restart_threshold and self.f_opt == self.best_fitness_history[-self.restart_threshold//10]:
                self.population = np.random.uniform(lb, ub, size=(self.popsize, self.dim))
                self.fitness = np.array([func(x) for x in self.population])
                self.eval_count += self.popsize
                if np.min(self.fitness) < self.f_opt:
                  self.f_opt = np.min(self.fitness)
                  self.x_opt = self.population[np.argmin(self.fitness)]
                self.F = np.random.normal(self.F_mu, self.F_sigma, self.popsize)
                self.F = np.clip(self.F, 0.0, 1.0)
                self.CR = np.random.normal(self.CR_mu, self.CR_sigma, self.popsize)
                self.CR = np.clip(self.CR, 0.0, 1.0)
                

            if self.eval_count > self.budget:
                break

        return self.f_opt, self.x_opt