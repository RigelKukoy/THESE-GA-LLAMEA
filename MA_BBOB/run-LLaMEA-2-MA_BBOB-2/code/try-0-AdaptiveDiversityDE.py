import numpy as np

class AdaptiveDiversityDE:
    def __init__(self, budget=10000, dim=10, initial_popsize=None, F_initial=0.5, CR_initial=0.7, diversity_threshold=0.1):
        self.budget = budget
        self.dim = dim
        self.initial_popsize = initial_popsize if initial_popsize is not None else 10 * self.dim
        self.popsize = self.initial_popsize
        self.F = F_initial
        self.CR = CR_initial
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
            # Calculate population diversity
            diversity = self.calculate_diversity()

            for i in range(self.popsize):
                # Adaptive Mutation Strategy
                if diversity > self.diversity_threshold:
                    # Global Exploration: Use more diverse mutation
                    donor_indices = np.random.choice(self.popsize, 3, replace=False)
                    mutant = self.population[donor_indices[0]] + self.F * (self.population[donor_indices[1]] - self.population[donor_indices[2]])
                else:
                    # Local Exploitation: Perturb the current best solution
                    mutant = self.x_opt + self.F * (np.random.uniform(lb, ub, size=self.dim) - self.population[i])
                    

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
            
            if self.eval_count > self.budget:
                break
        return self.f_opt, self.x_opt

    def calculate_diversity(self):
        # Calculate the average distance between individuals in the population
        distances = []
        for i in range(self.popsize):
            for j in range(i + 1, self.popsize):
                distances.append(np.linalg.norm(self.population[i] - self.population[j]))
        
        if distances:
            return np.mean(distances) / (np.max(self.population) - np.min(self.population)) # Normalize by range
        else:
            return 0.0