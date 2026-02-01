import numpy as np

class AdaptiveDE_OL:
    def __init__(self, budget=10000, dim=10, pop_multiplier=5, orthogonal_trials=5):
        self.budget = budget
        self.dim = dim
        self.pop_size = dim * pop_multiplier
        self.F = 0.5  # Initial mutation factor
        self.CR = 0.9 # Crossover rate
        self.orthogonal_trials = orthogonal_trials # Number of orthogonal design trials

    def __call__(self, func):
        self.f_opt = np.Inf
        self.x_opt = None
        self.population = np.random.uniform(func.bounds.lb, func.bounds.ub, size=(self.pop_size, self.dim))
        self.fitness = np.array([func(x) for x in self.population])
        self.budget -= self.pop_size

        while self.budget > 0:
            for i in range(self.pop_size):
                # Mutation
                idxs = [idx for idx in range(self.pop_size) if idx != i]
                a, b, c = self.population[np.random.choice(idxs, 3, replace=False)]
                mutant = np.clip(a + self.F * (b - c), func.bounds.lb, func.bounds.ub)

                # Crossover
                crossover = np.random.rand(self.dim) < self.CR
                trial = np.where(crossover, mutant, self.population[i])

                # Selection
                f_trial = func(trial)
                self.budget -= 1
                if f_trial < self.fitness[i]:
                    self.fitness[i] = f_trial
                    self.population[i] = trial

                    if f_trial < self.f_opt:
                        self.f_opt = f_trial
                        self.x_opt = trial
                
                # Orthogonal Learning
                else:
                    orthogonal_matrix = self.generate_orthogonal_array(self.dim, self.orthogonal_trials)
                    best_orthogonal_f = np.inf
                    best_orthogonal_x = None

                    for j in range(self.orthogonal_trials):
                        orthogonal_x = self.population[i].copy()
                        for k in range(self.dim):
                            if orthogonal_matrix[j,k] == 1:
                                orthogonal_x[k] = np.clip(self.population[i][k] + 0.1 * (np.random.rand() - 0.5), func.bounds.lb, func.bounds.ub)
                            else:
                                orthogonal_x[k] = np.clip(self.population[i][k] - 0.1 * (np.random.rand() - 0.5), func.bounds.lb, func.bounds.ub)

                        orthogonal_f = func(orthogonal_x)
                        self.budget -= 1

                        if orthogonal_f < best_orthogonal_f:
                            best_orthogonal_f = orthogonal_f
                            best_orthogonal_x = orthogonal_x

                        if self.budget <= 0:
                            break

                    if best_orthogonal_f < self.fitness[i]:
                         self.fitness[i] = best_orthogonal_f
                         self.population[i] = best_orthogonal_x

                         if best_orthogonal_f < self.f_opt:
                            self.f_opt = best_orthogonal_f
                            self.x_opt = best_orthogonal_x

                if self.budget > 0 and np.random.rand() < 0.1:  # Occasionally perturb F and CR
                    self.F = np.clip(np.random.normal(0.5, 0.3), 0.1, 1.0) # Adapt F
                    self.CR = np.clip(np.random.normal(0.9, 0.1), 0.1, 1.0) # Adapt CR

                if self.budget <=0:
                    break

        return self.f_opt, self.x_opt

    def generate_orthogonal_array(self, n, k):
          # generates a L_k(2^n) orthogonal array.
          # In this implementation, we use a very simple (non-optimized) version, sufficient for small n and k
          array = np.random.randint(0, 2, size=(k, n)) # Simple random array for demonstration purposes
          return array