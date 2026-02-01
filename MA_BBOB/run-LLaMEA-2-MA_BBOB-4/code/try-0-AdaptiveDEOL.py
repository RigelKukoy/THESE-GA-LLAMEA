import numpy as np

class AdaptiveDEOL:
    def __init__(self, budget=10000, dim=10, pop_size=50, F=0.5, Cr=0.9, archive_size=10, restart_trigger=0.01):
        self.budget = budget
        self.dim = dim
        self.pop_size = pop_size
        self.F = F  # Mutation factor
        self.Cr = Cr  # Crossover rate
        self.archive_size = archive_size
        self.archive = []
        self.restart_trigger = restart_trigger
        self.min_F = 0.1
        self.max_F = 0.9
        self.min_Cr = 0.1
        self.max_Cr = 0.9

    def __call__(self, func):
        self.f_opt = np.Inf
        self.x_opt = None
        self.population = np.random.uniform(func.bounds.lb, func.bounds.ub, size=(self.pop_size, self.dim))
        self.fitness = np.array([func(x) for x in self.population])
        self.eval_count = self.pop_size

        while self.eval_count < self.budget:
            for i in range(self.pop_size):
                # Parameter adaptation
                self.F = np.clip(np.random.normal(self.F, 0.1), self.min_F, self.max_F)
                self.Cr = np.clip(np.random.normal(self.Cr, 0.1), self.min_Cr, self.max_Cr)
                
                # Mutation
                if len(self.archive) > 0 and np.random.rand() < 0.1:
                    # Use archive with a probability of 0.1
                    idx = np.random.choice(len(self.archive), 2, replace=False)
                    x_r1 = self.archive[idx[0]]
                    x_r2 = self.archive[idx[1]]

                else:
                    idx = np.random.choice(self.pop_size, 3, replace=False)
                    while i in idx:
                        idx = np.random.choice(self.pop_size, 3, replace=False)
                    x_r1 = self.population[idx[0]]
                    x_r2 = self.population[idx[1]]
                    x_r3 = self.population[idx[2]]

                x_mutated = self.population[i] + self.F * (x_r1 - x_r2)

                # Orthogonal Learning
                levels = 3
                ol_vector = np.zeros(self.dim)
                for j in range(self.dim):
                    level_idx = np.random.randint(levels)
                    ol_vector[j] = x_r1[j] + (level_idx - 1) * (x_r2[j] - x_r3[j])

                # Crossover
                x_trial = np.copy(self.population[i])
                j_rand = np.random.randint(self.dim)
                for j in range(self.dim):
                    if np.random.rand() < self.Cr or j == j_rand:
                        x_trial[j] = x_mutated[j]
                    else:
                        x_trial[j] = ol_vector[j] # Incorporate orthogonal learning

                x_trial = np.clip(x_trial, func.bounds.lb, func.bounds.ub)
                
                # Selection
                f_trial = func(x_trial)
                self.eval_count += 1

                if f_trial < self.fitness[i]:
                    # Update population and archive
                    self.fitness[i] = f_trial
                    self.population[i] = x_trial

                    # Update archive
                    if len(self.archive) < self.archive_size:
                        self.archive.append(self.population[i])
                    else:
                        # Replace a random element in the archive
                        idx_replace = np.random.randint(self.archive_size)
                        self.archive[idx_replace] = self.population[i]

                    # Update best solution
                    if f_trial < self.f_opt:
                        self.f_opt = f_trial
                        self.x_opt = x_trial

                else:
                    # Add parent to archive if trial is worse
                     if len(self.archive) < self.archive_size:
                        self.archive.append(self.population[i])
                     else:
                        # Replace a random element in the archive
                        idx_replace = np.random.randint(self.archive_size)
                        self.archive[idx_replace] = self.population[i]

            # Restart mechanism: Checks for stagnation and restarts the population if necessary
            if self.eval_count > self.budget * 0.1 and np.std(self.fitness) < self.restart_trigger: # Check for stagnation after 10% budget is used
                self.population = np.random.uniform(func.bounds.lb, func.bounds.ub, size=(self.pop_size, self.dim))
                self.fitness = np.array([func(x) for x in self.population])
                self.archive = []
                self.eval_count += self.pop_size
                
        return self.f_opt, self.x_opt