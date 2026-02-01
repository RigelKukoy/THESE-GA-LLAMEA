import numpy as np

class BanditAdaptiveDE:
    def __init__(self, budget=10000, dim=10, pop_size=50, archive_size=10, F_values=[0.1, 0.5, 0.9], CR_values=[0.1, 0.5, 0.9]):
        self.budget = budget
        self.dim = dim
        self.pop_size = pop_size
        self.archive_size = archive_size
        self.F_values = F_values
        self.CR_values = CR_values
        self.F_rewards = {F: 0.0 for F in F_values}
        self.CR_rewards = {CR: 0.0 for CR in CR_values}
        self.F_counts = {F: 0 for F in F_values}
        self.CR_counts = {CR: 0 for CR in CR_values}
        self.archive = []
        self.p = 0.1 
        self.epsilon = 0.1 # Exploration rate for the bandit

    def select_parameter(self, rewards, counts, values):
        if np.random.rand() < self.epsilon:
            return np.random.choice(values)
        else:
            avg_rewards = {v: rewards[v] / (counts[v] + 1e-6) for v in values}
            return max(avg_rewards, key=avg_rewards.get)
            
    def __call__(self, func):
        self.f_opt = np.Inf
        self.x_opt = None
        self.population = np.random.uniform(func.bounds.lb, func.bounds.ub, size=(self.pop_size, self.dim))
        fitness = np.array([func(x) for x in self.population])
        self.budget -= self.pop_size

        for i in range(self.pop_size):
            if fitness[i] < self.f_opt:
                self.f_opt = fitness[i]
                self.x_opt = self.population[i]

        while self.budget > 0:
            for i in range(self.pop_size):
                # Bandit-based parameter selection
                F = self.select_parameter(self.F_rewards, self.F_counts, self.F_values)
                CR = self.select_parameter(self.CR_rewards, self.CR_counts, self.CR_values)
                

                # Mutation
                idxs = np.random.choice(self.pop_size, 3, replace=False)
                x_1, x_2, x_3 = self.population[idxs]
                mutant = x_1 + F * (x_2 - x_3)
                mutant = np.clip(mutant, func.bounds.lb, func.bounds.ub)

                # Crossover
                crossover = np.random.uniform(size=self.dim) < CR
                trial = np.where(crossover, mutant, self.population[i])

                # Selection
                f_trial = func(trial)
                self.budget -= 1

                if f_trial < self.f_opt:
                    self.f_opt = f_trial
                    self.x_opt = trial
                    
                if (fitness[i] < 0 and f_trial < 0) or np.random.rand() < self.p:
                    if f_trial < fitness[i]:
                        reward = fitness[i] - f_trial
                        
                        # Update Bandit Rewards
                        self.F_rewards[F] += reward
                        self.CR_rewards[CR] += reward
                        
                        # Update Bandit Counts
                        self.F_counts[F] += 1
                        self.CR_counts[CR] += 1

                        fitness[i] = f_trial
                        self.population[i] = trial
                        
                        if len(self.archive) < self.archive_size:
                            self.archive.append(self.population[i].copy())
                        else:
                            idx_to_replace = np.random.randint(0, self.archive_size)
                            self.archive[idx_to_replace] = self.population[i].copy()

                else:
                    if f_trial < fitness[i]:
                        fitness[i] = f_trial
                        self.population[i] = trial

                        if len(self.archive) < self.archive_size:
                            self.archive.append(self.population[i].copy())
                        else:
                            idx_to_replace = np.random.randint(0, self.archive_size)
                            self.archive[idx_to_replace] = self.population[i].copy()

        return self.f_opt, self.x_opt