import numpy as np

class DynPopCMAESDE:
    def __init__(self, budget=10000, dim=10, popsize_init=None, F=0.5, CR=0.7, target_success_rate=0.25, popsize_reduction_factor=0.5, popsize_increase_factor=2.0):
        self.budget = budget
        self.dim = dim
        self.popsize = popsize_init if popsize_init is not None else 10 * self.dim
        self.popsize = int(self.popsize)
        self.F = F
        self.CR = CR
        self.target_success_rate = target_success_rate
        self.popsize_reduction_factor = popsize_reduction_factor
        self.popsize_increase_factor = popsize_increase_factor
        self.success_history = []
        self.success_history_length = 10

    def __call__(self, func):
        lb = func.bounds.lb
        ub = func.bounds.ub
        self.population = np.random.uniform(lb, ub, size=(self.popsize, self.dim))
        self.fitness = np.array([func(x) for x in self.population])
        self.eval_count = self.popsize
        self.f_opt = np.min(self.fitness)
        self.x_opt = self.population[np.argmin(self.fitness)]

        C = np.eye(self.dim)  # Covariance matrix
        learning_rate = 0.1

        while self.eval_count < self.budget:
            # Mutation and Crossover
            trial_population = np.zeros_like(self.population)
            trial_fitness = np.zeros_like(self.fitness)
            successful_count = 0

            for i in range(self.popsize):
                idxs = np.random.choice(self.popsize, 3, replace=False)
                x1, x2, x3 = self.population[idxs]

                # CMA-ES inspired mutation
                z = np.random.multivariate_normal(np.zeros(self.dim), C)
                mutant = x1 + self.F * z
                mutant = np.clip(mutant, lb, ub)
                
                # Crossover
                crossover_mask = np.random.rand(self.dim) < self.CR
                trial = np.where(crossover_mask, mutant, self.population[i])

                # Evaluation
                f_trial = func(trial)
                self.eval_count += 1
                trial_population[i] = trial
                trial_fitness[i] = f_trial

                # Selection
                if f_trial < self.fitness[i]:
                    self.population[i] = trial
                    self.fitness[i] = f_trial
                    if self.fitness[i] < self.f_opt:
                        self.f_opt = self.fitness[i]
                        self.x_opt = self.population[i]
                    successful_count += 1

            # Update Covariance Matrix (simplified)
            diff = self.population - np.mean(self.population, axis=0)
            C = (1 - learning_rate) * C + learning_rate * np.cov(diff.T)
            
            # Adjust population size
            success_rate = successful_count / self.popsize
            self.success_history.append(success_rate)
            if len(self.success_history) > self.success_history_length:
                self.success_history.pop(0)
            
            avg_success_rate = np.mean(self.success_history)

            if avg_success_rate < self.target_success_rate / 2 and self.popsize > 4:
                self.popsize = int(self.popsize * self.popsize_reduction_factor)
                self.population = self.population[np.argsort(self.fitness)[:self.popsize]]
                self.fitness = self.fitness[np.argsort(self.fitness)[:self.popsize]]
                print(f"Reducing popsize to {self.popsize}")

            elif avg_success_rate > self.target_success_rate * 2 and self.eval_count < self.budget // 2:
                self.popsize = int(self.popsize * self.popsize_increase_factor)
                self.popsize = min(self.popsize, self.budget // 2)
                new_population = np.random.uniform(lb, ub, size=(self.popsize - len(self.population), self.dim))
                new_fitness = np.array([func(x) for x in new_population])
                self.eval_count += len(new_population)
                self.population = np.vstack((self.population, new_population))
                self.fitness = np.concatenate((self.fitness, new_fitness))
                print(f"Increasing popsize to {self.popsize}")

        return self.f_opt, self.x_opt