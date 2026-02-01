import numpy as np

class AdaptiveDEMutationLocalSearch:
    def __init__(self, budget=10000, dim=10, popsize=None, F_initial=0.5, CR=0.7, local_search_prob=0.1, local_search_stepsize=0.1, success_history_size=10):
        self.budget = budget
        self.dim = dim
        self.popsize = popsize if popsize is not None else 10 * self.dim
        self.F_initial = F_initial
        self.CR = CR
        self.local_search_prob = local_search_prob
        self.local_search_stepsize = local_search_stepsize
        self.success_history_size = success_history_size
        self.success_history = []  # Store the success/failure of recent mutations

    def __call__(self, func):
        lb = func.bounds.lb
        ub = func.bounds.ub
        self.population = np.random.uniform(lb, ub, size=(self.popsize, self.dim))
        self.fitness = np.array([func(x) for x in self.population])
        self.eval_count = self.popsize
        self.f_opt = np.min(self.fitness)
        self.x_opt = self.population[np.argmin(self.fitness)]
        self.F = self.F_initial  # Start with the initial F value

        while self.eval_count < self.budget:
            for i in range(self.popsize):
                # Mutation: Adjust mutation strength (F) based on success history
                if self.success_history:
                    success_rate = np.mean(self.success_history)
                    self.F = self.F_initial * (1 + success_rate)  # Increase F if mutations are frequently successful
                    self.F = np.clip(self.F, 0.1, 1.0) # Keep F within reasonable bounds
                else:
                     self.F = self.F_initial

                indices = np.random.choice(self.popsize, 3, replace=False)
                x1, x2, x3 = self.population[indices]
                mutant = self.population[i] + self.F * (x2 - x3)
                mutant = np.clip(mutant, lb, ub)

                # Crossover
                crossover_mask = np.random.rand(self.dim) < self.CR
                trial = np.where(crossover_mask, mutant, self.population[i])

                # Selection
                f_trial = func(trial)
                self.eval_count += 1

                if f_trial < self.fitness[i]:
                    self.success_history.append(1)
                    self.population[i] = trial
                    self.fitness[i] = f_trial
                    if self.fitness[i] < self.f_opt:
                        self.f_opt = self.fitness[i]
                        self.x_opt = self.population[i]
                else:
                    self.success_history.append(0)
                
                # Local Search
                if np.random.rand() < self.local_search_prob:
                    x_local = self.population[i].copy()
                    for d in range(self.dim):
                        x_local[d] += np.random.uniform(-self.local_search_stepsize, self.local_search_stepsize)
                        x_local[d] = np.clip(x_local[d], lb, ub)
                    
                    f_local = func(x_local)
                    self.eval_count += 1
                    
                    if f_local < self.fitness[i]:
                        self.population[i] = x_local
                        self.fitness[i] = f_local
                        if self.fitness[i] < self.f_opt:
                            self.f_opt = self.fitness[i]
                            self.x_opt = self.population[i]

            # Maintain success history size
            if len(self.success_history) > self.success_history_size:
                self.success_history = self.success_history[-self.success_history_size:]
            
            if self.eval_count > self.budget:
                break

        return self.f_opt, self.x_opt