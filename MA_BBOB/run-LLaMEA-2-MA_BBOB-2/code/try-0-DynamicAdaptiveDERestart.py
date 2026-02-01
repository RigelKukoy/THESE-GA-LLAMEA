import numpy as np

class DynamicAdaptiveDERestart:
    def __init__(self, budget=10000, dim=10, popsize=None, F_initial=0.5, CR_initial=0.7, F_adapt_rate=0.1, CR_adapt_rate=0.1, restart_trigger=0.05):
        self.budget = budget
        self.dim = dim
        self.popsize = popsize if popsize is not None else 10 * self.dim
        self.F = F_initial
        self.CR = CR_initial
        self.F_adapt_rate = F_adapt_rate
        self.CR_adapt_rate = CR_adapt_rate
        self.restart_trigger = restart_trigger  # Threshold for fitness improvement to trigger restart
        self.last_improvement = 0  # Generation count since last improvement
        self.restart_interval = 50 # Number of iterations before checking restart condition


    def __call__(self, func):
        lb = func.bounds.lb
        ub = func.bounds.ub
        self.population = np.random.uniform(lb, ub, size=(self.popsize, self.dim))
        self.fitness = np.array([func(x) for x in self.population])
        self.eval_count = self.popsize
        self.f_opt = np.min(self.fitness)
        self.x_opt = self.population[np.argmin(self.fitness)]
        self.initial_f_opt = self.f_opt


        generation = 0
        while self.eval_count < self.budget:
            generation += 1
            successful_mutations = 0

            for i in range(self.popsize):
                # Mutation
                idxs = np.random.choice(self.popsize, 3, replace=False)
                x1, x2, x3 = self.population[idxs]
                mutant = x1 + self.F * (x2 - x3)
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
                    successful_mutations += 1

                    if self.fitness[i] < self.f_opt:
                        self.f_opt = self.fitness[i]
                        self.x_opt = self.population[i]
                        self.last_improvement = generation
                        

            # Adapt F and CR based on success rate
            success_rate = successful_mutations / self.popsize
            self.F = np.clip(self.F + self.F_adapt_rate * (success_rate - 0.5), 0.1, 0.9)
            self.CR = np.clip(self.CR + self.CR_adapt_rate * (success_rate - 0.5), 0.1, 0.9)

            # Restart mechanism
            if generation - self.last_improvement > self.restart_interval:
                if (self.initial_f_opt - self.f_opt) / self.initial_f_opt < self.restart_trigger:
                    # Trigger restart: re-initialize a portion of the population
                    num_to_restart = int(0.2 * self.popsize)  # Restart 20% of population
                    idxs_to_restart = np.random.choice(self.popsize, num_to_restart, replace=False)
                    self.population[idxs_to_restart] = np.random.uniform(lb, ub, size=(num_to_restart, self.dim))
                    self.fitness[idxs_to_restart] = np.array([func(x) for x in self.population[idxs_to_restart]])
                    self.eval_count += num_to_restart
                    
                    best_idx = np.argmin(self.fitness)
                    self.f_opt = self.fitness[best_idx]
                    self.x_opt = self.population[best_idx]
                    self.last_improvement = generation
                    self.initial_f_opt = self.f_opt


        return self.f_opt, self.x_opt