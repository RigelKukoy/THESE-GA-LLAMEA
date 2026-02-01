import numpy as np

class CooperativeAdaptiveDE:
    def __init__(self, budget=10000, dim=10, popsize=None, F=0.5, CR=0.7,
                 F_adaptive=True, CR_adaptive=True, restart_patience=500):
        self.budget = budget
        self.dim = dim
        self.popsize = popsize if popsize is not None else 10 * self.dim
        self.F = F
        self.CR = CR
        self.F_adaptive = F_adaptive
        self.CR_adaptive = CR_adaptive
        self.restart_patience = restart_patience
        self.best_fitness_history = []

    def __call__(self, func):
        lb = func.bounds.lb
        ub = func.bounds.ub
        self.population = np.random.uniform(lb, ub, size=(self.popsize, self.dim))
        self.fitness = np.array([func(x) for x in self.population])
        self.eval_count = self.popsize
        self.f_opt = np.min(self.fitness)
        self.x_opt = self.population[np.argmin(self.fitness)]

        memory_F = np.ones(self.popsize) * self.F
        memory_CR = np.ones(self.popsize) * self.CR
        success_F = []
        success_CR = []

        self.best_fitness_history.append(self.f_opt)
        stagnation_counter = 0

        while self.eval_count < self.budget:
            for i in range(self.popsize):
                # Cooperative strategy: choose a different random base vector for each dimension
                base_idx = np.random.randint(0, self.popsize)
                idxs = np.random.choice(self.popsize, 2, replace=False)
                x1, x2 = self.population[idxs]
                base_vector = self.population[base_idx]

                if self.F_adaptive:
                    F = memory_F[i]
                else:
                    F = self.F

                mutant = base_vector + F * (x1 - x2)
                mutant = np.clip(mutant, lb, ub)

                if self.CR_adaptive:
                    CR = memory_CR[i]
                else:
                    CR = self.CR

                crossover_mask = np.random.rand(self.dim) < CR
                trial = np.where(crossover_mask, mutant, self.population[i])

                f_trial = func(trial)
                self.eval_count += 1

                if f_trial < self.fitness[i]:
                    success_F.append(F)
                    success_CR.append(CR)
                    self.population[i] = trial
                    self.fitness[i] = f_trial
                    if self.fitness[i] < self.f_opt:
                        self.f_opt = self.fitness[i]
                        self.x_opt = self.population[i]
                        stagnation_counter = 0 # Reset stagnation counter

                    if self.F_adaptive:
                        memory_F[i] = np.random.normal(self.F, 0.1)
                        memory_F[i] = np.clip(memory_F[i], 0.1, 1.0)
                    if self.CR_adaptive:
                        memory_CR[i] = np.random.normal(self.CR, 0.1)
                        memory_CR[i] = np.clip(memory_CR[i], 0.1, 1.0)
                else:
                    stagnation_counter += 1 # Increment stagnation counter

            # Adaptive F and CR based on success history
            if self.F_adaptive and len(success_F) > 0:
                self.F = np.mean(success_F)
                self.F = np.clip(self.F, 0.1, 1.0)
                success_F = []
            if self.CR_adaptive and len(success_CR) > 0:
                self.CR = np.mean(success_CR)
                self.CR = np.clip(self.CR, 0.1, 1.0)
                success_CR = []

            self.best_fitness_history.append(self.f_opt)

            # Restart strategy based on stagnation
            if stagnation_counter > self.restart_patience:
                self.population = np.random.uniform(lb, ub, size=(self.popsize, self.dim))
                self.fitness = np.array([func(x) for x in self.population])
                self.eval_count += self.popsize
                best_idx = np.argmin(self.fitness)
                self.f_opt = self.fitness[best_idx]
                self.x_opt = self.population[best_idx]
                memory_F = np.ones(self.popsize) * self.F
                memory_CR = np.ones(self.popsize) * self.CR
                stagnation_counter = 0
                self.best_fitness_history.append(self.f_opt)
                print("Restarting population due to stagnation.")

        return self.f_opt, self.x_opt