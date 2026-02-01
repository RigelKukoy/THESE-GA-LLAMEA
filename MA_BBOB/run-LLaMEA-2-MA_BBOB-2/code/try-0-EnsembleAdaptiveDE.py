import numpy as np

class EnsembleAdaptiveDE:
    def __init__(self, budget=10000, dim=10, popsize=None, F=0.5, CR=0.7, local_search_iterations=5, local_search_stepsize=0.01, mutation_strategy_probs=None):
        self.budget = budget
        self.dim = dim
        self.popsize = popsize if popsize is not None else 10 * self.dim
        self.F = F
        self.CR = CR
        self.local_search_iterations = local_search_iterations
        self.local_search_stepsize = local_search_stepsize
        self.mutation_strategy_probs = mutation_strategy_probs if mutation_strategy_probs is not None else [0.4, 0.3, 0.3]  # Probabilities for rand/1, current-to-best, and rand/2

    def gradient_descent(self, func, x, iterations, step_size):
        x_current = x.copy()
        f_current = func(x_current)
        eval_count = 1
        for _ in range(iterations):
            # Estimate gradient (simplified finite difference)
            gradient = np.zeros_like(x_current)
            for i in range(self.dim):
                x_plus = x_current.copy()
                x_minus = x_current.copy()
                delta = step_size
                x_plus[i] += delta
                x_minus[i] -= delta
                x_plus = np.clip(x_plus, func.bounds.lb, func.bounds.ub)
                x_minus = np.clip(x_minus, func.bounds.lb, func.bounds.ub)
                f_plus = func(x_plus)
                f_minus = func(x_minus)
                eval_count += 2
                gradient[i] = (f_plus - f_minus) / (2 * delta)

            x_new = x_current - step_size * gradient
            x_new = np.clip(x_new, func.bounds.lb, func.bounds.ub)
            f_new = func(x_new)
            eval_count += 1
            if f_new < f_current:
                f_current = f_new
                x_current = x_new
            else:
                break # Stop if no improvement
        return f_current, x_current, eval_count


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
                # Mutation Strategy Ensemble
                rand_val = np.random.rand()
                if rand_val < self.mutation_strategy_probs[0]:  # rand/1
                    idxs = np.random.choice(self.popsize, 3, replace=False)
                    x1, x2, x3 = self.population[idxs]
                    mutant = x1 + self.F * (x2 - x3)
                elif rand_val < self.mutation_strategy_probs[0] + self.mutation_strategy_probs[1]:  # current-to-best
                    best_idx = np.argmin(self.fitness)
                    x_best = self.population[best_idx]
                    idxs = np.random.choice(self.popsize, 1, replace=False)
                    x1 = self.population[idxs[0]]
                    mutant = self.population[i] + self.F * (x_best - self.population[i]) + self.F * (self.population[i] - x1)

                else:  # rand/2
                    idxs = np.random.choice(self.popsize, 5, replace=False)
                    x1, x2, x3, x4, x5 = self.population[idxs]
                    mutant = x1 + self.F * (x2 - x3) + self.F * (x4 - x5)

                mutant = np.clip(mutant, lb, ub)

                # Adaptive Crossover
                CR_individual = np.random.normal(self.CR, 0.1)
                CR_individual = np.clip(CR_individual, 0, 1)
                crossover_mask = np.random.rand(self.dim) < CR_individual
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

            # Gradient-based Local Search (applied to best individual)
            f_local, x_local, ls_eval_count = self.gradient_descent(func, self.x_opt, self.local_search_iterations, self.local_search_stepsize)
            self.eval_count += ls_eval_count
            if f_local < self.f_opt:
                self.f_opt = f_local
                self.x_opt = x_local

        return self.f_opt, self.x_opt