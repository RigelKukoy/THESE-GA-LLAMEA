import numpy as np

class GradientAdaptiveDE:
    def __init__(self, budget=10000, dim=10, popsize=None, F=0.5, CR=0.7, grad_samples=5, grad_step=0.1):
        self.budget = budget
        self.dim = dim
        self.popsize = popsize if popsize is not None else 10 * self.dim
        self.F = F
        self.CR = CR
        self.grad_samples = grad_samples  # Number of samples to estimate gradient
        self.grad_step = grad_step  # Step size for gradient estimation
        self.population = None
        self.fitness = None
        self.eval_count = 0
        self.f_opt = np.Inf
        self.x_opt = None

    def estimate_gradient(self, func, x):
        """Estimates the gradient of the fitness landscape at a given point x."""
        gradient = np.zeros(self.dim)
        for _ in range(self.grad_samples):
            direction = np.random.randn(self.dim)
            direction /= np.linalg.norm(direction)  # Normalize direction

            x_plus = x + self.grad_step * direction
            x_minus = x - self.grad_step * direction

            # Clip values to stay within bounds
            x_plus = np.clip(x_plus, func.bounds.lb, func.bounds.ub)
            x_minus = np.clip(x_minus, func.bounds.lb, func.bounds.ub)

            f_plus = func(x_plus)
            f_minus = func(x_minus)
            self.eval_count += 2  # Account for function evaluations

            gradient += (f_plus - f_minus) * direction

        return gradient / (2 * self.grad_samples * self.grad_step)

    def gradient_guided_mutation(self, func, i, lb, ub):
        """Performs mutation guided by the estimated gradient."""
        gradient = self.estimate_gradient(func, self.population[i])
        idxs = np.random.choice(self.popsize, 2, replace=False)  # Select two random individuals
        x1, x2 = self.population[idxs]

        # The mutation moves towards the negative gradient direction, scaled by F
        mutant = self.population[i] + self.F * (self.population[i] - x1) + self.F * (x2 - self.population[i]) - self.F * gradient

        mutant = np.clip(mutant, lb, ub)
        return mutant

    def __call__(self, func):
        lb = func.bounds.lb
        ub = func.bounds.ub

        # Initialize population
        self.population = np.random.uniform(lb, ub, size=(self.popsize, self.dim))
        self.fitness = np.array([func(x) for x in self.population])
        self.eval_count = self.popsize
        self.f_opt = np.min(self.fitness)
        self.x_opt = self.population[np.argmin(self.fitness)]

        while self.eval_count < self.budget:
            for i in range(self.popsize):
                # Mutation
                mutant = self.gradient_guided_mutation(func, i, lb, ub)

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

        return self.f_opt, self.x_opt