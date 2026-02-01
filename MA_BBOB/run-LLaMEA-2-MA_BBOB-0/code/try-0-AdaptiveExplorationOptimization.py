import numpy as np

class AdaptiveExplorationOptimization:
    def __init__(self, budget=10000, dim=10, pop_size=20, de_rate=0.7, local_rate=0.1):
        """
        Args:
            budget (int): The evaluation budget.
            dim (int): The dimension of the problem.
            pop_size (int): The population size.
            de_rate (float): Probability of performing differential evolution.
            local_rate (float): Probability of performing local search.
        """
        self.budget = budget
        self.dim = dim
        self.pop_size = pop_size
        self.de_rate = de_rate
        self.local_rate = local_rate
        self.f_opt = np.inf
        self.x_opt = None

    def __call__(self, func):
        """
        Optimizes the given function using the allocated budget.

        Args:
            func (callable): The function to optimize.

        Returns:
            tuple: A tuple containing the best function value found and the corresponding solution vector.
        """
        # Initialize population within bounds
        population = np.random.uniform(func.bounds.lb, func.bounds.ub, size=(self.pop_size, self.dim))
        fitness = np.array([func(x) for x in population])
        self.budget -= self.pop_size

        # Update the best solution
        best_index = np.argmin(fitness)
        if fitness[best_index] < self.f_opt:
            self.f_opt = fitness[best_index]
            self.x_opt = population[best_index].copy()

        # Optimization loop
        while self.budget > 0:
            for i in range(self.pop_size):
                # Adaptive strategy selection
                if np.random.rand() < self.de_rate:
                    # Differential Evolution
                    idxs = np.random.choice(self.pop_size, 3, replace=False)
                    x_r1, x_r2, x_r3 = population[idxs]
                    mutant = population[i] + 0.8 * (x_r1 - x_r2) 
                    
                    # Ensure mutant stays within bounds
                    mutant = np.clip(mutant, func.bounds.lb, func.bounds.ub)

                    # Crossover
                    cross_points = np.random.rand(self.dim) < 0.9
                    if not np.any(cross_points):
                        cross_points[np.random.randint(0, self.dim)] = True
                    trial = np.where(cross_points, mutant, population[i])
                elif np.random.rand() < self.local_rate:
                    # Local Search (perturbation)
                    trial = population[i] + 0.1 * np.random.normal(0, 1, self.dim) * (func.bounds.ub - func.bounds.lb)
                    trial = np.clip(trial, func.bounds.lb, func.bounds.ub)
                else:
                    # Global Search (random)
                    trial = np.random.uniform(func.bounds.lb, func.bounds.ub)

                # Evaluation
                f_trial = func(trial)
                self.budget -= 1

                # Selection
                if f_trial < fitness[i]:
                    fitness[i] = f_trial
                    population[i] = trial.copy()

                    # Update the best solution
                    if f_trial < self.f_opt:
                        self.f_opt = f_trial
                        self.x_opt = trial.copy()
                
                if self.budget <= 0:
                    break

        return self.f_opt, self.x_opt