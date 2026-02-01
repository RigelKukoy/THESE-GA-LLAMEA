import numpy as np

class AdaptiveHybridOptimization:
    def __init__(self, budget=10000, dim=10, pop_size=None, de_rate=0.7, cma_rate=0.2, restart_trigger=0.01):
        """
        Args:
            budget (int): The evaluation budget.
            dim (int): The dimension of the problem.
            pop_size (int): The population size. If None, it's set to 4 + int(3 * np.log(dim)).
            de_rate (float): Probability of performing differential evolution.
            cma_rate (float): Probability of performing CMA-ES-like adaptation.
            restart_trigger (float): Threshold for triggering a population restart based on fitness stagnation.
        """
        self.budget = budget
        self.dim = dim
        self.pop_size = 4 + int(3 * np.log(dim)) if pop_size is None else pop_size
        self.de_rate = de_rate
        self.cma_rate = cma_rate
        self.restart_trigger = restart_trigger
        self.f_opt = np.inf
        self.x_opt = None
        self.mean = None  # CMA-ES like mean
        self.sigma = 0.5  # CMA-ES like step size
        self.cov = None # Covariance matrix

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
        self.mean = np.mean(population, axis=0)
        self.cov = np.eye(self.dim) # Initialize covariance matrix
        fitness = np.array([func(x) for x in population])
        self.budget -= self.pop_size

        # Update the best solution
        best_index = np.argmin(fitness)
        if fitness[best_index] < self.f_opt:
            self.f_opt = fitness[best_index]
            self.x_opt = population[best_index].copy()
            self.best_fit_history = [self.f_opt]

        # Optimization loop
        while self.budget > 0:
            new_population = np.zeros_like(population)
            new_fitness = np.zeros_like(fitness)

            for i in range(self.pop_size):
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
                elif np.random.rand() < self.cma_rate:
                    # CMA-ES-like adaptation
                    try:
                        z = np.random.multivariate_normal(np.zeros(self.dim), self.cov)
                        trial = self.mean + self.sigma * z
                    except: # Handle singular covariance matrix
                        trial = np.random.uniform(func.bounds.lb, func.bounds.ub, size=self.dim)
                    trial = np.clip(trial, func.bounds.lb, func.bounds.ub)
                else:
                    # Global Search (random)
                    trial = np.random.uniform(func.bounds.lb, func.bounds.ub)

                # Evaluation
                f_trial = func(trial)
                self.budget -= 1

                # Selection
                if f_trial < fitness[i]:
                    new_fitness[i] = f_trial
                    new_population[i] = trial.copy()
                else:
                    new_fitness[i] = fitness[i]
                    new_population[i] = population[i].copy()

                # Update the best solution
                if f_trial < self.f_opt:
                    self.f_opt = f_trial
                    self.x_opt = trial.copy()
                
                if self.budget <= 0:
                    break
            
            population = new_population
            fitness = new_fitness

            # CMA-ES adaptation
            self.mean = np.mean(population, axis=0)
            C = np.cov(population.T)
            if np.linalg.det(C) > 0: # Check determinant
                self.cov = C

            # Restart mechanism
            self.best_fit_history.append(self.f_opt)
            if len(self.best_fit_history) > 100:
                self.best_fit_history.pop(0)
                if np.std(self.best_fit_history) < self.restart_trigger:
                    # Restart population
                    population = np.random.uniform(func.bounds.lb, func.bounds.ub, size=(self.pop_size, self.dim))
                    fitness = np.array([func(x) for x in population])
                    self.mean = np.mean(population, axis=0)
                    self.cov = np.eye(self.dim)
                    best_index = np.argmin(fitness)
                    if fitness[best_index] < self.f_opt:
                        self.f_opt = fitness[best_index]
                        self.x_opt = population[best_index].copy()
                    self.sigma = 0.5
                    self.best_fit_history = [self.f_opt]

        return self.f_opt, self.x_opt