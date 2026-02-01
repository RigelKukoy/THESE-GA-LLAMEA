import numpy as np
from scipy.stats import norm
from scipy.optimize import minimize

class BayesianAdaptiveOptimizer:
    def __init__(self, budget=10000, dim=10, pop_size=5, local_search_probability=0.2, exploration_probability=0.2, acquisition_function="EI", xi=0.01, exploitation_intensity=0.1):
        self.budget = budget
        self.dim = dim
        self.pop_size = pop_size
        self.local_search_probability = local_search_probability
        self.exploration_probability = exploration_probability
        self.acquisition_function = acquisition_function
        self.xi = xi
        self.exploitation_intensity = exploitation_intensity
        self.population = np.random.uniform(-5, 5, size=(pop_size, dim))
        self.fitness = np.full(pop_size, np.inf)
        self.best_fitness = np.inf
        self.best_position = None
        self.gaussian_process_mean = np.zeros(dim)
        self.gaussian_process_std = np.ones(dim)

    def expected_improvement(self, x):
        mean = np.dot(x, self.gaussian_process_mean)
        std = np.sqrt(np.dot(x**2, self.gaussian_process_std**2))
        
        if std == 0:
            return 0  # Avoid division by zero

        z = (self.best_fitness - mean - self.xi) / std
        return (self.best_fitness - mean - self.xi) * norm.cdf(z) + std * norm.pdf(z)

    def local_search(self, func, x_start, budget_fraction):
         # Define the objective function for local search
        def objective(x):
            return func(x)

        # Define the bounds for the optimization
        bounds = [(func.bounds.lb, func.bounds.ub) for _ in range(self.dim)]
        
        # Initialize the best fitness with a large value
        best_fitness = np.inf
        best_x = None
        
        # Cap the number of iterations based on the budget fraction
        max_iterations = int(budget_fraction * self.budget)
        
        # Perform local optimization using L-BFGS-B
        result = minimize(objective, x_start, method='L-BFGS-B', bounds=bounds, options={'maxiter': max_iterations})
        
        # Update best fitness and best x
        if result.success and result.fun < best_fitness:
            best_fitness = result.fun
            best_x = result.x
        
        return best_fitness, best_x

    def exploration_sample(self, func):
        return np.random.uniform(func.bounds.lb, func.bounds.ub, size=self.dim)
    
    def __call__(self, func):
        eval_count = 0
        
        while eval_count < self.budget:
            # Evaluate initial population
            for i in range(self.pop_size):
                if eval_count < self.budget:
                    f = func(self.population[i])
                    eval_count += 1
                    self.fitness[i] = f
                    if f < self.best_fitness:
                        self.best_fitness = f
                        self.best_position = self.population[i].copy()

            # Select individual for update
            selected_index = np.random.randint(self.pop_size)
            
            # Adaptive Strategy Selection
            rand = np.random.rand()
            
            if rand < self.local_search_probability:
                # Local Search
                budget_fraction = self.exploitation_intensity * (1- eval_count / self.budget) #reduce intensity of exploitation over time
                
                local_fitness, local_position = self.local_search(func, self.population[selected_index].copy(), budget_fraction)
                
                if local_fitness < self.fitness[selected_index]:
                   self.fitness[selected_index] = local_fitness
                   self.population[selected_index] = local_position.copy()
                   if local_fitness < self.best_fitness:
                       self.best_fitness = local_fitness
                       self.best_position = local_position.copy()
            elif rand < self.local_search_probability + self.exploration_probability:
                # Global Exploration
                new_sample = self.exploration_sample(func)
                f = func(new_sample)
                eval_count += 1

                if f < self.fitness[selected_index]:
                    self.fitness[selected_index] = f
                    self.population[selected_index] = new_sample.copy()
                    if f < self.best_fitness:
                        self.best_fitness = f
                        self.best_position = new_sample.copy()
            else:
                # Bayesian Optimization Update (Simplified)
                bounds = [(func.bounds.lb, func.bounds.ub) for _ in range(self.dim)]
                
                # Define the acquisition function for optimization
                def acquisition(x):
                   return -self.expected_improvement(x) #Scipy minimize finds minimum, so need to take negative of EI
                
                # Start the optimization from a random point
                x0 = np.random.uniform(func.bounds.lb, func.bounds.ub, size=self.dim)
                
                # Run the optimization
                result = minimize(acquisition, x0, method='L-BFGS-B', bounds=bounds)
                
                # Get the new sample
                new_sample = result.x

                f = func(new_sample)
                eval_count += 1

                if f < self.fitness[selected_index]:
                    self.fitness[selected_index] = f
                    self.population[selected_index] = new_sample.copy()
                    if f < self.best_fitness:
                        self.best_fitness = f
                        self.best_position = new_sample.copy()

            # Update Gaussian Process (Very simplified)
            self.gaussian_process_mean = self.best_position
            self.gaussian_process_std = np.std(self.population, axis=0)

        return self.best_fitness, self.best_position