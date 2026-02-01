import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C

class GPSurrogateDE:
    def __init__(self, budget=10000, dim=10, pop_size=20, n_initial_samples=10, exploration_weight=0.1):
        self.budget = budget
        self.dim = dim
        self.pop_size = pop_size
        self.n_initial_samples = n_initial_samples
        self.exploration_weight = exploration_weight
        self.X = None
        self.y = None
        self.gpr = GaussianProcessRegressor(kernel=C(1.0, (1e-3, 1e3)) * RBF(10, (1e-2, 1e2)),
                                             n_restarts_optimizer=10, alpha=1e-5)
        self.F = 0.5
        self.CR = 0.7


    def __call__(self, func):
        self.f_opt = np.Inf
        self.x_opt = None

        # Initial sampling
        X_initial = np.random.uniform(func.bounds.lb, func.bounds.ub, size=(self.n_initial_samples, self.dim))
        y_initial = np.array([func(x) for x in X_initial])
        self.budget -= self.n_initial_samples
        self.X = X_initial
        self.y = y_initial

        best_index = np.argmin(self.y)
        self.f_opt = self.y[best_index]
        self.x_opt = self.X[best_index]


        # DE population initialization
        population = np.random.uniform(func.bounds.lb, func.bounds.ub, size=(self.pop_size, self.dim))
        fitness = np.array([func(x) for x in population])
        self.budget -= self.pop_size

        for i in range(self.pop_size):
            if fitness[i] < self.f_opt:
                self.f_opt = fitness[i]
                self.x_opt = population[i]

        while self.budget > 0:
            # Train GP model
            self.gpr.fit(self.X, self.y)

            for i in range(self.pop_size):
                # Mutation
                idxs = np.random.choice(self.pop_size, 3, replace=False)
                x_1, x_2, x_3 = population[idxs]
                mutant = x_1 + self.F * (x_2 - x_3)
                mutant = np.clip(mutant, func.bounds.lb, func.bounds.ub)

                # Crossover
                crossover = np.random.uniform(size=self.dim) < self.CR
                trial = np.where(crossover, mutant, population[i])

                # Evaluate trial point using GP surrogate and exploration bonus
                f_trial_pred, sigma = self.gpr.predict(trial.reshape(1, -1), return_std=True)
                f_trial_gp = f_trial_pred[0] - self.exploration_weight * sigma[0]  # Exploration bonus

                # Evaluate trial point with the actual function with a small probability
                if np.random.rand() < 0.1 or self.budget < 50: # Explore more in the end
                    f_trial = func(trial)
                    self.budget -= 1

                    self.X = np.vstack((self.X, trial))
                    self.y = np.append(self.y, f_trial)

                    if f_trial < self.f_opt:
                        self.f_opt = f_trial
                        self.x_opt = trial
                    
                    if f_trial < fitness[i]:
                        fitness[i] = f_trial
                        population[i] = trial
                else:
                     f_trial = f_trial_gp
                
                if f_trial < fitness[i]:
                    population[i] = trial

            # Update best solution based on GP predictions (exploitation)
            fitness_pred, _ = self.gpr.predict(population, return_std=True)
            best_index = np.argmin(fitness_pred)
            if fitness_pred[best_index] < self.f_opt:
                self.x_opt = population[best_index]
                
                #Evaluate the real function for the predicted optimum
                f_opt_real = func(self.x_opt)
                self.budget -= 1

                self.X = np.vstack((self.X, self.x_opt))
                self.y = np.append(self.y, f_opt_real)

                self.f_opt = f_opt_real
                

        return self.f_opt, self.x_opt