import numpy as np
from scipy.optimize import minimize

class DE_NM:
    def __init__(self, budget=10000, dim=10, pop_size=30, F=0.5, Cr=0.7, nm_iterations=5, stagnation_threshold=100):
        self.budget = budget
        self.dim = dim
        self.pop_size = pop_size
        self.F = F
        self.Cr = Cr
        self.nm_iterations = nm_iterations
        self.stagnation_threshold = stagnation_threshold
        self.stagnation_counter = 0
        self.de_active = True # Start with DE

    def __call__(self, func):
        # Initialize population
        population = np.random.uniform(func.bounds.lb, func.bounds.ub, size=(self.pop_size, self.dim))
        fitness = np.array([func(x) for x in population])
        self.budget -= self.pop_size

        best_index = np.argmin(fitness)
        best_position = population[best_index].copy()
        best_value = fitness[best_index]

        while self.budget > 0:
            if self.de_active:
                # Differential Evolution
                for i in range(self.pop_size):
                    # Mutation
                    idxs = [idx for idx in range(self.pop_size) if idx != i]
                    a, b, c = population[np.random.choice(idxs, 3, replace=False)]
                    mutant = a + self.F * (b - c)
                    mutant = np.clip(mutant, func.bounds.lb, func.bounds.ub)

                    # Crossover
                    crossover = np.random.rand(self.dim) < self.Cr
                    trial = np.where(crossover, mutant, population[i])

                    # Selection
                    trial_fitness = func(trial)
                    self.budget -= 1
                    if self.budget <= 0:
                        break

                    if trial_fitness < fitness[i]:
                        fitness[i] = trial_fitness
                        population[i] = trial.copy()

                        if trial_fitness < best_value:
                            best_value = trial_fitness
                            best_position = trial.copy()

                if self.budget <= 0:
                    break

                # Check for stagnation
                if np.min(fitness) >= best_value:
                    self.stagnation_counter += 1
                else:
                     self.stagnation_counter = 0
                
                if self.stagnation_counter >= self.stagnation_threshold:
                    self.de_active = False # Switch to Nelder-Mead
                    self.stagnation_counter = 0 #reset counter
                    #print("Switching to Nelder-Mead")

            else:
                # Nelder-Mead Local Search around the best solution
                result = minimize(func, best_position, method='Nelder-Mead', options={'maxiter': self.nm_iterations, 'maxfev':self.budget})
                if result.success:
                    best_position = result.x
                    best_value = result.fun
                self.budget -= result.nfev
                if self.budget <= 0:
                    break
                self.de_active = True # Switch back to DE after local search
                #print("Switching back to DE")
                # Re-evaluate population to update values based on the NM improvement
                fitness = np.array([func(x) for x in population])
                self.budget -= self.pop_size - 1 #subtract 1, as best_value is known from above
                if self.budget <= 0:
                    break



        self.f_opt = best_value
        self.x_opt = best_position
        return self.f_opt, self.x_opt