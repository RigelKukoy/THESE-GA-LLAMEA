import numpy as np
from scipy.optimize import minimize

class HybridPSO_DE_NM:
    def __init__(self, budget=10000, dim=10, pop_size=20, pso_weight=0.7, de_cross_rate=0.9, nm_iterations=5):
        self.budget = budget
        self.dim = dim
        self.pop_size = pop_size
        self.pso_weight = pso_weight
        self.de_cross_rate = de_cross_rate
        self.nm_iterations = nm_iterations

    def __call__(self, func):
        # Initialize population
        pop = np.random.uniform(func.bounds.lb, func.bounds.ub, size=(self.pop_size, self.dim))
        fitness = np.array([func(x) for x in pop])
        self.budget -= self.pop_size
        
        # Initialize velocities for PSO
        velocities = np.random.uniform(-1, 1, size=(self.pop_size, self.dim))

        # Initialize best positions and fitness
        personal_best_positions = pop.copy()
        personal_best_fitness = fitness.copy()
        global_best_index = np.argmin(fitness)
        global_best_position = pop[global_best_index].copy()
        global_best_fitness = fitness[global_best_index]
        
        while self.budget > 0:
            for i in range(self.pop_size):
                # PSO update
                inertia = self.pso_weight * velocities[i]
                cognitive = np.random.rand() * (personal_best_positions[i] - pop[i])
                social = np.random.rand() * (global_best_position - pop[i])
                velocities[i] = inertia + cognitive + social
                pop[i] += velocities[i]
                pop[i] = np.clip(pop[i], func.bounds.lb, func.bounds.ub)
                
                # Differential Evolution mutation
                if np.random.rand() < self.de_cross_rate:
                    indices = np.random.choice(self.pop_size, 3, replace=False)
                    x_r1, x_r2, x_r3 = pop[indices[0]], pop[indices[1]], pop[indices[2]]
                    mutation = x_r1 + 0.5 * (x_r2 - x_r3)
                    
                    # Crossover
                    crossover_mask = np.random.rand(self.dim) < self.de_cross_rate
                    pop[i] = np.where(crossover_mask, mutation, pop[i])
                    pop[i] = np.clip(pop[i], func.bounds.lb, func.bounds.ub)
                
                # Evaluate new position
                f = func(pop[i])
                self.budget -= 1
                
                if f < personal_best_fitness[i]:
                    personal_best_fitness[i] = f
                    personal_best_positions[i] = pop[i].copy()

                    if f < global_best_fitness:
                        global_best_fitness = f
                        global_best_position = pop[i].copy()

                # Nelder-Mead local search (applied to the best particle)
                if i == global_best_index and self.budget > 0:
                    nm_result = minimize(func, global_best_position, method='Nelder-Mead',
                                          options={'maxiter': self.nm_iterations, 'maxfev': self.budget})

                    if nm_result.success and nm_result.fun < global_best_fitness:
                        global_best_fitness = nm_result.fun
                        global_best_position = nm_result.x
                        
                    self.budget -= nm_result.nfev
            
            global_best_index = np.argmin(personal_best_fitness)
            global_best_position = personal_best_positions[global_best_index].copy()
            global_best_fitness = personal_best_fitness[global_best_index]

        return global_best_fitness, global_best_position