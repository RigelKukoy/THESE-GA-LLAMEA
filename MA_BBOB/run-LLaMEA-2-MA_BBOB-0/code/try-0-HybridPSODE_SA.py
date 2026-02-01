import numpy as np

class HybridPSODE_SA:
    def __init__(self, budget=10000, dim=10, pop_size=20, pso_weight=0.7, de_cross_rate=0.9, initial_temp=1.0, cooling_rate=0.95):
        self.budget = budget
        self.dim = dim
        self.pop_size = pop_size
        self.pso_weight = pso_weight
        self.de_cross_rate = de_cross_rate
        self.initial_temp = initial_temp
        self.cooling_rate = cooling_rate

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
        
        temp = self.initial_temp  # Initialize temperature for SA

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
                    mutation = x_r1 + 0.5 * (x_r2 - x_r3)  #DE mutation
                    
                    # Crossover
                    crossover_mask = np.random.rand(self.dim) < self.de_cross_rate
                    pop[i] = np.where(crossover_mask, mutation, pop[i])
                    pop[i] = np.clip(pop[i], func.bounds.lb, func.bounds.ub)
                
                # Evaluate new position
                f = func(pop[i])
                self.budget -= 1
                
                # Simulated Annealing acceptance criterion
                delta_e = f - personal_best_fitness[i]
                if delta_e < 0:
                    personal_best_fitness[i] = f
                    personal_best_positions[i] = pop[i].copy()
                    if f < global_best_fitness:
                        global_best_fitness = f
                        global_best_position = pop[i].copy()
                else:
                    acceptance_prob = np.exp(-delta_e / temp)
                    if np.random.rand() < acceptance_prob:
                        personal_best_fitness[i] = f
                        personal_best_positions[i] = pop[i].copy()
                        pop[i] = pop[i].copy()  # SA accepted, update pop
                
            # Cooling the temperature
            temp *= self.cooling_rate
            
            global_best_index = np.argmin(personal_best_fitness)
            global_best_position = personal_best_positions[global_best_index].copy()
            global_best_fitness = personal_best_fitness[global_best_index]

        return global_best_fitness, global_best_position