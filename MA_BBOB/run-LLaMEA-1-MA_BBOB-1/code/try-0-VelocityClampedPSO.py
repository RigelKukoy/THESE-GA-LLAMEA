import numpy as np

class VelocityClampedPSO:
    def __init__(self, budget=10000, dim=10, pop_size=20, c1=1.49, c2=1.49, v_max_factor=0.2):
        self.budget = budget
        self.dim = dim
        self.pop_size = pop_size
        self.c1 = c1  # Cognitive coefficient
        self.c2 = c2  # Social coefficient
        self.v_max_factor = v_max_factor  # Max velocity as fraction of search space

    def __call__(self, func):
        lb = func.bounds.lb
        ub = func.bounds.ub
        
        # Initialize population and velocities
        population = np.random.uniform(lb, ub, size=(self.pop_size, self.dim))
        velocities = np.random.uniform(-abs(ub - lb) * self.v_max_factor, abs(ub - lb) * self.v_max_factor, size=(self.pop_size, self.dim))
        
        # Initialize personal best positions and fitnesses
        pbest_positions = population.copy()
        pbest_fitnesses = np.array([func(x) for x in population])
        
        # Initialize global best position and fitness
        global_best_index = np.argmin(pbest_fitnesses)
        global_best_position = pbest_positions[global_best_index].copy()
        global_best_fitness = pbest_fitnesses[global_best_index]
        
        eval_count = self.pop_size #initial evalutations
        
        # Iterate until budget is exhausted
        while eval_count < self.budget:
            # Update velocities and positions
            r1 = np.random.rand(self.pop_size, self.dim)
            r2 = np.random.rand(self.pop_size, self.dim)
            
            velocities = velocities + self.c1 * r1 * (pbest_positions - population) + self.c2 * r2 * (global_best_position - population)

            # Clamp velocities
            v_max = abs(ub - lb) * self.v_max_factor
            velocities = np.clip(velocities, -v_max, v_max)
            
            population = population + velocities
            
            # Reflect particles that hit boundaries
            for i in range(self.pop_size):
                for j in range(self.dim):
                    if population[i, j] < lb:
                        population[i, j] = lb + (lb - population[i, j])
                        velocities[i,j] = -velocities[i,j]
                    elif population[i, j] > ub:
                        population[i, j] = ub - (population[i, j] - ub)
                        velocities[i,j] = -velocities[i,j]

            # Evaluate new positions
            fitnesses = np.array([func(x) for x in population])
            eval_count += self.pop_size

            # Update personal best positions and fitnesses
            for i in range(self.pop_size):
                if fitnesses[i] < pbest_fitnesses[i]:
                    pbest_fitnesses[i] = fitnesses[i]
                    pbest_positions[i] = population[i].copy()
                    
                    # Update global best position and fitness
                    if fitnesses[i] < global_best_fitness:
                        global_best_fitness = fitnesses[i]
                        global_best_position = population[i].copy()
                        
        return global_best_fitness, global_best_position