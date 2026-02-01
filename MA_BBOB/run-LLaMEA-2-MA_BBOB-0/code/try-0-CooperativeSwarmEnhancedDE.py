import numpy as np

class CooperativeSwarmEnhancedDE:
    def __init__(self, budget=10000, dim=10, pop_size=50, num_swarms=3, w=0.7, c1=1.5, c2=1.5, F=0.5, CR=0.9):
        self.budget = budget
        self.dim = dim
        self.pop_size = pop_size
        self.num_swarms = num_swarms
        self.w = w  # Inertia weight for PSO
        self.c1 = c1 # Cognitive coefficient for PSO
        self.c2 = c2 # Social coefficient for PSO
        self.F = F    # Mutation factor for DE
        self.CR = CR   # Crossover rate for DE
        self.swarms = []
        self.swarm_best_fitness = []
        self.swarm_best_positions = []
        self.global_best_fitness = np.Inf
        self.global_best_position = None

        for _ in range(self.num_swarms):
            self.swarms.append(np.random.uniform(-5, 5, size=(self.pop_size, self.dim)))
            self.swarm_best_fitness.append(np.inf * np.ones(self.pop_size))
            self.swarm_best_positions.append(np.zeros((self.pop_size, self.dim)))

    def __call__(self, func):

        # Initialize velocities for each swarm
        velocities = [np.zeros_like(swarm) for swarm in self.swarms]

        # Evaluate initial population
        for i in range(self.num_swarms):
            for j in range(self.pop_size):
                fitness = func(self.swarms[i][j])
                self.budget -= 1
                if fitness < self.swarm_best_fitness[i][j]:
                    self.swarm_best_fitness[i][j] = fitness
                    self.swarm_best_positions[i][j] = self.swarms[i][j].copy()
                if fitness < self.global_best_fitness:
                    self.global_best_fitness = fitness
                    self.global_best_position = self.swarms[i][j].copy()

                if self.budget <= 0:
                    return self.global_best_fitness, self.global_best_position

        while self.budget > 0:
            for i in range(self.num_swarms):
                # Perform PSO update
                r1 = np.random.rand(self.pop_size, self.dim)
                r2 = np.random.rand(self.pop_size, self.dim)
                velocities[i] = (self.w * velocities[i] +
                                 self.c1 * r1 * (self.swarm_best_positions[i] - self.swarms[i]) +
                                 self.c2 * r2 * (np.tile(self.global_best_position, (self.pop_size, 1)) - self.swarms[i]))
                self.swarms[i] += velocities[i]
                self.swarms[i] = np.clip(self.swarms[i], -5, 5)


                # Perform DE mutation
                for j in range(self.pop_size):
                    idxs = np.random.choice(self.pop_size, 3, replace=False)
                    x_1, x_2, x_3 = self.swarms[i][idxs]
                    mutant = x_1 + self.F * (x_2 - x_3)
                    mutant = np.clip(mutant, -5, 5)

                    # Perform DE crossover
                    crossover = np.random.uniform(size=self.dim) < self.CR
                    trial = np.where(crossover, mutant, self.swarms[i][j])
                    trial = np.clip(trial, -5, 5)

                    # Evaluate trial vector
                    fitness = func(trial)
                    self.budget -= 1
                    if fitness < self.swarm_best_fitness[i][j]:
                        self.swarm_best_fitness[i][j] = fitness
                        self.swarm_best_positions[i][j] = trial.copy()
                    if fitness < self.global_best_fitness:
                        self.global_best_fitness = fitness
                        self.global_best_position = trial.copy()
                    
                    # Selection: replace population[i] with trial only if trial is better
                    if fitness < func(self.swarms[i][j]): # Budget already decremented!
                       self.swarms[i][j] = trial.copy() # Replace with trial vector

                    if self.budget <= 0:
                        return self.global_best_fitness, self.global_best_position
            
            # Swarm information exchange (periodically)
            if (self.budget % (self.pop_size * self.num_swarms)) < self.pop_size : # Example: every generation
                # Sort swarms by best fitness
                sorted_swarms_indices = np.argsort([np.min(fitness) for fitness in self.swarm_best_fitness])
                best_swarm_idx = sorted_swarms_indices[0]
                worst_swarm_idx = sorted_swarms_indices[-1]

                # Replace a portion of the worst swarm with the best swarm
                num_to_replace = int(0.2 * self.pop_size)
                worst_idxs = np.argsort(self.swarm_best_fitness[worst_swarm_idx])[::-1][:num_to_replace] # Worst fitness in worst swarm
                best_idxs = np.argsort(self.swarm_best_fitness[best_swarm_idx])[:num_to_replace] # Best fitness in best swarm

                self.swarms[worst_swarm_idx][worst_idxs] = self.swarms[best_swarm_idx][best_idxs].copy()  #Copy best positions from best swarm to worst swarm
                self.swarm_best_fitness[worst_swarm_idx][worst_idxs] = self.swarm_best_fitness[best_swarm_idx][best_idxs].copy()
                self.swarm_best_positions[worst_swarm_idx][worst_idxs] = self.swarm_best_positions[best_swarm_idx][best_idxs].copy()
        
        return self.global_best_fitness, self.global_best_position