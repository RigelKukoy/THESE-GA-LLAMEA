import numpy as np

class CooperativeSwarmOptimization:
    def __init__(self, budget=10000, dim=10, num_swarms=3, swarm_size=20):
        self.budget = budget
        self.dim = dim
        self.num_swarms = num_swarms
        self.swarm_size = swarm_size
        self.swarms = []
        self.swarm_positions = []
        self.swarm_velocities = []
        self.swarm_fitness = []
        self.swarm_best_positions = []
        self.swarm_best_fitness = []
        self.exploration_rates = np.ones(self.num_swarms) * 0.5 # Initial exploration rate for each swarm
        self.inertia_weights = np.ones(self.num_swarms) * 0.7
        self.cognitive_coeffs = np.ones(self.num_swarms) * 1.5
        self.social_coeffs = np.ones(self.num_swarms) * 1.5
        self.communication_probability = 0.1 #probability of inter-swarm communication


    def initialize_swarms(self, func):
        for i in range(self.num_swarms):
            positions = np.random.uniform(func.bounds.lb, func.bounds.ub, size=(self.swarm_size, self.dim))
            velocities = np.random.uniform(-1, 1, size=(self.swarm_size, self.dim)) * 0.1
            fitness = np.array([func(x) for x in positions])
            self.budget -= self.swarm_size
            best_indices = np.argmin(fitness)
            best_positions = positions[best_indices].copy()
            best_fitness = fitness[best_indices]
            self.swarms.append(i)
            self.swarm_positions.append(positions)
            self.swarm_velocities.append(velocities)
            self.swarm_fitness.append(fitness)
            self.swarm_best_positions.append(best_positions)
            self.swarm_best_fitness.append(best_fitness)

    def update_velocities(self, swarm_index, global_best_position):
        inertia_weight = self.inertia_weights[swarm_index]
        cognitive_coeff = self.cognitive_coeffs[swarm_index]
        social_coeff = self.social_coeffs[swarm_index]
        exploration_rate = self.exploration_rates[swarm_index]
        for i in range(self.swarm_size):
            r1 = np.random.rand(self.dim)
            r2 = np.random.rand(self.dim)
            self.swarm_velocities[swarm_index][i] = (
                inertia_weight * self.swarm_velocities[swarm_index][i]
                + cognitive_coeff * r1 * (self.swarm_best_positions[swarm_index] - self.swarm_positions[swarm_index][i])
                + social_coeff * r2 * (global_best_position - self.swarm_positions[swarm_index][i])
                + exploration_rate * np.random.uniform(-1, 1, size=self.dim) # Exploration term
            )

    def update_positions(self, swarm_index, func):
        self.swarm_positions[swarm_index] += self.swarm_velocities[swarm_index]
        self.swarm_positions[swarm_index] = np.clip(self.swarm_positions[swarm_index], func.bounds.lb, func.bounds.ub)
        fitness = np.array([func(x) for x in self.swarm_positions[swarm_index]])
        self.budget -= self.swarm_size
        
        for i in range(self.swarm_size):
            if fitness[i] < self.swarm_fitness[swarm_index][i]:
                self.swarm_fitness[swarm_index][i] = fitness[i]
                if fitness[i] < self.swarm_best_fitness[swarm_index]:
                    self.swarm_best_fitness[swarm_index] = fitness[i]
                    self.swarm_best_positions[swarm_index] = self.swarm_positions[swarm_index][i].copy()

    def adaptive_exploration(self, swarm_index):
         # Simple adjustment based on swarm's progress. Increase if stagnant, decrease if improving
        if np.std(self.swarm_fitness[swarm_index]) < 1e-3: #Stagnation
            self.exploration_rates[swarm_index] = min(1.0, self.exploration_rates[swarm_index] * 1.1)
        else:
            self.exploration_rates[swarm_index] = max(0.01, self.exploration_rates[swarm_index] * 0.9)

    def inter_swarm_communication(self):
        # Exchange information between swarms with a probability
        for i in range(self.num_swarms):
            if np.random.rand() < self.communication_probability:
                # Select another swarm to communicate with
                other_swarm = np.random.choice([s for s in range(self.num_swarms) if s != i])
                # Exchange best positions. The better swarm influences the other.
                if self.swarm_best_fitness[i] < self.swarm_best_fitness[other_swarm]:
                    self.swarm_best_positions[other_swarm] = self.swarm_best_positions[i].copy()
                    self.swarm_best_fitness[other_swarm] = self.swarm_best_fitness[i]
                else:
                     self.swarm_best_positions[i] = self.swarm_best_positions[other_swarm].copy()
                     self.swarm_best_fitness[i] = self.swarm_best_fitness[other_swarm]

    def __call__(self, func):
        # Initialization
        self.initialize_swarms(func)
        global_best_swarm = np.argmin(self.swarm_best_fitness)
        self.f_opt = self.swarm_best_fitness[global_best_swarm]
        self.x_opt = self.swarm_best_positions[global_best_swarm].copy()

        # Evolution loop
        while self.budget > 0:
            # Update velocities and positions for each swarm
            for i in range(self.num_swarms):
                self.update_velocities(i, self.x_opt)
                self.update_positions(i, func)
                self.adaptive_exploration(i)
            
            self.inter_swarm_communication()
            
            # Update global best
            global_best_swarm = np.argmin(self.swarm_best_fitness)
            if self.swarm_best_fitness[global_best_swarm] < self.f_opt:
                self.f_opt = self.swarm_best_fitness[global_best_swarm]
                self.x_opt = self.swarm_best_positions[global_best_swarm].copy()

        return self.f_opt, self.x_opt