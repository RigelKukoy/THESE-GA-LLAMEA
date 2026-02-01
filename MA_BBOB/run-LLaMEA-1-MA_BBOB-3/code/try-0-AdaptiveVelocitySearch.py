import numpy as np

class AdaptiveVelocitySearch:
    def __init__(self, budget=10000, dim=10, pop_size=20, inertia=0.7, cognitive_rate=1.5, social_rate=1.5, diversification_threshold=0.1):
        self.budget = budget
        self.dim = dim
        self.pop_size = pop_size
        self.inertia = inertia
        self.cognitive_rate = cognitive_rate
        self.social_rate = social_rate
        self.diversification_threshold = diversification_threshold
        self.positions = None
        self.velocities = None
        self.fitness = None
        self.best_position = None
        self.best_fitness = np.Inf
        self.eval_count = 0
        self.lb = None
        self.ub = None

    def initialize_population(self, func):
        self.lb = func.bounds.lb
        self.ub = func.bounds.ub
        self.positions = np.random.uniform(self.lb, self.ub, size=(self.pop_size, self.dim))
        self.velocities = np.random.uniform(-1, 1, size=(self.pop_size, self.dim))
        self.fitness = np.array([func(x) for x in self.positions])
        self.eval_count += self.pop_size
        self.best_position = self.positions[np.argmin(self.fitness)]
        self.best_fitness = np.min(self.fitness)

    def update_velocity(self, func):
        r1 = np.random.rand(self.pop_size, self.dim)
        r2 = np.random.rand(self.pop_size, self.dim)
        cognitive_component = self.cognitive_rate * r1 * (self.best_position - self.positions)
        social_component = self.social_rate * r2 * (self.positions[np.argmin(self.fitness)] - self.positions)
        self.velocities = self.inertia * self.velocities + cognitive_component + social_component
        
        # Velocity clamping based on remaining budget
        remaining_evals = self.budget - self.eval_count
        max_velocity = (self.ub[0] - self.lb[0]) / remaining_evals if remaining_evals > 0 else (self.ub[0] - self.lb[0])
        self.velocities = np.clip(self.velocities, -max_velocity, max_velocity)

        # Adaptive inertia based on population diversity
        diversity = self.calculate_diversity(func)
        if diversity < self.diversification_threshold:
          self.inertia = min(self.inertia + 0.05, 0.9)  # Increase inertia to explore more
        else:
          self.inertia = max(self.inertia - 0.05, 0.4)  # Decrease inertia to exploit more


    def update_position(self, func):
        self.positions = self.positions + self.velocities
        # Boundary handling
        self.positions = np.clip(self.positions, self.lb, self.ub)

        new_fitness = np.array([func(x) for x in self.positions])
        self.eval_count += self.pop_size

        for i in range(self.pop_size):
            if new_fitness[i] < self.fitness[i]:
                self.fitness[i] = new_fitness[i]
                if new_fitness[i] < self.best_fitness:
                    self.best_fitness = new_fitness[i]
                    self.best_position = self.positions[i]

    def calculate_diversity(self, func):
        # Calculate the average distance of each particle from the population mean
        mean_position = np.mean(self.positions, axis=0)
        distances = np.linalg.norm(self.positions - mean_position, axis=1)
        diversity = np.mean(distances) / (self.ub[0] - self.lb[0])  # Normalize by search space range
        return diversity
    
    def __call__(self, func):
        self.initialize_population(func)
        while self.eval_count < self.budget:
            self.update_velocity(func)
            self.update_position(func)
            
            # Check budget again in case func calls in update_position overran the budget slightly.
            if self.eval_count >= self.budget:
                break
        return self.best_fitness, self.best_position