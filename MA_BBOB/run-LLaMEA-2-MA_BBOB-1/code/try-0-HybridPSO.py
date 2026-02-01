import numpy as np

class HybridPSO:
    def __init__(self, budget=10000, dim=10, swarm_size=30, levy_flight_probability=0.1, cma_learning_rate=0.1):
        self.budget = budget
        self.dim = dim
        self.swarm_size = swarm_size
        self.levy_flight_probability = levy_flight_probability
        self.cma_learning_rate = cma_learning_rate
        self.swarm_positions = None
        self.swarm_velocities = None
        self.swarm_fitness = None
        self.personal_best_positions = None
        self.personal_best_fitness = None
        self.global_best_position = None
        self.global_best_fitness = np.inf
        self.covariance_matrix = None

    def initialize_swarm(self, func):
        self.swarm_positions = np.random.uniform(func.bounds.lb, func.bounds.ub, size=(self.swarm_size, self.dim))
        self.swarm_velocities = np.random.uniform(-1, 1, size=(self.swarm_size, self.dim)) * 0.1
        self.swarm_fitness = np.array([func(x) for x in self.swarm_positions])
        self.budget -= self.swarm_size
        self.personal_best_positions = self.swarm_positions.copy()
        self.personal_best_fitness = self.swarm_fitness.copy()

        best_index = np.argmin(self.swarm_fitness)
        self.global_best_position = self.swarm_positions[best_index].copy()
        self.global_best_fitness = self.swarm_fitness[best_index]
        self.covariance_matrix = np.eye(self.dim)  # Initialize covariance matrix

    def levy_flight(self, beta=1.5):
        sigma = (np.math.gamma(1 + beta) * np.sin(np.pi * beta / 2) / (np.math.gamma((1 + beta) / 2) * beta * (2 ** ((beta - 1) / 2)))) ** (1 / beta)
        u = np.random.normal(0, sigma, size=self.dim)
        v = np.random.normal(0, 1, size=self.dim)
        step = u / (abs(v) ** (1 / beta))
        return step

    def update_velocities(self, cognitive_coeff=1.5, social_coeff=1.5, inertia_weight=0.7):
        for i in range(self.swarm_size):
            r1 = np.random.rand(self.dim)
            r2 = np.random.rand(self.dim)

            self.swarm_velocities[i] = (inertia_weight * self.swarm_velocities[i]
                                       + cognitive_coeff * r1 * (self.personal_best_positions[i] - self.swarm_positions[i])
                                       + social_coeff * r2 * (self.global_best_position - self.swarm_positions[i]))

            # Lévy flight dispersal
            if np.random.rand() < self.levy_flight_probability:
                levy_step = self.levy_flight()
                self.swarm_velocities[i] += 0.01 * levy_step  # scale levy step
                

    def update_positions(self, func):
        self.swarm_positions += self.swarm_velocities
        self.swarm_positions = np.clip(self.swarm_positions, func.bounds.lb, func.bounds.ub)
        fitness = np.array([func(x) for x in self.swarm_positions])
        self.budget -= self.swarm_size

        for i in range(self.swarm_size):
            if fitness[i] < self.swarm_fitness[i]:
                self.swarm_fitness[i] = fitness[i]
                if fitness[i] < self.personal_best_fitness[i]:
                    self.personal_best_fitness[i] = fitness[i]
                    self.personal_best_positions[i] = self.swarm_positions[i].copy()
                    if fitness[i] < self.global_best_fitness:
                        self.global_best_fitness = fitness[i]
                        self.global_best_position = self.swarm_positions[i].copy()

    def update_covariance_matrix(self):
        # Simple CMA update: adjust covariance matrix based on successful steps
        diff = self.swarm_positions - self.global_best_position
        self.covariance_matrix = (1 - self.cma_learning_rate) * self.covariance_matrix + self.cma_learning_rate * np.cov(diff.T)
        #Ensure positive definiteness
        try:
            np.linalg.cholesky(self.covariance_matrix)
        except np.linalg.LinAlgError:
            self.covariance_matrix = np.eye(self.dim) #reset if not positive definite
            

    def __call__(self, func):
        self.initialize_swarm(func)

        while self.budget > 0:
            self.update_velocities()
            self.update_positions(func)
            self.update_covariance_matrix()  # Update covariance matrix

        return self.global_best_fitness, self.global_best_position