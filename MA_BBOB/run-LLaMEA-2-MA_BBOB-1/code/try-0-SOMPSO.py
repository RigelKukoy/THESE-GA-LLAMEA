import numpy as np
from scipy.stats import levy

class SOMPSO:
    def __init__(self, budget=10000, dim=10, pop_size=20, w=0.7, c1=1.5, c2=1.5, levy_scale=0.01, som_grid_size=5, som_learning_rate=0.1, som_sigma=1.0):
        """
        Initialize the SOM-PSO algorithm.

        Args:
            budget (int): Total number of function evaluations.
            dim (int): Dimensionality of the problem.
            pop_size (int): Population size.
            w (float): Inertia weight for PSO.
            c1 (float): Cognitive coefficient for PSO.
            c2 (float): Social coefficient for PSO.
            levy_scale (float): Scale parameter for the Lévy flight.
            som_grid_size (int): Size of the SOM grid (som_grid_size x som_grid_size).
            som_learning_rate (float): Learning rate for the SOM.
            som_sigma (float): Initial sigma for the SOM neighborhood function.
        """
        self.budget = budget
        self.dim = dim
        self.pop_size = pop_size
        self.w = w
        self.c1 = c1
        self.c2 = c2
        self.levy_scale = levy_scale
        self.som_grid_size = som_grid_size
        self.som_learning_rate = som_learning_rate
        self.som_sigma = som_sigma
        self.pop = None
        self.fitness = None
        self.velocities = None
        self.x_best_global = None
        self.f_best_global = np.inf
        self.lb = None
        self.ub = None
        self.v_max = None
        self.som = None  # Self-Organizing Map

    def initialize_population(self, func):
        """
        Initialize the population within the bounds of the function.

        Args:
            func: The black-box optimization function.
        """
        self.lb = func.bounds.lb
        self.ub = func.bounds.ub
        self.pop = np.random.uniform(self.lb, self.ub, size=(self.pop_size, self.dim))
        self.fitness = np.array([func(x) for x in self.pop])
        self.velocities = np.zeros((self.pop_size, self.dim))
        self.x_best_global = self.pop[np.argmin(self.fitness)].copy()
        self.f_best_global = np.min(self.fitness)
        self.v_max = 0.2 * (self.ub - self.lb)
        self.initialize_som()


    def initialize_som(self):
        """
        Initialize the Self-Organizing Map.
        """
        self.som = np.random.uniform(self.lb, self.ub, size=(self.som_grid_size, self.som_grid_size, self.dim))

    def levy_flight(self):
        """
        Generate a Lévy flight step.
        """
        return self.levy_scale * levy.rvs(0.5, loc=0, scale=1, size=self.dim)

    def find_closest_neuron(self, x):
        """
        Find the closest neuron in the SOM to a given particle.

        Args:
            x (numpy.ndarray): Particle's position.

        Returns:
            tuple: Coordinates of the closest neuron (row, column).
        """
        distances = np.sum((self.som - x)**2, axis=2)
        row, col = np.unravel_index(np.argmin(distances), distances.shape)
        return row, col
    
    def update_som(self, x, row, col):
        """
        Update the SOM based on the particle's position.

        Args:
            x (numpy.ndarray): Particle's position.
            row (int): Row index of the closest neuron.
            col (int): Column index of the closest neuron.
        """
        for i in range(self.som_grid_size):
            for j in range(self.som_grid_size):
                distance = np.sqrt((i - row)**2 + (j - col)**2)
                influence = np.exp(-distance**2 / (2 * self.som_sigma**2))
                self.som[i, j] += self.som_learning_rate * influence * (x - self.som[i, j])

    def pso_update(self, func):
        """
        Update the population using PSO principles with Lévy flight and SOM integration.

        Args:
            func: The black-box optimization function.
        """
        for i in range(self.pop_size):
            # Update velocities
            r1 = np.random.rand(self.dim)
            r2 = np.random.rand(self.dim)
            self.velocities[i] = self.w * self.velocities[i] + \
                                 self.c1 * r1 * (self.pop[i] - self.pop[i]) + \
                                 self.c2 * r2 * (self.x_best_global - self.pop[i])

            # Velocity clamping
            self.velocities[i] = np.clip(self.velocities[i], -self.v_max, self.v_max)

            # Update positions
            new_position = self.pop[i] + self.velocities[i] + self.levy_flight()

            # Boundary handling (clip to bounds)
            new_position = np.clip(new_position, self.lb, self.ub)

            new_fitness = func(new_position)  # Evaluate new position

            if new_fitness < self.fitness[i]:
                self.pop[i] = new_position
                self.fitness[i] = new_fitness
                if new_fitness < self.f_best_global:
                    self.f_best_global = new_fitness
                    self.x_best_global = self.pop[i].copy()

            # SOM update
            row, col = self.find_closest_neuron(self.pop[i])
            self.update_som(self.pop[i], row, col)

    def __call__(self, func):
        """
        Optimize the given function using the SOM-PSO algorithm.

        Args:
            func: The black-box optimization function.

        Returns:
            tuple: The best function value found and the corresponding solution.
        """
        self.initialize_population(func)
        eval_count = self.pop_size  # Account for initial population evaluation

        while eval_count < self.budget:
            # Adaptive parameter adjustment (example: linear decrease of inertia weight)
            self.w = 0.7 - 0.5 * (eval_count / self.budget)
            self.som_learning_rate = 0.1 * (1 - (eval_count / self.budget)) # Reduce SOM learning rate over time
            self.som_sigma = max(0.1, 1.0 * (1 - (eval_count / self.budget))) # reduce SOM sigma over time
            
            self.pso_update(func)
            eval_count += self.pop_size  # Account for population evaluation

            if eval_count > self.budget:
                break

        return self.f_best_global, self.x_best_global