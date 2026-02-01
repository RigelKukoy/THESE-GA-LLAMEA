import numpy as np

class SelfOrganizingScoutBee:
    def __init__(self, budget=10000, dim=10, n_scouts=10, initial_step_size=0.5, min_step_size=0.01, neighborhood_size=0.1):
        self.budget = budget
        self.dim = dim
        self.n_scouts = n_scouts
        self.step_size = initial_step_size
        self.min_step_size = min_step_size
        self.neighborhood_size = neighborhood_size
        self.x_opt = None
        self.f_opt = np.Inf

    def __call__(self, func):
        # Initialize scout bees randomly
        scout_positions = np.random.uniform(func.bounds.lb, func.bounds.ub, size=(self.n_scouts, self.dim))
        scout_fitnesses = np.array([func(pos) for pos in scout_positions])
        self.budget -= self.n_scouts

        # Update global best
        best_index = np.argmin(scout_fitnesses)
        if scout_fitnesses[best_index] < self.f_opt:
            self.f_opt = scout_fitnesses[best_index]
            self.x_opt = scout_positions[best_index]

        while self.budget > 0:
            # Exploration phase: Each scout searches its neighborhood
            for i in range(self.n_scouts):
                # Generate a new position within the neighborhood
                new_position = scout_positions[i] + np.random.uniform(-self.step_size, self.step_size, size=self.dim)
                new_position = np.clip(new_position, func.bounds.lb, func.bounds.ub)

                # Evaluate the new position
                new_fitness = func(new_position)
                self.budget -= 1
                if self.budget <= 0:
                    break

                # Update scout position if better
                if new_fitness < scout_fitnesses[i]:
                    scout_fitnesses[i] = new_fitness
                    scout_positions[i] = new_position

                    # Update global best if necessary
                    if new_fitness < self.f_opt:
                        self.f_opt = new_fitness
                        self.x_opt = new_position
            if self.budget <= 0:
                break

            # Self-Organization: Adjust step size and neighborhood based on performance
            fitness_improvement = self.f_opt - np.min(scout_fitnesses)

            if fitness_improvement > 0:
                # Reduce step size if progress is being made
                self.step_size *= 0.9
                self.step_size = max(self.step_size, self.min_step_size)
            else:
                # Increase step size if no progress
                self.step_size *= 1.1
                # Ensure step size doesn't become too large (avoid unbounded expansion)
                self.step_size = min(self.step_size, (func.bounds.ub - func.bounds.lb) / 20.0) #max step size 0.5

            #Adaptive Neighborhood size
            self.neighborhood_size = 0.1 + 0.9 * (self.step_size / ((func.bounds.ub - func.bounds.lb)/2))  #maps step size to neighborhood

            # Scout bee relocation: Replace worst performing scouts with new random scouts
            worst_index = np.argmax(scout_fitnesses)
            if np.random.rand() < 0.1: #10% probability
                scout_positions[worst_index] = np.random.uniform(func.bounds.lb, func.bounds.ub, size=self.dim)
                scout_fitnesses[worst_index] = func(scout_positions[worst_index])
                self.budget -= 1
                if self.budget <= 0:
                    break

                if scout_fitnesses[worst_index] < self.f_opt:
                    self.f_opt = scout_fitnesses[worst_index]
                    self.x_opt = scout_positions[worst_index]


        return self.f_opt, self.x_opt