import numpy as np
import copy

class PSO_SA:
    def __init__(self, budget=10000, dim=10, pop_size=20, inertia=0.7, cognitive_coeff=1.4, social_coeff=1.4, initial_temp=1.0, cooling_rate=0.95):
        self.budget = budget
        self.dim = dim
        self.pop_size = pop_size
        self.inertia = inertia
        self.cognitive_coeff = cognitive_coeff
        self.social_coeff = social_coeff
        self.initial_temp = initial_temp
        self.cooling_rate = cooling_rate

    def __call__(self, func):
        # Initialize population
        pop = np.random.uniform(func.bounds.lb, func.bounds.ub, size=(self.pop_size, self.dim))
        fitness = np.array([func(x) for x in pop])
        self.budget -= self.pop_size

        # Initialize velocities
        velocities = np.random.uniform(-1, 1, size=(self.pop_size, self.dim))

        # Initialize personal best positions and fitness
        personal_best_positions = pop.copy()
        personal_best_fitness = fitness.copy()

        # Initialize global best position and fitness
        global_best_index = np.argmin(fitness)
        global_best_position = pop[global_best_index].copy()
        global_best_fitness = fitness[global_best_index]

        # Simulated Annealing parameters
        temperature = self.initial_temp

        while self.budget > 0:
            for i in range(self.pop_size):
                # PSO update
                velocities[i] = (self.inertia * velocities[i] +
                                 self.cognitive_coeff * np.random.rand() * (personal_best_positions[i] - pop[i]) +
                                 self.social_coeff * np.random.rand() * (global_best_position - pop[i]))

                pop[i] += velocities[i]
                pop[i] = np.clip(pop[i], func.bounds.lb, func.bounds.ub)

                # Evaluate new position
                f = func(pop[i])
                self.budget -= 1

                # Simulated Annealing acceptance criterion
                delta_e = f - personal_best_fitness[i]

                if delta_e < 0:
                    # Accept the new solution if it's better
                    personal_best_fitness[i] = f
                    personal_best_positions[i] = pop[i].copy()

                    if f < global_best_fitness:
                        global_best_fitness = f
                        global_best_position = pop[i].copy()
                else:
                    # Accept the new solution with a probability based on temperature
                    acceptance_prob = np.exp(-delta_e / temperature)
                    if np.random.rand() < acceptance_prob:
                        personal_best_fitness[i] = f
                        personal_best_positions[i] = pop[i].copy()

            # Cooling
            temperature *= self.cooling_rate

        return global_best_fitness, global_best_position