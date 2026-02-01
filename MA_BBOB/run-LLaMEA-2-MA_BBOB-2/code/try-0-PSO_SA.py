import numpy as np

class PSO_SA:
    def __init__(self, budget=10000, dim=10, pop_size=30, inertia=0.7, c1=1.5, c2=1.5, temp_init=100.0, temp_decay=0.95):
        self.budget = budget
        self.dim = dim
        self.pop_size = pop_size
        self.inertia = inertia
        self.c1 = c1
        self.c2 = c2
        self.temp_init = temp_init
        self.temp_decay = temp_decay

    def __call__(self, func):
        # Initialize particles and velocities
        particles = np.random.uniform(func.bounds.lb, func.bounds.ub, size=(self.pop_size, self.dim))
        velocities = np.random.uniform(-1, 1, size=(self.pop_size, self.dim))  # Initialize velocities

        # Initialize personal best positions and values
        pbest_positions = particles.copy()
        pbest_values = np.array([func(x) for x in particles])
        self.budget -= self.pop_size

        # Initialize global best position and value
        gbest_index = np.argmin(pbest_values)
        gbest_position = pbest_positions[gbest_index].copy()
        gbest_value = pbest_values[gbest_index]

        # Simulated Annealing parameters
        temperature = self.temp_init

        while self.budget > 0:
            for i in range(self.pop_size):
                # Update velocity
                r1 = np.random.rand(self.dim)
                r2 = np.random.rand(self.dim)
                velocities[i] = (self.inertia * velocities[i] +
                                 self.c1 * r1 * (pbest_positions[i] - particles[i]) +
                                 self.c2 * r2 * (gbest_position - particles[i]))

                # Update particle position
                particles[i] += velocities[i]
                particles[i] = np.clip(particles[i], func.bounds.lb, func.bounds.ub)  # Clip to bounds

                # Evaluate new position
                current_value = func(particles[i])
                self.budget -= 1
                if self.budget <= 0:
                    break

                # Update personal best
                if current_value < pbest_values[i]:
                    pbest_values[i] = current_value
                    pbest_positions[i] = particles[i].copy()

                    # Update global best
                    if current_value < gbest_value:
                        gbest_value = current_value
                        gbest_position = particles[i].copy()

                else:
                    # Simulated Annealing acceptance criterion
                    delta = current_value - pbest_values[i]
                    if delta > 0 and np.random.rand() < np.exp(-delta / temperature):
                        pbest_values[i] = current_value
                        pbest_positions[i] = particles[i].copy()
            
            if self.budget <= 0:
                break


            # Cool down the temperature
            temperature *= self.temp_decay

        self.f_opt = gbest_value
        self.x_opt = gbest_position
        return self.f_opt, self.x_opt