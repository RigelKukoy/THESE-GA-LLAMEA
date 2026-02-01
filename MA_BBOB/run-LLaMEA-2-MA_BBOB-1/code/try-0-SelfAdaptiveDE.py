import numpy as np

class SelfAdaptiveDE:
    def __init__(self, budget=10000, dim=10, initial_pop_size=50, F=0.5, CR=0.7, min_pop_size=10, max_pop_size=100):
        self.budget = budget
        self.dim = dim
        self.pop_size = initial_pop_size
        self.initial_pop_size = initial_pop_size
        self.F = F  # Initial mutation factor
        self.CR = CR  # Initial crossover rate
        self.min_pop_size = min_pop_size
        self.max_pop_size = max_pop_size
        self.success_history = []
        self.restart_counter = 0
        self.restart_threshold = 50

    def __call__(self, func):
        # Initialization
        population = np.random.uniform(func.bounds.lb, func.bounds.ub, size=(self.pop_size, self.dim))
        fitness = np.array([func(x) for x in population])
        self.budget -= self.pop_size

        best_index = np.argmin(fitness)
        self.f_opt = fitness[best_index]
        self.x_opt = population[best_index]

        # Evolution loop
        while self.budget > 0:
            new_population = np.zeros_like(population)
            new_fitness = np.zeros_like(fitness)

            for i in range(self.pop_size):
                # Mutation
                indices = np.random.choice(self.pop_size, 3, replace=False)
                x_r1, x_r2, x_r3 = population[indices]

                v = x_r1 + self.F * (x_r2 - x_r3)
                v = np.clip(v, func.bounds.lb, func.bounds.ub)

                # Crossover
                j_rand = np.random.randint(self.dim)
                u = np.copy(population[i])
                for j in range(self.dim):
                    if np.random.rand() < self.CR or j == j_rand:
                        u[j] = v[j]

                # Evaluation
                f_u = func(u)
                self.budget -= 1

                if f_u < fitness[i]:
                    new_fitness[i] = f_u
                    new_population[i] = u
                    self.success_history.append(1)

                else:
                    new_fitness[i] = fitness[i]
                    new_population[i] = population[i]
                    self.success_history.append(0)

                # Update best solution
                if f_u < self.f_opt:
                    self.f_opt = f_u
                    self.x_opt = u.copy()

                if self.budget <= 0:
                    return self.f_opt, self.x_opt

            population = new_population
            fitness = new_fitness

            # Adjust population size based on success rate
            success_rate = np.mean(self.success_history[-min(len(self.success_history), self.pop_size):]) if self.success_history else 0.5
            
            if success_rate > 0.25:
                self.pop_size = min(int(self.pop_size * 1.1), self.max_pop_size)  # Increase pop size
            elif success_rate < 0.15:
                self.pop_size = max(int(self.pop_size * 0.9), self.min_pop_size)  # Decrease pop size
            
            # Restart mechanism: if no improvement after a certain number of iterations, reinitialize
            self.restart_counter += 1
            if self.restart_counter > self.restart_threshold:
                self.restart_counter = 0
                # Reinitialize population (only a portion)
                num_reinitialize = int(self.pop_size * 0.2)
                indices_to_reinitialize = np.random.choice(self.pop_size, num_reinitialize, replace=False)
                population[indices_to_reinitialize] = np.random.uniform(func.bounds.lb, func.bounds.ub, size=(num_reinitialize, self.dim))
                fitness[indices_to_reinitialize] = [func(x) for x in population[indices_to_reinitialize]]
                self.budget -= num_reinitialize
                
                best_index = np.argmin(fitness)
                self.f_opt = fitness[best_index]
                self.x_opt = population[best_index]
            

        return self.f_opt, self.x_opt