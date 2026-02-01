import numpy as np

class SelfOrganizingScoutBee:
    def __init__(self, budget=10000, dim=10, num_bees=50, scout_rate=0.1, step_size_init=0.5, step_size_decay=0.99, opposition_rate=0.05):
        self.budget = budget
        self.dim = dim
        self.num_bees = num_bees
        self.scout_rate = scout_rate
        self.step_size_init = step_size_init
        self.step_size_decay = step_size_decay
        self.opposition_rate = opposition_rate

    def __call__(self, func):
        # Initialize bee positions randomly
        bees = np.random.uniform(func.bounds.lb, func.bounds.ub, size=(self.num_bees, self.dim))
        fitness = np.array([func(bee) for bee in bees])
        self.budget -= self.num_bees

        # Find the best bee
        best_index = np.argmin(fitness)
        best_bee = bees[best_index].copy()
        best_fitness = fitness[best_index]

        step_size = self.step_size_init

        while self.budget > 0:
            for i in range(self.num_bees):
                # Scout bees: random exploration
                if np.random.rand() < self.scout_rate:
                    new_bee = np.random.uniform(func.bounds.lb, func.bounds.ub)
                    new_fitness = func(new_bee)
                    self.budget -= 1

                    if new_fitness < fitness[i]:
                        bees[i] = new_bee
                        fitness[i] = new_fitness

                        if new_fitness < best_fitness:
                            best_fitness = new_fitness
                            best_bee = new_bee.copy()

                # Employed bees: exploitation around the best bee
                else:
                    # Select a random dimension to modify
                    dim_index = np.random.randint(self.dim)

                    # Generate a random step in that dimension
                    step = np.random.uniform(-step_size, step_size)

                    # Create a new bee based on the current bee and the random step
                    new_bee = bees[i].copy()
                    new_bee[dim_index] += step
                    new_bee = np.clip(new_bee, func.bounds.lb, func.bounds.ub)

                    # Evaluate the new bee
                    new_fitness = func(new_bee)
                    self.budget -= 1
                    
                    if self.budget <= 0:
                        break

                    # Update if the new bee is better
                    if new_fitness < fitness[i]:
                        bees[i] = new_bee
                        fitness[i] = new_fitness
                        
                        if new_fitness < best_fitness:
                            best_fitness = new_fitness
                            best_bee = new_bee.copy()
                    
                    # Opposition based learning:
                    elif np.random.rand() < self.opposition_rate:
                        opposite_bee = func.bounds.ub + func.bounds.lb - new_bee
                        opposite_bee = np.clip(opposite_bee, func.bounds.lb, func.bounds.ub)
                        opposite_fitness = func(opposite_bee)
                        self.budget -=1
                        if self.budget <=0:
                            break
                        
                        if opposite_fitness < fitness[i]:
                            bees[i] = opposite_bee
                            fitness[i] = opposite_fitness

                            if opposite_fitness < best_fitness:
                                best_fitness = opposite_fitness
                                best_bee = opposite_bee.copy()
            
            if self.budget <= 0:
                break
            # Decay step size
            step_size *= self.step_size_decay

        self.f_opt = best_fitness
        self.x_opt = best_bee
        return self.f_opt, self.x_opt