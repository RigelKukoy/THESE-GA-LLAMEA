import numpy as np

class AntColonyOptimizer:
    def __init__(self, budget=10000, dim=10, num_ants=20, evaporation_rate=0.1, pheromone_influence=1.0, random_influence=0.5):
        self.budget = budget
        self.dim = dim
        self.num_ants = num_ants
        self.evaporation_rate = evaporation_rate
        self.pheromone_influence = pheromone_influence
        self.random_influence = random_influence

    def __call__(self, func):
        lb = func.bounds.lb
        ub = func.bounds.ub
        
        # Initialize pheromone trail
        pheromone = np.ones(self.dim) * 0.001

        # Initialize best solution
        self.f_opt = np.Inf
        self.x_opt = None

        for i in range(self.budget // self.num_ants):  # Iterate until budget is exhausted
            
            ant_positions = np.zeros((self.num_ants, self.dim))
            ant_fitness = np.zeros(self.num_ants)
            
            #Ant colony exploration
            for ant in range(self.num_ants):
                position = np.random.uniform(lb, ub, self.dim)

                #Pheromone guided movement
                for d in range(self.dim):
                    probability = (self.pheromone_influence * pheromone[d] + self.random_influence)
                    if np.random.rand() < probability :
                      position[d] = np.random.uniform(lb, ub)
                    else:
                      #Move in direction of best solution, only slightly
                      if self.x_opt is not None:
                          position[d] += np.random.normal(0, 0.01) * (self.x_opt[d] - position[d])
                
                position = np.clip(position, lb, ub) #Bounds check

                ant_positions[ant] = position
                ant_fitness[ant] = func(position)
                
                if ant_fitness[ant] < self.f_opt:
                    self.f_opt = ant_fitness[ant]
                    self.x_opt = position

            # Update pheromone trails
            pheromone *= (1 - self.evaporation_rate)
            
            #Deposit pheromones proportional to the quality of the solutions
            for ant in range(self.num_ants):
                pheromone += (1 / (1 + ant_fitness[ant])) #Better solutions increase pheromone more

        return self.f_opt, self.x_opt