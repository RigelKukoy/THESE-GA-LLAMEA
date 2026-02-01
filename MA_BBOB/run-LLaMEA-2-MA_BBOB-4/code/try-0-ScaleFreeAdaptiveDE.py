import numpy as np

class ScaleFreeAdaptiveDE:
    def __init__(self, budget=10000, dim=10, pop_size=40, Cr=0.9, F_initial=0.5, stagnation_threshold=100, restart_prob=0.1, sf_connectivity=3):
        self.budget = budget
        self.dim = dim
        self.pop_size = pop_size
        self.Cr = Cr
        self.F = F_initial
        self.stagnation_threshold = stagnation_threshold
        self.restart_prob = restart_prob
        self.sf_connectivity = sf_connectivity # Parameter for scale-free network
        self.best_fitness_history = []
        self.last_improvement = 0
        self.generation = 0
        self.network = self.create_scale_free_network()

    def create_scale_free_network(self):
        """Creates a scale-free network using the Barabási-Albert model."""
        # Initialize with a fully connected network of sf_connectivity nodes
        network = {i: list(range(self.sf_connectivity)) for i in range(self.sf_connectivity)}
        for i in range(self.sf_connectivity):
            network[i].remove(i)  # Remove self-loops
        
        for new_node in range(self.sf_connectivity, self.pop_size):
            # Connect new node to existing nodes with probability proportional to their degree
            degrees = [len(network[node]) for node in network]
            probabilities = np.array(degrees) / np.sum(degrees)
            
            # Choose nodes without replacement
            connections = np.random.choice(list(network.keys()), size=self.sf_connectivity, replace=False, p=probabilities)
            
            network[new_node] = list(connections)
            for connected_node in connections:
                network[connected_node].append(new_node)
                
        return network

    def __call__(self, func):
        # Initialization
        population = np.random.uniform(func.bounds.lb, func.bounds.ub, size=(self.pop_size, self.dim))
        fitness = np.array([func(x) for x in population])
        self.budget -= self.pop_size
        
        self.f_opt = np.min(fitness)
        self.x_opt = population[np.argmin(fitness)]
        self.best_fitness_history.append(self.f_opt)
        self.last_improvement = 0
        self.generation = 0

        while self.budget > self.pop_size:
            # Mutation and Crossover
            new_population = np.copy(population)
            for i in range(self.pop_size):
                # Scale-Free Network based Mutation
                neighbors = self.network[i]
                if len(neighbors) < 2:
                  # Handle edge cases where degree < 2 (rare, but possible)
                  indices = [j for j in range(self.pop_size) if j != i]
                  idxs = np.random.choice(indices, size=2, replace=False)
                  x_r1, x_r2 = population[idxs[0]], population[idxs[1]]
                  mutant = population[i] + self.F * (x_r1 - x_r2)
                else:
                  idxs = np.random.choice(neighbors, size=2, replace=False)
                  x_r1, x_r2 = population[idxs[0]], population[idxs[1]]
                  mutant = population[i] + self.F * (x_r1 - x_r2)

                # Crossover
                for j in range(self.dim):
                    if np.random.rand() < self.Cr:
                        new_population[i, j] = mutant[j]
                    else:
                        new_population[i, j] = population[i, j]
                
                new_population[i] = np.clip(new_population[i], func.bounds.lb, func.bounds.ub)
            
            # Evaluation
            new_fitness = np.array([func(x) for x in new_population])
            self.budget -= self.pop_size
            
            # Selection
            for i in range(self.pop_size):
                if new_fitness[i] < fitness[i]:
                    population[i] = new_population[i]
                    fitness[i] = new_fitness[i]
                    
                    if new_fitness[i] < self.f_opt:
                        self.f_opt = new_fitness[i]
                        self.x_opt = new_population[i]
                        self.last_improvement = self.generation
                        
            self.best_fitness_history.append(self.f_opt)
            
            # Stagnation check and restart
            if (self.generation - self.last_improvement) > self.stagnation_threshold:
                if np.random.rand() < self.restart_prob:
                    # Restart the population and network
                    population = np.random.uniform(func.bounds.lb, func.bounds.ub, size=(self.pop_size, self.dim))
                    fitness = np.array([func(x) for x in population])
                    self.budget -= self.pop_size
                    self.f_opt = np.min(fitness)
                    self.x_opt = population[np.argmin(fitness)]
                    self.last_improvement = self.generation
                    self.F = 0.5 #reset F
                    self.network = self.create_scale_free_network()  # Recreate the network
                else:
                    # Adaptive F: Reduce mutation strength upon stagnation
                    self.F *= 0.9  # Reduce F, but prevent it from becoming zero.
                    self.F = max(self.F, 0.1)

            self.generation += 1
        
        return self.f_opt, self.x_opt