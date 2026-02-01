import numpy as np
from sklearn.cluster import MiniBatchKMeans

class RepulsiveClusteringDE:
    def __init__(self, budget=10000, dim=10, pop_size=40, Cr=0.9, F=0.5, num_clusters=5, repulsion_strength=0.1):
        self.budget = budget
        self.dim = dim
        self.pop_size = pop_size
        self.Cr = Cr
        self.F = F
        self.num_clusters = num_clusters
        self.repulsion_strength = repulsion_strength
        self.f_opt = np.inf
        self.x_opt = None

    def __call__(self, func):
        # Initialization
        population = np.random.uniform(func.bounds.lb, func.bounds.ub, size=(self.pop_size, self.dim))
        fitness = np.array([func(x) for x in population])
        self.budget -= self.pop_size

        self.f_opt = np.min(fitness)
        self.x_opt = population[np.argmin(fitness)].copy()
        
        while self.budget > self.pop_size:
            new_population = np.copy(population)
            
            # Clustering
            kmeans = MiniBatchKMeans(n_clusters=self.num_clusters, random_state=0, n_init=5).fit(population)
            clusters = [[] for _ in range(self.num_clusters)]
            for i in range(self.pop_size):
                clusters[kmeans.labels_[i]].append(i)

            # Find worst individual
            worst_index = np.argmax(fitness)

            for i in range(self.pop_size):
                # Mutation with repulsive force from the worst
                idxs = np.random.choice(self.pop_size, 3, replace=False)
                mutant = population[idxs[0]] + self.F * (population[idxs[1]] - population[idxs[2]]) + self.repulsion_strength * (population[i] - population[worst_index])
                mutant = np.clip(mutant, func.bounds.lb, func.bounds.ub)
                
                # Crossover
                for j in range(self.dim):
                    if np.random.rand() < self.Cr:
                        new_population[i, j] = mutant[j]
                    
            # Evaluation
            new_fitness = np.array([func(x) for x in new_population])
            self.budget -= self.pop_size
            
            # Selection
            for i in range(self.pop_size):
                if new_fitness[i] < fitness[i]:
                    population[i] = new_population[i].copy()
                    fitness[i] = new_fitness[i]
                    
                    if new_fitness[i] < self.f_opt:
                        self.f_opt = new_fitness[i]
                        self.x_opt = new_population[i].copy()

        return self.f_opt, self.x_opt