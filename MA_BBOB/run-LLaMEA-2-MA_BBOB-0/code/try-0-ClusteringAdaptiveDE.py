import numpy as np
from sklearn.cluster import KMeans

class ClusteringAdaptiveDE:
    def __init__(self, budget=10000, dim=10, pop_size=50, num_clusters=5, lr_F=0.1, lr_CR=0.1, F_base=0.5, CR_base=0.9):
        self.budget = budget
        self.dim = dim
        self.pop_size = pop_size
        self.num_clusters = num_clusters
        self.lr_F = lr_F
        self.lr_CR = lr_CR
        self.F = np.full(pop_size, F_base)
        self.CR = np.full(pop_size, CR_base)
        self.F_base = F_base
        self.CR_base = CR_base

    def __call__(self, func):
        self.f_opt = np.Inf
        self.x_opt = None
        self.population = np.random.uniform(func.bounds.lb, func.bounds.ub, size=(self.pop_size, self.dim))
        fitness = np.array([func(x) for x in self.population])
        self.budget -= self.pop_size
        
        for i in range(self.pop_size):
            if fitness[i] < self.f_opt:
                self.f_opt = fitness[i]
                self.x_opt = self.population[i]

        while self.budget > 0:
            # Clustering
            kmeans = KMeans(n_clusters=self.num_clusters, random_state=0, n_init="auto").fit(self.population)
            clusters = [[] for _ in range(self.num_clusters)]
            for i in range(self.pop_size):
                clusters[kmeans.labels_[i]].append(i)

            for cluster_id in range(self.num_clusters):
                cluster_indices = clusters[cluster_id]
                if not cluster_indices:
                    continue

                # Differential Evolution within cluster
                for i in cluster_indices:
                    # Mutation
                    idxs = np.random.choice(cluster_indices, 3, replace=False)
                    x_r1, x_r2, x_r3 = self.population[idxs]
                    mutant = self.population[i] + self.F[i] * (x_r1 - x_r2) + self.F[i] * (x_r3 - self.population[i])
                    mutant = np.clip(mutant, func.bounds.lb, func.bounds.ub)
                
                    # Crossover
                    crossover = np.random.uniform(size=self.dim) < self.CR[i]
                    trial = np.where(crossover, mutant, self.population[i])
                
                    # Selection
                    f_trial = func(trial)
                    self.budget -= 1
                
                    if f_trial < self.f_opt:
                        self.f_opt = f_trial
                        self.x_opt = trial

                    if f_trial < fitness[i]:
                        # Adaptive Parameter Control
                        delta_f = fitness[i] - f_trial
                        self.F[i] = max(0, min(1, self.F[i] + self.lr_F * delta_f, 1.0))
                        self.CR[i] = max(0, min(1, self.CR[i] + self.lr_CR * delta_f, 1.0))
                    
                        fitness[i] = f_trial
                        self.population[i] = trial
                    else:
                        #If no improvement, revert to base values, encouraging exploration
                        self.F[i] = self.F_base
                        self.CR[i] = self.CR_base


        return self.f_opt, self.x_opt