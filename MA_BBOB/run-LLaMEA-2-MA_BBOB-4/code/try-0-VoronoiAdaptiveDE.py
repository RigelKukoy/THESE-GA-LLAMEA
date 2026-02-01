import numpy as np
from scipy.spatial import Voronoi, voronoi_plot_2d

class VoronoiAdaptiveDE:
    def __init__(self, budget=10000, dim=10, pop_size=40, Cr=0.9, F=0.5, voronoi_refresh_rate=50):
        self.budget = budget
        self.dim = dim
        self.pop_size = pop_size
        self.Cr = Cr
        self.F = F
        self.voronoi_refresh_rate = voronoi_refresh_rate
        self.population = None
        self.fitness = None
        self.f_opt = np.Inf
        self.x_opt = None
        self.generation = 0
        self.voronoi = None
        self.regions = None
        self.vertices = None
        self.point_region = None

    def __call__(self, func):
        # Initialization
        self.population = np.random.uniform(func.bounds.lb, func.bounds.ub, size=(self.pop_size, self.dim))
        self.fitness = np.array([func(x) for x in self.population])
        self.budget -= self.pop_size

        self.f_opt = np.min(self.fitness)
        self.x_opt = self.population[np.argmin(self.fitness)]
        
        while self.budget > self.pop_size:
            if self.generation % self.voronoi_refresh_rate == 0:
                try:
                  self.voronoi = Voronoi(self.population)
                  self.regions = self.voronoi.regions
                  self.vertices = self.voronoi.vertices
                  self.point_region = self.voronoi.point_region
                except Exception as e:
                  # Handle cases where Voronoi computation fails (e.g., due to identical points)
                  # Fallback strategy: small random perturbation
                  self.population += np.random.normal(0, 1e-6, size=self.population.shape)
                  try:
                    self.voronoi = Voronoi(self.population)
                    self.regions = self.voronoi.regions
                    self.vertices = self.voronoi.vertices
                    self.point_region = self.voronoi.point_region
                  except:
                    pass

            new_population = np.copy(self.population)
            for i in range(self.pop_size):
                # Mutation
                indices = [j for j in range(self.pop_size) if j != i]
                idxs = np.random.choice(indices, size=3, replace=False)
                x_r1, x_r2, x_r3 = self.population[idxs[0]], self.population[idxs[1]], self.population[idxs[2]]

                # Voronoi-based F adaptation
                if self.voronoi is not None and self.point_region[i] != -1 and self.regions[self.point_region[i]] and all(v >= 0 for v in self.regions[self.point_region[i]]):
                    region_index = self.point_region[i]
                    num_vertices = len(self.regions[region_index]) #Density of voronoi cell
                    F = 0.1 + 0.9 * np.exp(-num_vertices/10) #Denser Voronoi cell, smaller F
                else:
                    F = self.F
                
                mutant = self.population[i] + F * (x_r1 - x_r2)

                # Crossover
                for j in range(self.dim):
                    if np.random.rand() < self.Cr:
                        new_population[i, j] = mutant[j]
                    else:
                        new_population[i, j] = self.population[i, j]

                new_population[i] = np.clip(new_population[i], func.bounds.lb, func.bounds.ub)

            # Evaluation
            new_fitness = np.array([func(x) for x in new_population])
            self.budget -= self.pop_size

            # Selection
            for i in range(self.pop_size):
                if new_fitness[i] < self.fitness[i]:
                    self.population[i] = new_population[i]
                    self.fitness[i] = new_fitness[i]

                    if new_fitness[i] < self.f_opt:
                        self.f_opt = new_fitness[i]
                        self.x_opt = new_population[i]

            self.generation += 1

        return self.f_opt, self.x_opt