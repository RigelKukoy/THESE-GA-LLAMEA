import numpy as np
import pywt

class WaveletMutationDE:
    def __init__(self, budget=10000, dim=10, pop_size=40, F=0.5, Cr=0.7, wavelet='db1', local_search_iterations=5):
        self.budget = budget
        self.dim = dim
        self.pop_size = pop_size
        self.F = F
        self.Cr = Cr
        self.wavelet = wavelet
        self.local_search_iterations = local_search_iterations

    def __call__(self, func):
        # Initialization
        population = np.random.uniform(func.bounds.lb, func.bounds.ub, size=(self.pop_size, self.dim))
        fitness = np.array([func(x) for x in population])
        self.budget -= self.pop_size
        
        self.f_opt = np.min(fitness)
        self.x_opt = population[np.argmin(fitness)].copy()
        
        while self.budget > self.pop_size:
            new_population = np.copy(population)
            for i in range(self.pop_size):
                # Mutation
                idxs = np.random.choice(self.pop_size, 3, replace=False)
                x_r1, x_r2, x_r3 = population[idxs]
                mutant = population[i] + self.F * (x_r2 - x_r3)

                # Wavelet Mutation
                coeffs = pywt.wavedec(mutant, self.wavelet, level=min(4, pywt.dwt_max_level(len(mutant), pywt.Wavelet(self.wavelet).dec_len)))
                for j in range(1, len(coeffs)):  # Skip the coarse approximation coefficients
                    coeffs[j] = np.random.normal(0, self.F * np.std(coeffs[j]), size=coeffs[j].shape)
                mutant = pywt.waverec(coeffs, self.wavelet)
                
                mutant = np.clip(mutant, func.bounds.lb, func.bounds.ub)
                
                # Crossover
                for j in range(self.dim):
                    if np.random.rand() < self.Cr:
                        new_population[i, j] = mutant[j]
                    else:
                        new_population[i, j] = population[i, j]

            # Evaluation
            new_fitness = np.array([func(x) for x in new_population])
            self.budget -= self.pop_size

            # Selection and Local Search
            for i in range(self.pop_size):
                if new_fitness[i] < fitness[i]:
                    population[i] = new_population[i]
                    fitness[i] = new_fitness[i]
                    if new_fitness[i] < self.f_opt:
                        self.f_opt = new_fitness[i]
                        self.x_opt = new_population[i].copy()
                else:
                    # Local search around the better solution
                    x_local = population[i].copy()
                    f_local = fitness[i]
                    for _ in range(self.local_search_iterations):
                        x_neighbor = x_local + np.random.normal(0, 0.05, size=self.dim)
                        x_neighbor = np.clip(x_neighbor, func.bounds.lb, func.bounds.ub)
                        f_neighbor = func(x_neighbor)
                        self.budget -= 1
                        if f_neighbor < f_local:
                            x_local = x_neighbor.copy()
                            f_local = f_neighbor
                            if f_local < self.f_opt:
                                self.f_opt = f_local
                                self.x_opt = x_local.copy()
                    population[i] = x_local
                    fitness[i] = f_local
        return self.f_opt, self.x_opt