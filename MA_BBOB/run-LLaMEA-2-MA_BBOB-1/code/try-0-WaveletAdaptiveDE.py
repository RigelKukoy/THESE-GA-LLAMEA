import numpy as np
import pywt

class WaveletAdaptiveDE:
    def __init__(self, budget=10000, dim=10, pop_size=50, F=0.5, CR=0.7, wavelet='db4'):
        self.budget = budget
        self.dim = dim
        self.pop_size = pop_size
        self.F = F  # Initial mutation factor
        self.CR = CR  # Initial crossover rate
        self.wavelet = wavelet # Wavelet type
        self.archive_size = int(self.pop_size * 0.2)  # Archive size for storing successful solutions
        self.archive = []
        self.success_F = []
        self.success_CR = []
        self.memory_size = 10

    def wavelet_mutation(self, x_r1, x_r2, x_r3):
        """Applies wavelet transform to the difference vector."""
        diff = x_r2 - x_r3
        coeffs = pywt.wavedec(diff, self.wavelet, level=1)  # Decompose to level 1

        # Add noise to detail coefficients (high-frequency components)
        coeffs[1] += np.random.normal(0, 0.1, size=coeffs[1].shape)  # Small noise
        
        # Reconstruct the signal
        mutated_diff = pywt.waverec(coeffs, self.wavelet)
        
        # Ensure the mutated difference has the same dimension
        if len(mutated_diff) != self.dim:
            mutated_diff = np.resize(mutated_diff, self.dim)
        
        return x_r1 + self.F * mutated_diff

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
            for i in range(self.pop_size):
                # Mutation
                indices = np.random.choice(self.pop_size, 3, replace=False)
                x_r1, x_r2, x_r3 = population[indices]

                # Wavelet mutation
                v = self.wavelet_mutation(x_r1, x_r2, x_r3)
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
                    # Replacement
                    fitness[i] = f_u
                    population[i] = u

                    # Update archive (if necessary)
                    if len(self.archive) < self.archive_size:
                        self.archive.append(population[i])
                    else:
                        # Replace a random element in the archive
                        replace_index = np.random.randint(self.archive_size)
                        self.archive[replace_index] = population[i]

                    # Store successful F and CR values
                    self.success_F.append(self.F)
                    self.success_CR.append(self.CR)
                    if len(self.success_F) > self.memory_size:
                        self.success_F.pop(0)
                        self.success_CR.pop(0)
                    
                    # Adaptive F and CR: Adjust based on success history
                    if self.success_F:
                        self.F = np.mean(self.success_F)
                    
                    if self.success_CR:
                        self.CR = np.mean(self.success_CR)

                    self.F = self.F + np.random.normal(0, 0.05)
                    self.CR = self.CR + np.random.normal(0, 0.05)
                    self.F = np.clip(self.F, 0.1, 1.0)
                    self.CR = np.clip(self.CR, 0.1, 0.9)
                else:
                     # Reduce CR if the trial vector is not better
                     self.CR = self.CR - 0.05 # Decrease CR
                     self.CR = np.clip(self.CR, 0.1, 0.9)

                # Update best solution
                if f_u < self.f_opt:
                    self.f_opt = f_u
                    self.x_opt = u.copy()

        return self.f_opt, self.x_opt