import numpy as np
from minisom import MiniSom  # Requires: pip install MiniSom

class SOMGuidedDE:
    def __init__(self, budget=10000, dim=10, popsize=50, som_grid_size=10, F=0.5, CR=0.7):
        self.budget = budget
        self.dim = dim
        self.popsize = popsize
        self.som_grid_size = som_grid_size
        self.F = F
        self.CR = CR
        self.population = None
        self.fitness = None
        self.som = None
        self.best_fitness = np.inf
        self.best_solution = None

    def initialize_population(self, func):
        self.population = np.random.uniform(func.bounds.lb, func.bounds.ub, size=(self.popsize, self.dim))
        self.fitness = np.array([func(x) for x in self.population])
        self.best_fitness = np.min(self.fitness)
        self.best_solution = self.population[np.argmin(self.fitness)]

    def train_som(self):
        self.som = MiniSom(self.som_grid_size, self.som_grid_size, self.dim, sigma=0.3, learning_rate=0.5)
        self.som.train(self.population, 1000)  # Train for 1000 iterations

    def mutate(self, i):
        # Get the winning neuron for the current individual
        winner = self.som.winner(self.population[i])
        
        # Calculate coordinates in the SOM grid
        x, y = winner

        # Adjust F and CR based on the SOM neuron's position (example)
        F = self.F + (x / self.som_grid_size - 0.5) * 0.2  # Small adjustment
        CR = self.CR + (y / self.som_grid_size - 0.5) * 0.2  # Small adjustment
        F = np.clip(F, 0.1, 0.9)
        CR = np.clip(CR, 0.1, 0.9)

        # DE/rand/1 mutation with adjusted parameters
        indices = np.random.choice(self.popsize, 3, replace=False)
        x_r1, x_r2, x_r3 = self.population[indices]
        v = x_r1 + F * (x_r2 - x_r3)

        return v, CR

    def crossover(self, i, mutant, CR):
        trial_vector = np.copy(self.population[i])
        j_rand = np.random.randint(self.dim)
        for j in range(self.dim):
            if np.random.rand() < CR or j == j_rand:
                trial_vector[j] = mutant[j]
        return trial_vector

    def handle_boundary(self, trial_vector, func):
        return np.clip(trial_vector, func.bounds.lb, func.bounds.ub)

    def __call__(self, func):
        self.initialize_population(func)
        self.train_som()  # Train SOM initially
        evals = self.popsize

        generation = 0

        while evals < self.budget:
            for i in range(self.popsize):
                mutant, CR = self.mutate(i)
                trial_vector = self.crossover(i, mutant, CR)
                trial_vector = self.handle_boundary(trial_vector, func)

                f_trial = func(trial_vector)
                evals += 1

                if f_trial < self.fitness[i]:
                    self.population[i] = trial_vector
                    self.fitness[i] = f_trial

                    if f_trial < self.best_fitness:
                        self.best_fitness = f_trial
                        self.best_solution = trial_vector

                if evals >= self.budget:
                    break
            
            generation +=1
            if generation % 10 == 0:
               self.train_som() #Retrain every 10 generations

        return self.best_fitness, self.best_solution