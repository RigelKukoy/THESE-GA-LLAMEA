import numpy as np

class AdaptiveOrthogonalDE:
    def __init__(self, budget=10000, dim=10, pop_size=50, F_initial=0.5, CR=0.7, diversity_threshold=0.1, orthogonal_learning_rate=0.1):
        self.budget = budget
        self.dim = dim
        self.pop_size = pop_size
        self.F_initial = F_initial  # Initial mutation factor
        self.CR = CR  # Crossover rate
        self.diversity_threshold = diversity_threshold # Threshold for triggering mutation adaptation
        self.orthogonal_learning_rate = orthogonal_learning_rate
        self.F = np.full(pop_size, F_initial)

    def calculate_diversity(self, population):
        # Calculate the average pairwise distance between individuals in the population
        distances = np.sum((population[:, np.newaxis, :] - population[np.newaxis, :, :])**2, axis=2)
        diversity = np.mean(distances)
        return diversity
        
    def orthogonal_design(self, x, num_samples=5):
        # Generate orthogonal samples around x
        orthogonal_samples = []
        for _ in range(num_samples):
            direction = np.random.randn(self.dim)
            direction /= np.linalg.norm(direction) # Normalize direction
            step_size = np.random.uniform(-self.orthogonal_learning_rate, self.orthogonal_learning_rate)
            sample = x + step_size * direction
            orthogonal_samples.append(sample)
        return np.array(orthogonal_samples)

    def __call__(self, func):
        # Initialization
        lb = func.bounds.lb
        ub = func.bounds.ub
        population = np.random.uniform(lb, ub, size=(self.pop_size, self.dim))
        fitness = np.array([func(x) for x in population])
        self.budget -= self.pop_size

        best_index = np.argmin(fitness)
        self.f_opt = fitness[best_index]
        self.x_opt = population[best_index]

        # Evolution loop
        while self.budget > 0:
            diversity = self.calculate_diversity(population)
            
            for i in range(self.pop_size):
                # Mutation strategy selection based on diversity
                if diversity > self.diversity_threshold:
                    # Current-to-best mutation
                    r1, r2 = np.random.choice(self.pop_size, 2, replace=False)
                    mutated_vector = population[i] + self.F[i] * (self.x_opt - population[i]) + self.F[i] * (population[r1] - population[r2])
                else:
                    # Random differential mutation
                    r1, r2, r3 = np.random.choice(self.pop_size, 3, replace=False)
                    mutated_vector = population[r1] + self.F[i] * (population[r2] - population[r3])

                mutated_vector = np.clip(mutated_vector, lb, ub) # keep within bounds

                # Crossover
                j_rand = np.random.randint(self.dim)
                trial_vector = np.copy(population[i])
                for j in range(self.dim):
                    if np.random.rand() < self.CR or j == j_rand:
                        trial_vector[j] = mutated_vector[j]

                # Evaluation
                f_trial = func(trial_vector)
                self.budget -= 1

                if f_trial < fitness[i]:
                    # Replacement
                    fitness[i] = f_trial
                    population[i] = trial_vector
                    
                    # Orthogonal Learning
                    orthogonal_samples = self.orthogonal_design(trial_vector)
                    orthogonal_fitness = [func(sample) for sample in orthogonal_samples]
                    self.budget -= len(orthogonal_samples)

                    best_orthogonal_index = np.argmin(orthogonal_fitness)
                    if orthogonal_fitness[best_orthogonal_index] < f_trial:
                         population[i] = orthogonal_samples[best_orthogonal_index]
                         fitness[i] = orthogonal_fitness[best_orthogonal_index]
                         f_trial = orthogonal_fitness[best_orthogonal_index]
                    # Update best solution
                    if f_trial < self.f_opt:
                        self.f_opt = f_trial
                        self.x_opt = population[i].copy()
                
                # Adapt mutation factor
                if f_trial < fitness[i]:
                    self.F[i] = self.F_initial # Reset if improvement found
                else:
                    self.F[i] = min(self.F[i] * 1.1, 1.0) # Increase slightly if no improvement
                
                if self.budget <= 0:
                    break
            if self.budget <= 0:
                break
        return self.f_opt, self.x_opt