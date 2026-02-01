import numpy as np

class AdaptiveGaussianSearchMomentum:
    def __init__(self, budget=10000, dim=10, pop_size=20, lr=0.1, momentum=0.9):
        self.budget = budget
        self.dim = dim
        self.pop_size = pop_size
        self.lr = lr  # Learning rate for step size adaptation
        self.step_size = 1.0  # Initial step size
        self.momentum = momentum
        self.velocity = np.zeros(dim) # Initialize velocity for momentum

    def __call__(self, func):
        # Initialize population within bounds
        population = np.random.uniform(func.bounds.lb, func.bounds.ub, size=(self.pop_size, self.dim))
        fitness = np.array([func(x) for x in population])
        self.budget -= self.pop_size

        self.f_opt = np.min(fitness)
        self.x_opt = population[np.argmin(fitness)]

        generation = 0
        while self.budget > 0:
            generation += 1
            # Mutation: Gaussian perturbation with adaptive step size and momentum
            mutation = np.random.normal(0, self.step_size, size=(self.pop_size, self.dim))
            
            # Update velocity with momentum
            self.velocity = self.momentum * self.velocity + (1 - self.momentum) * np.mean(mutation, axis=0)
            
            offspring = population + mutation #+ self.velocity # Adding momentum to the offspring

            # Clip offspring to remain within bounds
            offspring = np.clip(offspring, func.bounds.lb, func.bounds.ub)

            # Evaluate offspring
            offspring_fitness = np.array([func(x) for x in offspring])
            self.budget -= self.pop_size
            if self.budget <= 0:
                offspring_fitness = offspring_fitness[:self.pop_size + self.budget]

            # Selection: Replace parents with better offspring
            for i in range(len(offspring_fitness)):
                if offspring_fitness[i] < fitness[i]:
                    fitness[i] = offspring_fitness[i]
                    population[i] = offspring[i]

            # Update best solution
            best_index = np.argmin(fitness)
            if fitness[best_index] < self.f_opt:
                self.f_opt = fitness[best_index]
                self.x_opt = population[best_index]

            # Adapt step size
            if generation % 10 == 0:  # Adjust step size every 10 generations
                success_rate = np.sum(offspring_fitness < fitness) / self.pop_size
                if success_rate > 0.2:
                    self.step_size *= (1 + self.lr)  # Increase step size if exploration is promising
                    self.lr *= 0.95 # Reduce learning rate when step size increases
                else:
                    self.step_size *= (1 - self.lr)  # Decrease step size if exploration is not fruitful
                    self.lr *= 1.05 # Increase learning rate when step size decreases
                    self.lr = min(self.lr, 0.5) # Cap the learning rate
                self.step_size = max(self.step_size, 1e-6)  # Ensure step size doesn't become too small
                
        return self.f_opt, self.x_opt