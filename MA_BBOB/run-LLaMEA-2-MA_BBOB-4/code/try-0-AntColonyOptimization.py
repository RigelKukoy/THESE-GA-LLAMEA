import numpy as np

class AntColonyOptimization:
    def __init__(self, budget=10000, dim=10, n_ants=50, rho=0.1, alpha=1, beta=2, q=1):
        self.budget = budget
        self.dim = dim
        self.n_ants = n_ants
        self.rho = rho  # Evaporation rate
        self.alpha = alpha  # Pheromone influence
        self.beta = beta  # Heuristic influence (fitness)
        self.q = q # Pheromone deposit constant
        self.lb = -5.0
        self.ub = 5.0
        self.pheromone = np.ones(dim) * 1e-6  # Initialize pheromone levels
        self.best_solution = None
        self.best_fitness = np.inf
        self.eval_count = 0

    def construct_solution(self):
        solution = np.zeros(self.dim)
        for i in range(self.dim):
            probabilities = (self.pheromone[i]**self.alpha) * ((self.ub - self.lb)**-self.beta)  # Simplified heuristic
            probabilities /= np.sum(probabilities)
            solution[i] = np.random.uniform(self.lb, self.ub)
        return solution

    def deposit_pheromone(self, solution, fitness):
        delta_pheromone = self.q / (fitness + 1e-9)  # Avoid division by zero
        return delta_pheromone
    
    def evaporate_pheromone(self):
        self.pheromone *= (1 - self.rho)
        self.pheromone = np.clip(self.pheromone, 1e-6, 1e6)

    def __call__(self, func):
        while self.eval_count < self.budget:
            solutions = []
            fitnesses = []

            # Ant colony constructs solutions
            for _ in range(self.n_ants):
                solution = self.construct_solution()
                fitness = func(solution)
                self.eval_count += 1

                solutions.append(solution)
                fitnesses.append(fitness)
                
                if fitness < self.best_fitness:
                    self.best_fitness = fitness
                    self.best_solution = solution.copy()
                
                if self.eval_count >= self.budget:
                    break
                    
            if self.eval_count >= self.budget:
                break
            
            # Pheromone update
            for i in range(self.n_ants):
                delta_pheromone = self.deposit_pheromone(solutions[i], fitnesses[i])
                
            self.pheromone += delta_pheromone
            self.pheromone = np.clip(self.pheromone, 1e-6, 1e6) # Clip for stability
                
            self.evaporate_pheromone()

        return self.best_fitness, self.best_solution