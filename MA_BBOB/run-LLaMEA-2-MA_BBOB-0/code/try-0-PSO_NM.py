import numpy as np
from scipy.optimize import minimize

class PSO_NM:
    def __init__(self, budget=10000, dim=10, pop_size=30, inertia_weight=0.7, cognitive_coeff=1.4, social_coeff=1.4, nm_max_iter=50):
        self.budget = budget
        self.dim = dim
        self.pop_size = pop_size
        self.inertia_weight = inertia_weight
        self.cognitive_coeff = cognitive_coeff
        self.social_coeff = social_coeff
        self.nm_max_iter = nm_max_iter
        self.particles = None
        self.velocities = None
        self.personal_best_positions = None
        self.personal_best_fitness = None
        self.global_best_position = None
        self.global_best_fitness = np.inf
        self.lb = None
        self.ub = None

    def initialize_particles(self):
        self.particles = np.random.uniform(self.lb, self.ub, size=(self.pop_size, self.dim))
        self.velocities = np.random.uniform(-0.1 * (self.ub - self.lb), 0.1 * (self.ub - self.lb), size=(self.pop_size, self.dim))
        self.personal_best_positions = np.copy(self.particles)
        self.personal_best_fitness = np.full(self.pop_size, np.inf)

    def pso_step(self, func):
        for i in range(self.pop_size):
            inertia = self.inertia_weight * self.velocities[i]
            cognitive = self.cognitive_coeff * np.random.rand(self.dim) * (self.personal_best_positions[i] - self.particles[i])
            social = self.social_coeff * np.random.rand(self.dim) * (self.global_best_position - self.particles[i])

            self.velocities[i] = inertia + cognitive + social
            self.particles[i] += self.velocities[i]
            self.particles[i] = np.clip(self.particles[i], self.lb, self.ub)

            fitness = func(self.particles[i])
            self.budget -= 1
            
            if fitness < self.global_best_fitness:
                self.global_best_fitness = fitness
                self.global_best_position = np.copy(self.particles[i])

            if fitness < self.personal_best_fitness[i]:
                self.personal_best_fitness[i] = fitness
                self.personal_best_positions[i] = np.copy(self.particles[i])
    
    def nelder_mead_optimization(self, func, x0):
        bounds = [(self.lb, self.ub)] * self.dim
        result = minimize(func, x0, method='Nelder-Mead', options={'maxiter': self.nm_max_iter}, bounds=bounds)
        self.budget -= result.nit # Account function evaluations
        return result.fun, result.x

    def __call__(self, func):
        self.lb, self.ub = func.bounds.lb, func.bounds.ub
        self.initialize_particles()

        # Initial evaluation
        for i in range(self.pop_size):
            fitness = func(self.particles[i])
            self.budget -= 1
            if fitness < self.global_best_fitness:
                self.global_best_fitness = fitness
                self.global_best_position = np.copy(self.particles[i])
            self.personal_best_fitness[i] = fitness
            self.personal_best_positions[i] = np.copy(self.particles[i])
            
        while self.budget > 0:
            # PSO Step
            self.pso_step(func)

            # Adaptive switching based on population diversity
            diversity = np.std(self.particles)

            if diversity < 0.1 * (self.ub - self.lb):  # Low diversity: switch to Nelder-Mead
                # Apply Nelder-Mead around the global best
                nm_fitness, nm_position = self.nelder_mead_optimization(func, self.global_best_position)
                if nm_fitness < self.global_best_fitness:
                    self.global_best_fitness = nm_fitness
                    self.global_best_position = nm_position
                
                # Perturb particles to increase diversity
                self.particles = np.random.uniform(self.lb, self.ub, size=(self.pop_size, self.dim))
                for i in range(self.pop_size):
                   self.personal_best_positions[i] = np.copy(self.particles[i])
                   self.personal_best_fitness[i] = func(self.particles[i])
                   self.budget -= 1
                   if self.personal_best_fitness[i] < self.global_best_fitness:
                       self.global_best_fitness = self.personal_best_fitness[i]
                       self.global_best_position = np.copy(self.particles[i])

            
        return self.global_best_fitness, self.global_best_position