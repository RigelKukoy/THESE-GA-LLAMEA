import numpy as np

class AdaptiveRadiusMarginSampling:
    def __init__(self, budget=10000, dim=10, initial_radius=0.5, radius_decay=0.99, success_threshold=0.2, radius_increase=1.1):
        self.budget = budget
        self.dim = dim
        self.radius = initial_radius
        self.radius_decay = radius_decay
        self.success_threshold = success_threshold
        self.radius_increase = radius_increase
        self.f_opt = np.Inf
        self.x_opt = None
        self.success_count = 0
        self.iteration = 0

    def __call__(self, func):
        # Initial sample
        x = np.random.uniform(func.bounds.lb, func.bounds.ub, size=self.dim)
        f = func(x)
        self.budget -= 1
        self.f_opt = f
        self.x_opt = x
        
        while self.budget > 0:
            self.iteration += 1
            # Sample within a radius of the best solution
            lb = np.maximum(self.x_opt - self.radius, func.bounds.lb)
            ub = np.minimum(self.x_opt + self.radius, func.bounds.ub)
            x_new = np.random.uniform(lb, ub, size=self.dim)

            f_new = func(x_new)
            self.budget -= 1

            if f_new < self.f_opt:
                self.f_opt = f_new
                self.x_opt = x_new
                self.success_count += 1
            
            # Adjust the radius based on success rate
            if self.iteration % 10 == 0:  # Adjust every 10 iterations
                success_rate = self.success_count / 10
                if success_rate < self.success_threshold:
                    self.radius *= self.radius_increase # Increase radius if success rate is low to explore more
                else:
                    self.radius *= self.radius_decay # Decrease radius if success rate is high to exploit more
                
                self.radius = np.clip(self.radius, 1e-6, 5)  # Keep radius within reasonable bounds
                self.success_count = 0  # Reset success count

        return self.f_opt, self.x_opt