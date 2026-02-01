import numpy as np

class AdaptiveNeighborhoodSearch:
    def __init__(self, budget=10000, dim=10, initial_step_size=0.1, reduction_factor=0.9, expansion_factor=1.1):
        self.budget = budget
        self.dim = dim
        self.step_size = initial_step_size
        self.reduction_factor = reduction_factor
        self.expansion_factor = expansion_factor
        self.f_opt = np.Inf
        self.x_opt = None
        self.eval_count = 0

    def __call__(self, func):
        # Initialize with a random solution
        x = np.random.uniform(func.bounds.lb, func.bounds.ub, size=self.dim)
        f = func(x)
        self.eval_count += 1
        self.f_opt = f
        self.x_opt = x.copy()
        
        while self.eval_count < self.budget:
            # Generate a neighbor by adding a random displacement
            x_new = x + np.random.normal(0, self.step_size, size=self.dim)
            x_new = np.clip(x_new, func.bounds.lb, func.bounds.ub)
            
            f_new = func(x_new)
            self.eval_count += 1
            
            if f_new < self.f_opt:
                # Accept the new solution
                self.f_opt = f_new
                self.x_opt = x_new.copy()
                x = x_new.copy()
                # Increase the step size (expansion)
                self.step_size *= self.expansion_factor
            else:
                # Reduce the step size (contraction)
                self.step_size *= self.reduction_factor
            
            # Limit the step size
            self.step_size = np.clip(self.step_size, 1e-6, 1.0)  # Ensure step size stays within reasonable bounds

            if self.eval_count >= self.budget:
                break
                
        return self.f_opt, self.x_opt