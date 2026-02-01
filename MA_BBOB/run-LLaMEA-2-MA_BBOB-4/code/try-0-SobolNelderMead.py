import numpy as np
from scipy.optimize import minimize

class SobolNelderMead:
    def __init__(self, budget=10000, dim=10, num_sobol_points=100):
        self.budget = budget
        self.dim = dim
        self.lb = -5.0
        self.ub = 5.0
        self.num_sobol_points = num_sobol_points

    def __call__(self, func):
        # Generate Sobol sequence points
        sobol_points = self.generate_sobol(self.num_sobol_points, self.dim)
        sobol_points = self.lb + sobol_points * (self.ub - self.lb)

        f_opt = np.Inf
        x_opt = None
        eval_count = 0

        # Evaluate Sobol points
        for i in range(self.num_sobol_points):
            f = func(sobol_points[i])
            eval_count += 1
            if f < f_opt:
                f_opt = f
                x_opt = sobol_points[i].copy()

            if eval_count >= self.budget:
                return f_opt, x_opt


        # Local search using Nelder-Mead from the best Sobol point
        remaining_budget = self.budget - eval_count

        if remaining_budget > 0:
            result = minimize(func, x_opt, method='Nelder-Mead',
                                options={'maxiter': remaining_budget, 'maxfev': remaining_budget})
            
            if result.fun < f_opt:
                f_opt = result.fun
                x_opt = result.x
                
        return f_opt, x_opt

    def generate_sobol(self, n, dim):
        """Generate Sobol sequence points."""
        try:
            import sobol_seq
            points = sobol_seq.i4_sobol_generate(dim, n)
            return points
        except ImportError:
            print("Sobol sequence library not found. Please install 'sobol_seq'. Returning random samples instead.")
            return np.random.rand(n, dim)