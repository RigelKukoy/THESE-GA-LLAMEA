import numpy as np

class QuadraticTrustRegion:
    def __init__(self, budget=10000, dim=10, initial_radius=0.1, min_radius=1e-6, expand_factor=2.0, contract_factor=0.5, success_threshold=0.25):
        self.budget = budget
        self.dim = dim
        self.radius = initial_radius
        self.min_radius = min_radius
        self.expand_factor = expand_factor
        self.contract_factor = contract_factor
        self.success_threshold = success_threshold
        self.x_best = np.random.uniform(-5, 5, size=dim)
        self.f_best = np.inf
        self.evals = 0

    def approximate_quadratic(self, func, x):
        """Approximate the function locally with a quadratic model."""
        delta = 1e-3  # Small perturbation for derivative estimation
        grad = np.zeros(self.dim)
        for i in range(self.dim):
            x_plus = x.copy()
            x_minus = x.copy()
            x_plus[i] += delta
            x_minus[i] -= delta
            
            # Evaluate only if within budget
            if self.evals + 2 <= self.budget:
                grad[i] = (func(x_plus) - func(x_minus)) / (2 * delta)
                self.evals += 2
            else:
                return None, None # Return None if out of budget
            
        hess = np.zeros((self.dim, self.dim))
        for i in range(self.dim):
            for j in range(self.dim):
                x_plus_i_plus_j = x.copy()
                x_plus_i_minus_j = x.copy()
                x_minus_i_plus_j = x.copy()
                x_minus_i_minus_j = x.copy()

                x_plus_i_plus_j[i] += delta
                x_plus_i_plus_j[j] += delta

                x_plus_i_minus_j[i] += delta
                x_plus_i_minus_j[j] -= delta
                
                x_minus_i_plus_j[i] -= delta
                x_minus_i_plus_j[j] += delta

                x_minus_i_minus_j[i] -= delta
                x_minus_i_minus_j[j] -= delta

                # Evaluate only if within budget
                if self.evals + 4 <= self.budget:
                    hess[i, j] = (func(x_plus_i_plus_j) - func(x_plus_i_minus_j) - func(x_minus_i_plus_j) + func(x_minus_i_minus_j)) / (4 * delta * delta)
                    self.evals += 4
                else:
                    return None, None  # Return None if out of budget
        return grad, hess

    def solve_trust_region_subproblem(self, grad, hess):
        """Solve the trust region subproblem using a simple approach."""
        try:
            # Attempt to solve directly using Newton step
            delta_x = -np.linalg.solve(hess, grad)
            if np.linalg.norm(delta_x) <= self.radius:
                return delta_x
        except np.linalg.LinAlgError:
            pass  # Hessian is singular or not positive definite
        
        # If Newton step fails, use gradient descent within the trust region
        delta_x = -self.radius * grad / np.linalg.norm(grad) if np.linalg.norm(grad) > 0 else np.random.uniform(-self.radius, self.radius, self.dim)
        return delta_x

    def __call__(self, func):
        self.x_best = np.random.uniform(func.bounds.lb, func.bounds.ub, size=self.dim)
        self.f_best = func(self.x_best)
        self.evals += 1

        while self.evals < self.budget:
            grad, hess = self.approximate_quadratic(func, self.x_best)
            if grad is None or hess is None:
                break # Break if approximate_quadratic returns None (out of budget)
            
            delta_x = self.solve_trust_region_subproblem(grad, hess)
            x_new = self.x_best + delta_x

            # Ensure x_new stays within bounds using clipping
            x_new = np.clip(x_new, func.bounds.lb, func.bounds.ub)

            f_new = func(x_new)
            self.evals += 1
            
            actual_reduction = self.f_best - f_new
            predicted_reduction = -grad @ delta_x - 0.5 * delta_x @ hess @ delta_x
            
            if predicted_reduction > 0:
                rho = actual_reduction / predicted_reduction
            else:
                rho = 0  # Avoid division by zero
            
            if rho > self.success_threshold:
                self.x_best = x_new
                self.f_best = f_new
                self.radius *= self.expand_factor
            else:
                self.radius *= self.contract_factor

            self.radius = max(self.radius, self.min_radius)

        return self.f_best, self.x_best