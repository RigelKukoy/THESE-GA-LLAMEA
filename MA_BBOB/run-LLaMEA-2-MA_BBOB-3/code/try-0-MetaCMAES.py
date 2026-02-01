import numpy as np
from scipy.optimize import minimize

class MetaCMAES:
    def __init__(self, budget=10000, dim=10, popsize=None, sigma0=0.2, validation_fraction=0.2, num_validation_samples=100):
        self.budget = budget
        self.dim = dim
        self.sigma0 = sigma0
        self.popsize = popsize if popsize is not None else 4 + int(3 * np.log(self.dim))
        self.mu = self.popsize // 2
        self.validation_fraction = validation_fraction
        self.num_validation_samples = num_validation_samples

        self.weights = np.log(self.mu + 1/2) - np.log(np.arange(1, self.mu + 1))
        self.weights = self.weights / np.sum(self.weights)

        self.m = None
        self.sigma = None
        self.C = None
        self.pc = None
        self.ps = None
        self.eigenspace = None
        self.eigenvalues = None

        self.f_opt = np.Inf
        self.x_opt = None
        self.func_evals = 0
        self.validation_func = None
        self.best_params = None

    def initialize(self):
        self.m = np.random.uniform(-2, 2, size=self.dim)
        self.sigma = self.sigma0
        self.C = np.eye(self.dim)
        self.pc = np.zeros(self.dim)
        self.ps = np.zeros(self.dim)
        self.eigenspace = np.eye(self.dim)
        self.eigenvalues = np.ones(self.dim)

    def sample(self):
        z = np.random.normal(0, 1, size=(self.dim, self.popsize))
        x = self.m[:, np.newaxis] + self.sigma * (self.eigenspace @ (np.diag(np.sqrt(self.eigenvalues)) @ z))
        return x

    def update(self, x, fitness_values):
        idx = np.argsort(fitness_values)
        x_mu = x[:, idx[:self.mu]]
        z_mu = np.linalg.solve(self.eigenspace @ np.diag(np.sqrt(self.eigenvalues)), (x_mu - self.m[:, np.newaxis]) / self.sigma)

        m_old = self.m.copy()
        self.m = np.sum(x_mu * self.weights[np.newaxis, :], axis=1)
        self.ps = (1 - self.params['cs']) * self.ps + np.sqrt(self.params['cs'] * (2 - self.params['cs']) * np.sum(self.weights)) * (self.eigenspace @ z_mu.mean(axis=1))
        self.pc = (1 - self.params['c_cov']) * self.pc + np.sqrt(self.params['c_cov'] * (2 - self.params['c_cov']) * np.sum(self.weights)) * ((self.m - m_old) / self.sigma)

        C_temp = self.params['c_cov_mu'] * (self.pc[:, np.newaxis] @ self.pc[np.newaxis, :]) + \
               (1 - self.params['c_cov_mu']) * (self.C)

        self.sigma *= np.exp((self.params['cs'] / self.params['damps']) * (np.linalg.norm(self.ps) / np.sqrt(self.dim) - 1))
        self.C = C_temp

        self.eigenvalues, self.eigenspace = np.linalg.eigh(self.C)
        self.eigenvalues = np.maximum(self.eigenvalues, 1e-12)
    
    def set_params(self, params):
         self.params = {
            'cs': params[0],
            'damps': params[1],
            'c_cov': params[2],
            'c_cov_mu': params[3],
        }
    
    def get_params(self):
        return np.array([self.params['cs'], self.params['damps'], self.params['c_cov'], self.params['c_cov_mu']])

    def validate(self):
        validation_evals = 0
        f_val_opt = np.Inf
        
        m_temp = self.m.copy()
        sigma_temp = self.sigma
        C_temp = self.C.copy()
        pc_temp = self.pc.copy()
        ps_temp = self.ps.copy()
        eigenspace_temp = self.eigenspace.copy()
        eigenvalues_temp = self.eigenvalues.copy()

        for _ in range(self.num_validation_samples):
            x = np.random.uniform(-5, 5, size=(self.dim))  # Sample a point in search space
            f_val = self.validation_func(x)
            validation_evals += 1
            f_val_opt = min(f_val_opt, f_val)
        
        self.m = m_temp.copy()
        self.sigma = sigma_temp
        self.C = C_temp.copy()
        self.pc = pc_temp.copy()
        self.ps = ps_temp.copy()
        self.eigenspace = eigenspace_temp.copy()
        self.eigenvalues = eigenvalues_temp.copy()
        
        return f_val_opt, validation_evals
        

    def __call__(self, func):
        # Split budget for training and validation
        validation_budget = int(self.budget * self.validation_fraction)
        training_budget = self.budget - validation_budget
        
        # Create a validation function
        def create_validation_func(f):
            def val_func(x):
                return f(x)
            return val_func
        
        self.validation_func = create_validation_func(func)
        
        # Meta-optimization of CMA-ES parameters using Nelder-Mead
        def objective(params):
            self.set_params(params)
            self.initialize()
            
            temp_f_opt = np.Inf
            evals = 0
            
            while evals < (training_budget // 10): #Small amount of training to evaluate.
                x = self.sample()
                fitness_values = np.array([func(x[:, i]) for i in range(self.popsize)])
                evals += self.popsize
                
                best_index = np.argmin(fitness_values)
                temp_f_opt = min(temp_f_opt, fitness_values[best_index])
                self.update(x, fitness_values)
            
            validation_loss, _ = self.validate()  # Minimize validation loss
            return validation_loss

        #Initial guess
        initial_params = np.array([0.3, 1 + 2 * np.max([0, np.sqrt((self.mu - 1)/(self.dim + 1)) - 1]) + 0.3, (1 / (self.dim * np.sqrt(self.dim))) * 10, (1 / (self.dim * np.sqrt(self.dim))) * 10])
        
        #Define bounds and constraints
        bounds = [(0.01, 1.0), (1.0, 10.0), (0.001, 1.0), (0.001, 1.0)]
        
        # Perform meta-optimization
        res = minimize(objective, initial_params, method='Nelder-Mead', bounds=bounds, options={'maxiter': 10})  # Reduced maxiter
        self.best_params = res.x
        
        #Final run with best params
        self.set_params(self.best_params)
        self.initialize()

        while self.func_evals < training_budget:
            x = self.sample()
            fitness_values = np.array([func(x[:, i]) for i in range(self.popsize)])
            self.func_evals += self.popsize

            best_index = np.argmin(fitness_values)
            if fitness_values[best_index] < self.f_opt:
                self.f_opt = fitness_values[best_index]
                self.x_opt = x[:, best_index]

            self.update(x, fitness_values)

        return self.f_opt, self.x_opt