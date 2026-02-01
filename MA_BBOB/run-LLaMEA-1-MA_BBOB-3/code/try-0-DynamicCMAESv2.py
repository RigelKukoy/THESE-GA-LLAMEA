import numpy as np

class DynamicCMAESv2:
    def __init__(self, budget=10000, dim=10, sigma0=0.5, history_length=5, success_rate_history=10):
        self.budget = budget
        self.dim = dim
        self.sigma0 = sigma0
        self.history_length = history_length
        self.success_rate_history_length = success_rate_history
        self.min_popsize = 4
        self.max_popsize = 50
        self.target_popsize = min(self.max_popsize, self.min_popsize + int(np.floor(3 * np.log(self.dim))))
        self.popsize = self.target_popsize
        self.mu = self.popsize // 2
        self.weights = np.log(self.mu + 0.5) - np.log(np.arange(1, self.mu + 1))
        self.weights = self.weights / np.sum(self.weights)
        self.mueff = np.sum(self.weights)**2 / np.sum(self.weights**2)
        self.step_history = []
        self.last_f_opt = np.inf
        self.stagnation_counter = 0
        self.restart_iterations = 50
        self.success_history = []
        self.success_rate = 0.5 # Initial success rate
        self.success_rate_window = np.array([0.5] * self.success_rate_history_length) # initialize success rate
        self.cs = (self.mueff + 2) / (self.dim + self.mueff + 5)
        self.cc = (4 + self.mueff / self.dim) / (self.dim + 4 + 2 * self.mueff / self.dim)
        self.c1 = 2 / ((self.dim + 1.3)**2 + self.mueff)
        self.cmu = min(1 - self.c1, 2 * (self.mueff - 1 + 1 / self.mueff) / ((self.dim + 2)**2 + self.mueff))
        self.damps = 1 + 2 * max(0, np.sqrt((self.mueff - 1) / (self.dim + 1)) - 1) + self.cs
        self.chiN = np.sqrt(self.dim) * (1 - (1 / (4 * self.dim)) + 1 / (21 * self.dim**2))


    def __call__(self, func):
        mean = np.random.uniform(func.bounds.lb, func.bounds.ub, size=self.dim)
        sigma = self.sigma0
        C = np.eye(self.dim)
        ps = np.zeros(self.dim)
        pc = np.zeros(self.dim)
        
        f_opt = np.Inf
        x_opt = None
        eval_count = 0

        while eval_count < self.budget:
            # Adjust population size based on success rate
            if len(self.success_history) > self.success_rate_history_length:
                self.success_rate_window = np.array(self.success_history[-self.success_rate_history_length:])
                self.success_rate = np.mean(self.success_rate_window)

                if self.success_rate > 0.7 and self.popsize < self.max_popsize:
                    self.popsize = min(self.popsize + 1, self.max_popsize)
                    self.mu = self.popsize // 2
                    self.weights = np.log(self.mu + 0.5) - np.log(np.arange(1, self.mu + 1))
                    self.weights = self.weights / np.sum(self.weights)
                    self.mueff = np.sum(self.weights)**2 / np.sum(self.weights**2)
                    self.c1 = 2 / ((self.dim + 1.3)**2 + self.mueff)
                    self.cmu = min(1 - self.c1, 2 * (self.mueff - 1 + 1 / self.mueff) / ((self.dim + 2)**2 + self.mueff))

                elif self.success_rate < 0.3 and self.popsize > self.min_popsize:
                    self.popsize = max(self.popsize - 1, self.min_popsize)
                    self.mu = self.popsize // 2
                    self.weights = np.log(self.mu + 0.5) - np.log(np.arange(1, self.mu + 1))
                    self.weights = self.weights / np.sum(self.weights)
                    self.mueff = np.sum(self.weights)**2 / np.sum(self.weights**2)
                    self.c1 = 2 / ((self.dim + 1.3)**2 + self.mueff)
                    self.cmu = min(1 - self.c1, 2 * (self.mueff - 1 + 1 / self.mueff) / ((self.dim + 2)**2 + self.mueff))
            
            try:
                C_evals, C_evecs = np.linalg.eigh(C)
                C_evals = np.maximum(C_evals, 1e-10)
                C_sqrt = C_evecs @ np.diag(np.sqrt(C_evals)) @ C_evecs.T
                C_invsqrt = C_evecs @ np.diag(1 / np.sqrt(C_evals)) @ C_evecs.T
            except np.linalg.LinAlgError:
                C = np.eye(self.dim)
                C_evals, C_evecs = np.linalg.eigh(C)
                C_sqrt = C_evecs @ np.diag(np.sqrt(C_evals)) @ C_evecs.T
                C_invsqrt = C_evecs @ np.diag(1 / np.sqrt(C_evals)) @ C_evecs.T

            z = np.random.normal(0, 1, size=(self.popsize, self.dim))
            x = mean + sigma * (C_sqrt @ z.T).T
            
            f = np.array([func(xi) if eval_count + i < self.budget else np.inf for i, xi in enumerate(x)])
            eval_count += len(x)

            idx = np.argsort(f)
            x = x[idx]
            z = z[idx]
            f = f[idx]

            if f[0] < f_opt:
                f_opt = f[0]
                x_opt = x[0]
                self.stagnation_counter = 0
                self.success_history.append(1)
            else:
                self.stagnation_counter += 1
                self.success_history.append(0)
                
            if self.stagnation_counter > self.restart_iterations:
                mean = np.random.uniform(func.bounds.lb, func.bounds.ub, size=self.dim)
                sigma = self.sigma0
                C = np.eye(self.dim)
                ps = np.zeros(self.dim)
                pc = np.zeros(self.dim)
                C_evals, C_evecs = np.linalg.eigh(C)
                C_sqrt = C_evecs @ np.diag(np.sqrt(C_evals)) @ C_evecs.T
                C_invsqrt = C_evecs @ np.diag(1 / np.sqrt(C_evals)) @ C_evecs.T
                self.stagnation_counter = 0

            x_mu = x[:self.mu]
            z_mu = z[:self.mu]

            mean_new = np.sum(x_mu * self.weights[:, None], axis=0)
            z_w = np.sum(z_mu * self.weights[:, None], axis=0)

            ps = (1 - self.cs) * ps + np.sqrt(self.cs * (2 - self.cs) * self.mueff) * (C_invsqrt @ (mean_new - mean) / sigma)
            hsig = np.linalg.norm(ps) / np.sqrt(1 - (1 - self.cs)**(2 * eval_count / self.popsize)) < self.chiN * (1.4 + 2 / (self.dim + 1))
            pc = (1 - self.cc) * pc + hsig * np.sqrt(self.cc * (2 - self.cc) * self.mueff) * (mean_new - mean) / sigma
            
            C = (1 - self.c1 - self.cmu) * C + self.c1 * (pc[:, None] @ pc[None, :])
            C += self.cmu * np.sum(self.weights[:, None, None] * (z_mu[:, :, None] @ z_mu[:, None, :]), axis=0)

            # Adapt step size based on success rate
            if len(self.success_history) > 0:
                self.success_rate_window = np.concatenate((self.success_rate_window[1:], [self.success_history[-1]]))
                self.success_rate = np.mean(self.success_rate_window)
                sigma *= np.exp(0.1 * (self.success_rate - 0.5))

            mean = mean_new
            
            if eval_count % (self.popsize * 5) == 0:
                C = np.triu(C) + np.triu(C, 1).T
                try:
                    C_evals, C_evecs = np.linalg.eigh(C)
                    C_evals = np.maximum(C_evals, 1e-10)
                    C = C_evecs @ np.diag(C_evals) @ C_evecs.T
                except np.linalg.LinAlgError:
                    C = np.eye(self.dim)
                    C_evals, C_evecs = np.linalg.eigh(C)
                    C = C_evecs @ np.diag(C_evals) @ C_evecs.T
                    

        return f_opt, x_opt