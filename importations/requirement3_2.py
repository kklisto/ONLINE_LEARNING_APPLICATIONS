import numpy as np

class UCB1Agent_2D:
    def __init__(self, K1,K2 ,T, range=1):
        self.K1 = K1
        self.K2 = K2
        self.T = T
        self.range = range
        self.a_t_1 = None
        self.a_t_2 = None
        self.average_rewards_1 = np.zeros(K1)
        self.average_rewards_2 = np.zeros(K2)
        self.N_pulls_1 = np.zeros(K1)
        self.N_pulls_2 = np.zeros(K2)
        self.t = 0
    
    def pull_arm(self):
        if self.t < self.K1:
            self.a_t_1 = self.t 
        if self.t < self.K2:
            self.a_t_2 = self.t    
        else:
            ucbs_1 = self.average_rewards_1 + self.range*np.sqrt(2*np.log(self.T)/self.N_pulls_1)
            ucbs_2 = self.average_rewards_2 + self.range*np.sqrt(2*np.log(self.T)/self.N_pulls_2)
            self.a_t_1 = np.argmax(ucbs_1)
            self.a_t_2 = np.argmax(ucbs_2)
        return self.a_t_1, self.a_t_2
    
    def update(self, r_t_1, r_t_2):
        self.N_pulls_1[self.a_t_1] += 1
        self.N_pulls_2[self.a_t_2] += 1
        self.average_rewards_1[self.a_t_1] += (r_t_1 - self.average_rewards_1[self.a_t_1])/self.N_pulls_1[self.a_t_1]
        self.average_rewards_2[self.a_t_2] += (r_t_2 - self.average_rewards_2[self.a_t_2])/self.N_pulls_2[self.a_t_2]
        self.t += 1


class PricingEnv_2D:
    def __init__(self,n_t,conversion_probability_1, conversion_probability_2,cost1 , cost2, prices1, prices2,seed):
        np.random.seed(seed)
        self.conversion_probability_1 = conversion_probability_1
        self.conversion_probability_2 = conversion_probability_2
        self.cost1 = cost1
        self.cost2 = cost2
        self.rewards_1 = np.random.binomial(n=n_t, p=self.conversion_probability_1)*(prices1-cost1)
        self.rewards_2 = np.random.binomial(n=n_t, p=self.conversion_probability_2)*(prices2-cost2)
        self.t = 0

    def round(self, a_t_1, a_t_2):
        r_t_1 = self.rewards_1[a_t_1]
        r_t_2 = self.rewards_2[a_t_2]
        self.t +=1
        return r_t_1, r_t_2
    

class RBFGaussianProcess:
    def __init__(self, scale=1, reg=1e-2):
        self.scale = scale 
        self.reg = reg
        self.k_xx_inv = None

    def rbf_kernel_incr_inv(self, B, C, D):
        temp = np.linalg.inv(D - C @ self.k_xx_inv @ B)
        block1 = self.k_xx_inv + self.k_xx_inv @ B @ temp @ C @ self.k_xx_inv
        block2 = - self.k_xx_inv @ B @ temp
        block3 = - temp @ C @ self.k_xx_inv
        block4 = temp
        res1 = np.concatenate((block1, block2), axis=1)
        res2 = np.concatenate((block3, block4), axis=1)
        res = np.concatenate((res1, res2), axis=0)
        return res

    def rbf_kernel(self, a, b):
        a_ = a.reshape(-1, 1)
        b_ = b.reshape(-1, 1)
        output = -1 * np.ones((a_.shape[0], b_.shape[0]))
        for i in range(a_.shape[0]):
            output[i, :] = np.power(a_[i] - b_, 2).ravel()
        return np.exp(-self.scale * output)
    
    def fit(self, x=np.array([]), y=np.array([])):
        x,y = np.array(x),np.array(y)
        if self.k_xx_inv is None:
            self.y = y.reshape(-1,1)
            self.x = x.reshape(-1,1)
            k_xx = self.rbf_kernel(self.x, self.x) + self.reg * np.eye(self.x.shape[0])
            self.k_xx_inv = np.linalg.inv(k_xx)
        else:
            B = self.rbf_kernel(self.x, x)
            self.x = np.vstack((self.x, x))
            self.y = np.vstack((self.y, y))
            self.k_xx_inv = self.rbf_kernel_incr_inv(B, B.T, np.array([1 + self.reg]))

        return self

    def predict(self, x_predict):
        k = self.rbf_kernel(x_predict, self.x)

        mu_hat = k @ self.k_xx_inv @ self.y
        sigma_hat = 1 - np.diag(k @ self.k_xx_inv @ k.T)

        return mu_hat.ravel(), sigma_hat.ravel()
    

class GPUCBAgent_2D:
    def __init__(self, T, discretization=100):
        self.T = T
        self.arms_1 = np.linspace(0, 1, discretization)
        self.arms_2 = np.linspace(0, 1, discretization)
        self.gp_1 = RBFGaussianProcess(scale=2).fit()
        self.gp_2 = RBFGaussianProcess(scale=2).fit()
        self.a_t_1 = None
        self.a_t_2 = None
        self.action_hist_1 = np.array([])
        self.reward_hist_1 = np.array([])
        self.action_hist_2 = np.array([])
        self.reward_hist_2 = np.array([])
        self.mu_t_1 = np.zeros(discretization)
        self.sigma_t_1 = np.zeros(discretization)
        self.mu_t_2 = np.zeros(discretization)
        self.sigma_t_2 = np.zeros(discretization)
        self.gamma = lambda t: np.log(t+1)**2 
        self.beta = lambda t: 1 + 0.5*np.sqrt(2 * (self.gamma(t) + 1 + np.log(T)))
        self.N_pulls_1 = np.zeros(discretization)
        self.N_pulls_2 = np.zeros(discretization)
        self.t = 0
    
    def pull_arm(self):
        self.mu_t_1, self.sigma_t_1 = self.gp_1.predict(self.arms_1)
        self.mu_t_2, self.sigma_t_2 = self.gp_2.predict(self.arms_2) 
        ucbs_1 = self.mu_t_1 + self.beta(self.t) * self.sigma_t_1
        ucbs_2 = self.mu_t_2 + self.beta(self.t) * self.sigma_t_2
        self.a_t_1 = np.argmax(ucbs_1)
        self.a_t_2 = np.argmax(ucbs_2)
        return self.a_t_1 , self.a_t_2
    
    def update(self, r_t_1, r_t_2):
        self.N_pulls_1[self.a_t_1] += 1
        self.N_pulls_2[self.a_t_2] += 1
        self.action_hist_1 = np.append(self.action_hist_1, self.arms_1[self.a_t_1])
        self.reward_hist_1 = np.append(self.reward_hist_1, r_t_2)
        self.gp_1 = self.gp_1.fit(self.arms_1[self.a_t_1], r_t_1)
        self.gp_2 = self.gp_2.fit(self.arms_2[self.a_t_2], r_t_2)
        self.t += 1


