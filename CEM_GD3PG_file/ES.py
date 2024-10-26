import numpy as np
from copy import deepcopy
''''   本文copy from  https://github.com/apourchot/CEM-RL/blob/1a458223c4250a335485871b54064b077908452c/ES.py  未加任何修改'''
#from Optimizers import Adam, BasicSGD  #这行 暂时没用到 先注释


def compute_ranks(x):
    """
    Returns ranks in [0, len(x))]
    which returns ranks in [1, len(x)].
    (https://github.com/openai/evolution-strategies-starter/blob/master/es_distributed/es.py)
    """
    assert x.ndim == 1
    ranks = np.empty(len(x), dtype=int)
    ranks[x.argsort()] = np.arange(len(x))
    return ranks


def compute_centered_ranks(x):
    """
    https://github.com/openai/evolution-strategies-starter/blob/master/es_distributed/es.py
    """
    y = compute_ranks(x.ravel()).reshape(x.shape).astype(np.float32)
    y /= (x.size - 1)
    y -= .5
    return y


def compute_weight_decay(weight_decay, model_param_list):
    model_param_grid = np.array(model_param_list)
    return -weight_decay * np.mean(model_param_grid * model_param_grid, axis=1)


class VES:

    """
    Basic Version of OpenAI Evolution Strategies
    """

    def __init__(self, num_params,
                 mu_init=None,
                 sigma_init=0.1,
                 lr=10**-2,
                 pop_size=256,
                 antithetic=True,
                 weight_decay=0,
                 rank_fitness=True):

        # misc
        self.num_params = num_params
        self.first_interation = True

        # distribution parameters
        if mu_init is None:
            self.mu = np.zeros(self.num_params)
        else:
            self.mu = np.array(mu_init)
        self.sigma = sigma_init

        # optimization stuff
        self.learning_rate = lr
        self.optimizer = Adam(self.learning_rate)

        # sampling stuff
        self.pop_size = pop_size
        self.antithetic = antithetic
        if self.antithetic:
            assert (self.pop_size % 2 == 0), "Population size must be even"
        self.weight_decay = weight_decay
        self.rank_fitness = rank_fitness

    def ask(self):
        """
        Returns a list of candidates parameterss
        """
        if self.antithetic:
            epsilon_half = np.random.randn(self.pop_size // 2, self.num_params)
            epsilon = np.concatenate([epsilon_half, - epsilon_half])

        else:
            epsilon = np.random.randn(self.pop_size, self.num_params)

        return self.mu + epsilon * self.sigma

    def tell(self, scores, solutions):
        """
        Updates the distribution
        """
        assert(len(scores) ==
               self.pop_size), "Inconsistent reward_table size reported."

        reward = np.array(scores)
        if self.rank_fitness:
            reward = compute_centered_ranks(reward)

        if self.weight_decay > 0:
            l2_decay = compute_weight_decay(self.weight_decay, solutions)
            reward += l2_decay

        epsilon = (solutions - self.mu) / self.sigma
        grad = -1/(self.sigma * self.pop_size) * np.dot(reward, epsilon)

        # optimization step
        step = self.optimizer.step(grad)
        self.mu += step

    def get_distrib_params(self):
        """
        Returns the parameters of the distrubtion:
        the mean and sigma
        """
        return np.copy(self.mu), np.copy(self.sigma ** 2)


class GES:

    """
    Guided Evolution Strategies
    """

    def __init__(self, num_params,
                 mu_init=None,
                 sigma_init=0.1,
                 lr=10**-2,
                 alpha=0.5,
                 beta=2,
                 k=1,
                 pop_size=256,
                 antithetic=True,
                 weight_decay=0,
                 rank_fitness=False):

        # misc
        self.num_params = num_params
        self.first_interation = True

        # distribution parameters
        if mu_init is None:
            self.mu = np.zeros(self.num_params)
        else:
            self.mu = np.array(mu_init)
        self.sigma = sigma_init
        self.U = np.ones((self.num_params, k))

        # optimization stuff
        self.alpha = alpha
        self.beta = beta
        self.k = k
        self.learning_rate = lr
        self.optimizer = Adam(self.learning_rate)

        # sampling stuff
        self.pop_size = pop_size
        self.antithetic = antithetic
        if self.antithetic:
            assert (self.pop_size % 2 == 0), "Population size must be even"
        self.weight_decay = weight_decay
        self.rank_fitness = rank_fitness

    def ask(self):
        """
        Returns a list of candidates parameterss
        """
        if self.antithetic:
            epsilon_half = np.sqrt(self.alpha / self.num_params) * \
                np.random.randn(self.pop_size // 2, self.num_params)
            epsilon_half += np.sqrt((1 - self.alpha) / self.k) * \
                np.random.randn(self.pop_size // 2, self.k) @ self.U.T
            epsilon = np.concatenate([epsilon_half, - epsilon_half])

        else:
            epsilon = np.sqrt(self.alpha / self.num_params) * \
                np.random.randn(self.pop_size, self.num_params)
            epsilon += np.sqrt(1 - self.alpha) * \
                np.random.randn(self.pop_size, self.num_params) @ self.U.T

        return self.mu + epsilon * self.sigma

    def tell(self, scores, solutions):
        """
        Updates the distribution
        """
        assert(len(scores) ==
               self.pop_size), "Inconsistent reward_table size reported."

        reward = np.array(scores)
        if self.rank_fitness:
            reward = compute_centered_ranks(reward)

        if self.weight_decay > 0:
            l2_decay = compute_weight_decay(self.weight_decay, solutions)
            reward += l2_decay

        epsilon = (solutions - self.mu) / self.sigma
        grad = -self.beta/(self.sigma * self.pop_size) * \
            np.dot(reward, epsilon)

        # optimization step
        step = self.optimizer.step(grad)
        self.mu += step

    def add(self, params, grads, fitness):
        """
        Adds new "gradient" to U
        """
        if params is not None:
            self.mu = params
        grads = grads / np.linalg.norm(grads)
        self.U[:, -1] = grads

    def get_distrib_params(self):
        """
        Returns the parameters of the distrubtion:
        the mean and sigma
        """
        return np.copy(self.mu), np.copy(self.sigma ** 2)


class sepCMAES:

    """
    CMAES implementation adapted from
    https://en.wikipedia.org/wiki/CMA-ES#Example_code_in_MATLAB/Octave
    """

    def __init__(self,
                 num_params,
                 mu_init=None,
                 sigma_init=1,
                 step_size_init=1,
                 pop_size=255,
                 antithetic=False,
                 weight_decay=0.01,
                 rank_fitness=True):

        # distribution parameters
        self.num_params = num_params
        if mu_init is not None:
            self.mu = np.array(mu_init)
        else:
            self.mu = np.zeros(num_params)
        self.antithetic = antithetic

        # stuff
        self.step_size = step_size_init
        self.p_c = np.zeros(self.num_params)
        self.p_s = np.zeros(self.num_params)
        self.cov = sigma_init * np.ones(num_params)

        # selection parameters
        self.pop_size = pop_size
        self.parents = pop_size // 2
        self.weights = np.array([np.log((self.parents + 1) / i)
                                 for i in range(1, self.parents + 1)])
        self.weights /= self.weights.sum()
        self.parents_eff = 1 / (self.weights ** 2).sum()
        self.rank_fitness = rank_fitness
        self.weight_decay = weight_decay

        # adaptation  parameters
        self.g = 1
        self.c_s = (self.parents_eff + 2) / \
            (self.num_params + self.parents_eff + 3)
        self.c_c = 4 / (self.num_params + 4)
        self.c_cov = 1 / self.parents_eff * 2 / ((self.num_params + np.sqrt(2)) ** 2) + (1 - 1 / self.parents_eff) * \
            min(1, (2 * self.parents_eff - 1) /
                (self.parents_eff + (self.num_params + 2) ** 2))
        self.c_cov *= (self.num_params + 2) / 3
        self.d_s = 1 + 2 * \
            max(0, np.sqrt((self.parents_eff - 1) /
                           (self.num_params + 1) - 1)) + self.c_s
        self.chi = np.sqrt(self.num_params) * (1 - 1 / (4 *
                                                        self.num_params) + 1 / (21 * self.num_params ** 2))

    def ask(self, pop_size):
        """
        Returns a list of candidates parameters
        """
        if self.antithetic:
            epsilon_half = np.random.randn(pop_size // 2, self.num_params)
            epsilon = np.concatenate([epsilon_half, - epsilon_half])

        else:
            epsilon = np.random.randn(pop_size, self.num_params)

        return self.mu + self.step_size * epsilon * np.sqrt(self.cov)

    def tell(self, solutions, scores):
        """
        Updates the distribution
        """

        scores = np.array(scores)
        scores *= -1
        idx_sorted = np.argsort(scores)

        # update mean
        old_mu = deepcopy(self.mu)
        self.mu = self.weights @ solutions[idx_sorted[:self.parents]]
        z = 1 / self.step_size * 1 / \
            np.sqrt(self.cov) * (solutions[idx_sorted[:self.parents]] - old_mu)
        z_w = self.weights @ z

        # update evolution paths
        self.p_s = (1 - self.c_s) * self.p_s + \
            np.sqrt(self.c_s * (2 - self.c_s) * self.parents_eff) * z_w

        tmp_1 = np.linalg.norm(self.p_s) / np.sqrt(1 - (1 - self.c_s) ** (2 * self.g)) \
            <= self.chi * (1.4 + 2 / (self.num_params + 1))

        self.p_c = (1 - self.c_c) * self.p_c + \
            tmp_1 * np.sqrt(self.c_c * (2 - self.c_c)
                            * self.parents_eff) * np.sqrt(self.cov) * z_w

        # update covariance matrix
        self.cov = (1 - self.c_cov) * self.cov + \
            self.c_cov * 1 / self.parents_eff * self.p_c * self.p_c + \
            self.c_cov * (1 - 1 / self.parents_eff) * \
            (self.weights @ (self.cov * z * z))

        # update step size
        self.step_size *= np.exp((self.c_s / self.d_s) *
                                 (np.linalg.norm(self.p_s) / self.chi - 1))
        self.g += 1

        print(self.cov)
        return idx_sorted[:self.parents]

    def get_distrib_params(self):
        """
        Returns the parameters of the distrubtion:
        the mean and the covariance matrix
        """
        return np.copy(self.mu), np.copy(self.step_size)**2 * np.copy(self.cov)


class sepCEMv2:

    """
    Cross-entropy methods.
    """

    def __init__(self, num_params,
                 mu_init=None,
                 sigma_init=1e-3,
                 pop_size=256,
                 damp=1e-3,
                 damp_limit=1e-5,
                 parents=None,
                 elitism=False,
                 antithetic=False):

        # misc
        self.num_params = num_params

        # distribution parameters
        if mu_init is None:
            self.mu = np.zeros(self.num_params)
        else:
            self.mu = np.array(mu_init)
        self.sigma = sigma_init
        self.damp = damp
        self.damp_limit = damp_limit
        self.tau = 0.95
        self.cov = self.sigma * np.ones(self.num_params)

        # elite stuff
        self.elitism = elitism
        self.elite = np.sqrt(self.sigma) * np.random.rand(self.num_params)
        self.elite_score = None

        # sampling stuff
        self.pop_size = pop_size
        self.antithetic = antithetic

        if self.antithetic:
            assert (self.pop_size % 2 == 0), "Population size must be even"
        if parents is None or parents <= 0:
            self.parents = pop_size // 2
        else:
            self.parents = parents
        self.weights = np.array([np.log((self.parents + 1) / i)
                                 for i in range(1, self.parents + 1)])
        self.weights /= self.weights.sum()

    def ask(self, pop_size):
        """
        Returns a list of candidates parameters
        """
        if self.antithetic and not pop_size % 2:
            epsilon_half = np.random.randn(pop_size // 2, self.num_params)
            epsilon = np.concatenate([epsilon_half, - epsilon_half])

        else:
            epsilon = np.random.randn(pop_size, self.num_params)

        inds = self.mu + epsilon * np.sqrt(self.cov)
        if self.elitism:
            inds[-1] = self.elite

        return inds

    def tell(self, solutions, scores):
        """
        Updates the distribution
        """
        scores = np.array(scores)
        scores *= -1
        idx_sorted = np.argsort(scores)

        old_mu = self.mu
        self.damp = self.damp * self.tau + (1 - self.tau) * self.damp_limit
        self.mu = self.weights @ solutions[idx_sorted[:self.parents]]

        z = (solutions[idx_sorted[:self.parents]] - old_mu)
        tmp = self.weights @ (z * z)
        beta = self.num_params * self.damp / np.sum(tmp)
        tmp *= beta

        alpha = 1
        self.cov = (alpha * tmp + (1 - alpha) *
                    self.damp * np.ones(self.num_params))

        print(self.damp, beta, np.max(self.cov))

        self.elite = solutions[idx_sorted[0]]
        self.elite_score = scores[idx_sorted[0]]
        print(self.cov)

    def get_distrib_params(self):
        """
        Returns the parameters of the distrubtion:
        the mean and sigma
        """
        return np.copy(self.mu), np.copy(self.cov)


class sepCEM:

    """
    Cross-entropy methods.
    """

    def __init__(self, num_params,
                 mu_init=None, # 均值
                 sigma_init=1e-3, # 方差
                 pop_size=256, # 种群大小
                 damp=1e-3, # 阻尼
                 damp_limit=1e-5, # 阻尼限制
                 parents=None, # 父代数量 默认为种群大小的一半
                 elitism=False, # 精英策略
                 antithetic=False): # 反向策略 是否使用对称样本（反对称样本），这可以减少样本的方差

        # misc
        self.num_params = num_params 

        # distribution parameters
        if mu_init is None:
            self.mu = np.zeros(self.num_params) # 均值
        else:
            self.mu = np.array(mu_init)
        self.sigma = sigma_init
        self.damp = damp
        self.damp_limit = damp_limit
        self.tau = 0.95
        self.cov = self.sigma * np.ones(self.num_params) # 方差

        # elite stuff
        self.elitism = elitism
        self.elite = np.sqrt(self.sigma) * np.random.rand(self.num_params)
        self.elite_score = None

        # sampling stuff
        self.pop_size = pop_size
        self.antithetic = antithetic

        if self.antithetic:
            assert (self.pop_size % 2 == 0), "Population size must be even"
        if parents is None or parents <= 0:
            self.parents = pop_size // 2
        else:
            self.parents = parents
        self.weights = np.array([np.log((self.parents + 1) / i)
                                 for i in range(1, self.parents + 1)])
        self.weights /= self.weights.sum()

    def ask(self, pop_size):
        """
        Returns a list of candidates parameters  返回一组候选参数
        """
        if self.antithetic and not pop_size % 2:
            epsilon_half = np.random.randn(pop_size // 2, self.num_params)
            epsilon = np.concatenate([epsilon_half, - epsilon_half])

        else:
            epsilon = np.random.randn(pop_size, self.num_params)

        inds = self.mu + epsilon * np.sqrt(self.cov)
        if self.elitism:
            inds[-1] = self.elite

        return inds # 大小为pop_size*num_params

    def tell(self, solutions, scores):
        """
        Updates the distribution  更新分布
        """
        scores = np.array(scores)
        scores *= -1
        idx_sorted = np.argsort(scores)

        old_mu = self.mu
        self.damp = self.damp * self.tau + (1 - self.tau) * self.damp_limit
        self.mu = self.weights @ solutions[idx_sorted[:self.parents]]

        z = (solutions[idx_sorted[:self.parents]] - old_mu)
        self.cov = 1 / self.parents * self.weights @ (
            z * z) + self.damp * np.ones(self.num_params)

        self.elite = solutions[idx_sorted[0]]
        self.elite_score = scores[idx_sorted[0]]
        #print(self.cov)

    def get_distrib_params(self):
        """
        Returns the parameters of the distrubtion:
        the mean and sigma
        """
        return np.copy(self.mu), np.copy(self.cov)


class Control:

    """
    Cross-entropy methods.
    """

    def __init__(self, num_params, mu_init, pop_size=256, sigma_init=1e-3):

        # misc
        self.num_params = num_params
        self.pop = np.sqrt(sigma_init) * np.random.randn(pop_size, num_params) + mu_init
        self.mu = np.zeros(num_params)

    def ask(self, pop_size):
        """
        Returns a list of candidates parameters
        """
        return self.pop

    def tell(self, solutions, scores):
        """
        Updates the distribution
        """
        self.mu = solutions[np.argmax(scores)]
        self.pop = solutions
        np.random.shuffle(self.pop)


class sepCEMA:

    """
    Cross-entropy methods.
    """

    def __init__(self, num_params,
                 mu_init=None,
                 sigma_init=1e-3,
                 pop_size=256,
                 parents=None,
                 elitism=False,
                 antithetic=False):

        # misc
        self.num_params = num_params

        # distribution parameters
        if mu_init is None:
            self.mu = np.zeros(self.num_params)
        else:
            self.mu = np.array(mu_init)
        self.sigma = sigma_init
        self.cov = self.sigma * np.ones(self.num_params)

        # elite stuff
        self.elitism = elitism
        self.elite = np.sqrt(self.sigma) * np.random.rand(self.num_params)
        self.elite_score = -np.inf

        # sampling stuff
        self.pop_size = pop_size
        self.antithetic = antithetic

        if self.antithetic:
            assert (self.pop_size % 2 == 0), "Population size must be even"
        if parents is None or parents <= 0:
            self.parents = pop_size // 2
        else:
            self.parents = parents
        self.weights = np.array([np.log((self.parents + 1) / i)
                                 for i in range(1, self.parents + 1)])
        self.weights /= self.weights.sum()

    def ask(self, pop_size):
        """
        Returns a list of candidates parameters
        """
        if self.antithetic and not pop_size % 2:
            epsilon_half = np.random.randn(pop_size // 2, self.num_params)
            epsilon = np.concatenate([epsilon_half, - epsilon_half])

        else:
            epsilon = np.random.randn(pop_size, self.num_params)

        inds = self.mu + epsilon * np.sqrt(self.cov)
        if self.elitism:
            inds[-1] = self.elite

        return inds

    def tell(self, solutions, scores):
        """
        Updates the distribution
        """
        scores = np.array(scores)
        scores *= -1
        idx_sorted = np.argsort(scores)

        # new and old mean
        old_mu = self.mu
        self.mu = self.weights @ solutions[idx_sorted[:self.parents]]

        # sigma adaptation
        if scores[idx_sorted[0]] > 0.95 * self.elite_score:
            self.sigma *= 0.95
        else:
            self.sigma *= 1.05
        self.elite = solutions[idx_sorted[0]]
        self.elite_score = scores[idx_sorted[0]]

        z = (solutions[idx_sorted[:self.parents]] - old_mu)
        self.cov = self.weights @ (z * z)
        self.cov = self.sigma * self.cov / np.linalg.norm(self.cov)
        print(self.cov)
        print(self.sigma)

    def get_distrib_params(self):
        """
        Returns the parameters of the distrubtion:
        the mean and sigma
        """
        return np.copy(self.mu), np.copy(self.cov)


class sepMCEM:

    """
    Cross-entropy methods with multiplicative noise. Not really working.
    """

    def __init__(self, num_params,
                 mu_init=None,
                 sigma_init=0.1,
                 pop_size=256,
                 damp=0.01,
                 parents=None,
                 antithetic=False):

        # misc
        self.num_params = num_params

        # distribution parameters
        if mu_init is None:
            self.mu = np.zeros(self.num_params)
        else:
            self.mu = np.array(mu_init)
        self.sigma = sigma_init
        self.damp = sigma_init
        self.damp_limit = damp
        self.cov = self.sigma * np.ones(self.num_params)
        self.tau = 0.95

        # sampling stuff
        self.pop_size = pop_size
        self.antithetic = antithetic
        if self.antithetic:
            assert (self.pop_size % 2 == 0), "Population size must be even"
        if parents is None or parents <= 0:
            self.parents = pop_size // 2
        else:
            self.parents = parents
        self.weights = np.array([np.log((self.parents + 1) / i)
                                 for i in range(1, self.parents + 1)])
        self.weights /= self.weights.sum()

    def ask(self, pop_size):
        """
        Returns a list of candidates parameters
        """
        if self.antithetic:
            epsilon_half = np.random.randn(pop_size // 2, self.num_params)
            epsilon = np.concatenate([epsilon_half, - epsilon_half])

        else:
            epsilon = np.random.randn(pop_size, self.num_params)
        return self.mu * (epsilon * np.sqrt(self.cov) + 1)

    def tell(self, solutions, scores):
        """
        Updates the distribution
        """
        scores = np.array(scores)
        scores *= -1
        idx_sorted = np.argsort(scores)

        old_mu = self.mu
        self.damp = self.damp * self.tau + (1 - self.tau) * self.damp_limit
        self.mu = self.weights @ solutions[idx_sorted[:self.parents]]

        z = (solutions[idx_sorted[:self.parents]] - old_mu)
        self.cov = 1 / self.parents * self.weights @ (
            z * z) + self.damp * np.ones(self.num_params)

    def get_distrib_params(self):
        """
        Returns the parameters of the distrubtion:
        the mean and sigma
        """
        return np.copy(self.mu), np.copy(self.cov)