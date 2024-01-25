import ot
import numpy as np
from scipy.integrate import quad
from scipy.stats import norm
from scipy.stats import expon
from scipy.stats import gamma
from scipy.stats import geom
from tqdm import tqdm
from matplotlib import pyplot as mplt


###################################
### p=1, 1-e^{-8a} < q < 1


# define exponential cost function
def expcost(x,y):
    a = 1/(2**8) # 1-e^(-2^(-5)) ~ 0.031
    p = 1
    return np.exp(a*np.absolute(x-y)**p)-1


# number of MC
T = 10000
# max number of sample
N = 200
# max number of grid between 0, 1
M = 10000

# simulation
expected_cost = [] # expected optimal cost
for n in tqdm(range(1,N+1)): # number of samples
    m = np.floor(M/n) # number of semi-grid
    m = int(m)

    # define geometric distribution: X ~ (1-q)^{k-1}q, k \ge 1
    q = 0.9
    quantile = [(j - 0.5) / (n*m) for j in range(1, n*m + 1)]
    X = geom.ppf(quantile, q)

    # monte carlo
    cost_n = 0
    for t in tqdm(range(1, T+1), desc=f'n: {n}'):

        # generate empirical geometric distribution: Y
        Y = np.random.geometric(q, n)
        Y = np.repeat(Y, m)
        Y.sort()

        # compute cost
        cost_t = expcost(X, Y)/ (n*m)
        cost_t = np.sum(cost_t)
        cost_n += cost_t/T
    expected_cost.append(cost_n)

# memorize
np.save('geo_q9.npy', expected_cost)

# plot

sample = np.arange(1, N+1)
rate = sample**(-1/2)

mplt.loglog(sample, expected_cost, label='geometric', color='red')
mplt.loglog(sample, rate, label='rate', color='blue')
mplt.legend()

mplt.xlabel('sample')
mplt.ylabel('cost')
mplt.title('geometric')

mplt.show()