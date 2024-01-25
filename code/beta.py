import ot
import numpy as np
from scipy.integrate import quad
from scipy.stats import norm
from scipy.stats import expon
from scipy.stats import gamma
from scipy.stats import poisson
from scipy.stats import beta
from tqdm import tqdm
from matplotlib import pyplot as mplt


###################################
### p, alpha, beta: free


# define exponential cost function
def expcost(x,y):
    a = 1
    p = 5
    return np.exp(a*np.absolute(x-y)**p)-1


# number of MC
T = 10000
# max number of sample
N = 100
# max number of grid between 0, 1
M = 1000

# simulation
expected_cost = [] # expected optimal cost
for n in tqdm(range(1,N+1)): # number of samples
    m = np.floor(M/n) # number of semi-grid
    m = int(m)

    # define beta distribution: X
    alpha = 2
    beta = 4
    quantile = [(j - 0.5) / (n*m) for j in range(1, n*m + 1)]
    X = poisson.ppf(quantile, alpha, beta)

    # monte carlo
    cost_n = 0
    for t in tqdm(range(1, T+1), desc=f'n: {n}'):

        # generate empirical beta distribution: Y
        Y = np.random.beta(alpha, beta, n)
        Y = np.repeat(Y, m)
        Y.sort()

        # compute cost
        cost_t = expcost(X, Y)/ (n*m)
        cost_t = np.sum(cost_t)
        cost_n += cost_t/T
    expected_cost.append(cost_n)

# normalize
expected_cost = np.log(expected_cost)
expected_cost = expected_cost - expected_cost[N-1]-np.log(N)/2

# plot

sample = [i for i in range(1, N+1)]
rate = -np.log(sample)/2

mplt.plot(sample, expected_cost, label='beta', color='red')
mplt.plot(sample, rate, label='rate', color='blue')

mplt.legend()

mplt.xlabel('sample')
mplt.ylabel('cost')
mplt.title('beta')

mplt.show()