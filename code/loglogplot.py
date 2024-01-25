import ot
import numpy as np
from scipy.integrate import quad
from scipy.stats import norm
from scipy.stats import expon
from scipy.stats import gamma
from scipy.stats import poisson
from scipy.stats import beta
from scipy.stats import weibull_min
from tqdm import tqdm
from matplotlib import pyplot as mplt

N = 200
sample = np.arange(1, N+1)
rate = sample**(-1/2)

#############################################
###### different distribution with p=1 ######
#############################################

fig, ax = mplt.subplots(2, 2, figsize=(12, 8))

###### geometric ###########
geo_q1 = np.load('geo_q1.npy')
geo_q5 = np.load('geo_q5.npy')
geo_q9 = np.load('geo_q9.npy')


ax[0,0].loglog(sample, geo_q1, label='$q=0.1$', color='red', linewidth=3)
ax[0,0].loglog(sample, geo_q5, label='$q=0.5$', color='orange', linewidth=3)
ax[0,0].loglog(sample, geo_q9, label='$q=0.9$', color='violet', linewidth=3)
ax[0,0].loglog(sample, 0.1*rate, linestyle='--', label='$0.1N^{-1/2}$', color='navy')

ax[0,0].set_xlabel('$N$')
ax[0,0].set_title('Geometric with different $q$ values')

ax[0,0].legend()


###### poission ###########
poi_lam1 = np.load('poi_lam1.npy')
poi_lam2 = np.load('poi_lam2.npy')
poi_lam24 = np.load('poi_lam24.npy')


ax[0,1].loglog(sample, poi_lam1, label='$\lambda=1$', color='red', linewidth=3)
ax[0,1].loglog(sample, poi_lam2, label='$\lambda=2$', color='orange', linewidth=3)
ax[0,1].loglog(sample, poi_lam24, label='$\lambda=2^4$', color='violet', linewidth=3)
ax[0,1].loglog(sample, 0.1*rate, linestyle='--', label='$0.1N^{-1/2}$', color='navy')

ax[0,1].set_xlabel('$N$')
ax[0,1].set_title('Poisson with different $\lambda$ values')

ax[0,1].legend()



###### gaussian ###########
g_sig05 = np.load('gaussian_sig05.npy')
g_sig1 = np.load('gaussian_sig1.npy')
g_sig24 = np.load('gaussian_sig24.npy')


ax[1,0].loglog(sample, g_sig05, label='$\sigma=0.5$', color='red', linewidth=3)
ax[1,0].loglog(sample, g_sig1, label='$\sigma=1$', color='orange', linewidth=3)
ax[1,0].loglog(sample, g_sig24, label='$\sigma=2^4$', color='violet', linewidth=3)
ax[1,0].loglog(sample, 0.1*rate, linestyle='--', label='$0.1N^{-1/2}$', color='navy')

ax[1,0].set_xlabel('$N$')
ax[1,0].set_title('Gaussian with different $\sigma$ values')

ax[1,0].legend()

###### weibull ###########
wb_c1 = np.load('weibull_c1.npy')
wb_c3 = np.load('weibull_c3.npy')
wb_c5 = np.load('weibull_c5.npy')


ax[1,1].loglog(sample, wb_c1, label='$c=1$', color='red', linewidth=3)
ax[1,1].loglog(sample, wb_c3, label='$c=3$', color='orange', linewidth=3)
ax[1,1].loglog(sample, wb_c5, label='$c=5$', color='violet', linewidth=3)
ax[1,1].loglog(sample, 0.1*rate, linestyle='--', label='$0.1N^{-1/2}$', color='navy')

ax[1,1].set_xlabel('$N$')
ax[1,1].set_title('Weibull with different $c$ values')

ax[1,1].legend()

##### plot #######
mplt.tight_layout()
mplt.show()


# ###############################################
# ###### same distribution but different p ######
# ###############################################

# fig, ax = mplt.subplots(1, 2, figsize=(12, 6))  

# ###### gaussian ###########
# g_p1 = np.load('gaussian_sig1.npy')
# g_p15 = np.load('gaussian_p15.npy')
# g_p2 = np.load('gaussian_p2.npy')

# ax[0].loglog(sample, g_p1, label='$p=1$', color='red', linewidth=2)
# ax[0].loglog(sample, g_p15, label='$p=1.5$', color='orange', linewidth=2)
# ax[0].loglog(sample, g_p2, label='$p=2$', color='violet', linewidth=2)
# ax[0].loglog(sample, 0.01*rate, linestyle='--', label='$0.01N^{-1/2}$', color='navy')


# ax[0].set_xlabel('$N$')
# ax[0].set_title('Gaussian with different $p$ values')

# ax[0].legend()

# #### weibull #####
# wb_1 = np.load('weibull_c3.npy')
# wb_2 = np.load('weibull_p2.npy')
# wb_3 = np.load('weibull_p3.npy')

# ax[1].loglog(sample, wb_1, label='$p=1$', color='red', linewidth=2)
# ax[1].loglog(sample, wb_2, label='$p=2$', color='orange', linewidth=2)
# ax[1].loglog(sample, wb_3, label='$p=3$', color='violet', linewidth=2)
# ax[1].loglog(sample, 0.01*rate, linestyle='--', label='$0.01N^{-1/2}$', color='navy')

# ax[1].set_xlabel('$N$')
# ax[1].set_title('Weibull with different $p$ values')

# ax[1].legend()

# ##### plot #######
# mplt.tight_layout()
# mplt.show()