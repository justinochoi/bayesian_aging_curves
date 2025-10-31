import numpy as np
import pymc as pm
import pytensor.tensor as at
import arviz as az
from pymc.distributions import dist_math as dm

np.random.seed(123)

# Parameters
N = 500
p = 2  # outcome covariates
q = 2  # selection covariates
beta_true = np.array([1.0, -1.0])
gamma_true = np.array([0.5, 1.0])
sigma_eps_true = 1.0
rho_true = 0.6

# Simulate covariates
X = np.random.normal(size=(N, p))
W = np.random.normal(size=(N, q))

# Simulate correlated errors
cov = np.array([[1.0, rho_true * sigma_eps_true],
                [rho_true * sigma_eps_true, sigma_eps_true**2]])
errors = np.random.multivariate_normal([0, 0], cov, size=N)
u = errors[:,0]      # selection error
eps = errors[:,1]    # outcome error

# Selection equation
z_star = W @ gamma_true + u
z = (z_star > 0).astype(int)

# Outcome equation (observed only if z=1)
y = X @ beta_true + eps
y_obs = y[z == 1]
X_obs = X[z == 1, :]
W_sel = W[z == 1, :]
W_non = W[z == 0, :]

print(f"Selected observations: {len(y_obs)} / {N}")

def normal_cdf(x):
    return 0.5 * (1 + at.erf(x / at.sqrt(2.0)))

def normal_ccdf(x):
    # complementary: 1 - Phi(x)
    return 0.5 * (1 - at.erf(x / at.sqrt(2.0)))

with pm.Model() as heckman_model:
    # Priors
    beta = pm.Normal("beta", mu=0, sigma=10, shape=p)
    gamma = pm.Normal("gamma", mu=0, sigma=10, shape=q)
    sigma_eps = pm.HalfNormal("sigma_eps", sigma=10)
    rho_unconstrained = pm.Normal("rho_unconstrained", mu=0, sigma=1)
    rho = pm.Deterministic("rho", pm.math.tanh(rho_unconstrained))
    
    # Linear predictors
    w_dot_gamma = at.dot(W, gamma)
    x_dot_beta = at.dot(X_obs, beta)
    
    def logp_heckman(y_obs, x_dot_beta, w_dot_gamma, z, sigma_eps, rho):
        # For selected (z == 1)
        u_out = (y_obs - x_dot_beta) / sigma_eps
        w_sel = w_dot_gamma[z == 1]
        arg_sel = (w_sel + rho * u_out) / at.sqrt(1.0 - rho**2)
        term1 = at.log(normal_cdf(arg_sel))
        term2 = -0.5 * (u_out**2) - at.log(sigma_eps * at.sqrt(2.0 * np.pi))
        ll_sel = term1 + term2

        # For non‐selected (z == 0)
        w_non = w_dot_gamma[z == 0]
        ll_non = at.log(normal_ccdf(w_non))

        return at.sum(ll_sel) + at.sum(ll_non)

    
    pm.Potential("likelihood", logp_heckman(y_obs, x_dot_beta, w_dot_gamma, z, sigma_eps, rho))
    
    # Sampling
    trace = pm.sample(1000, tune=1000, target_accept=0.9, return_inferencedata=True)

az.summary(trace, var_names=["beta", "gamma", "sigma_eps", "rho"], round_to=2)
az.plot_trace(trace, var_names=["beta", "gamma", "sigma_eps", "rho"])
