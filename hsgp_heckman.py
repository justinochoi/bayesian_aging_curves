import pandas as pd
import numpy as np
import pymc as pm
import pytensor.tensor as at
import arviz as az
import preliz as pz 
import matplotlib.pyplot as plt 
from pymc.gp.util import plot_gp_dist 

# data 
df = pd.read_csv("bat_speed_heckman.csv")
z = df["observed"].values  
bat_speed = df["avg_swing_speed"].values    
age = df["player_age"].values       

# observed outcomes 
y_obs = bat_speed[z==1]  
age_obs = age[z==1] 

def normal_cdf(x):
    return 0.5 * (1 + at.erf(x / at.sqrt(2.0)))

def normal_ccdf(x):
    return 0.5 * (1 - at.erf(x / at.sqrt(2.0)))

lower, upper = 10, 20 
ell_dist, ax = pz.maxent(
    pz.LogNormal(),
    lower=lower,
    upper=upper,
    mass=0.9,
)

with pm.Model() as bat_speed_heckman:
    # Priors
    ell = ell_dist.to_pymc("ell") 
    eta = pm.HalfNormal("eta", sigma=2.5)  
    cov = eta**2 * pm.gp.cov.ExpQuad(1, ls=ell)  
    gp = pm.gp.HSGP(m=[11], c=7, cov_func=cov) 
    f = gp.prior("f", X=age_obs[:, None])

    b0_sel = pm.Normal("b0_sel", mu=0, sigma=2)
    b1_sel = pm.Normal("b1_sel", mu=0, sigma=2)

    sigma_eps = pm.HalfNormal("sigma_eps", sigma=5)
    rho_unconstrained = pm.Normal("rho_unconstrained", mu=0, sigma=1)
    rho = pm.Deterministic("rho", pm.math.tanh(rho_unconstrained))
    
    # Linear predictors
    mu_age = pm.Deterministic("mu_age", f)  
    mu_z = pm.Deterministic("mu_z", b0_sel + b1_sel * age) 

    def logp_heckman(y_obs, z, mu_age, mu_z, sigma_eps, rho):
        # Selected
        u_out = (y_obs - mu_age) / sigma_eps
        arg_sel = (mu_z[z == 1] + rho * u_out) / at.sqrt(1.0 - rho**2)
        term1 = at.log(normal_cdf(arg_sel))
        term2 = -0.5 * (u_out**2) - at.log(sigma_eps * at.sqrt(2.0 * np.pi))
        ll_sel = term1 + term2

        # Non-selected
        ll_non = at.log(normal_ccdf(mu_z[z == 0]))

        return at.sum(ll_sel) + at.sum(ll_non)
    
    # Custom likelihood
    pm.Potential("likelihood", logp_heckman(y_obs, z, mu_age, mu_z, sigma_eps, rho))

pm.model_to_graphviz(bat_speed_heckman) 

with bat_speed_heckman: 
    heckman_trace = pm.sample(cores=4, random_seed=76, target_accept=0.95) 

az.summary(heckman_trace, var_names=["eta","ell","sigma_eps","rho","b1_sel"], round_to=2) 
az.plot_trace(heckman_trace)

age_grid = np.linspace(age.min(), age.max(), 200)[:, None]

with bat_speed_heckman:
    # Compute GP posterior mean and covariance at new points
    f_pred = gp.conditional("f_pred", Xnew=age_grid)

    # Draw samples from the posterior predictive
    f_pred_samples = pm.sample_posterior_predictive(
        heckman_trace, var_names=["f_pred"], random_seed=76
    )

f_pred_post = f_pred_samples.posterior_predictive["f_pred"].stack(sample=("chain", "draw")).values.T

hdi_bounds = az.hdi(f_pred_post, hdi_prob=0.95)
mean_pred = f_pred_post.mean(axis=0)

fig, ax = plt.subplots(figsize=(10, 4))
ax.plot(age_grid.squeeze(), mean_pred, color="navy", lw=2, label="Posterior Mean")
ax.fill_between(
    age_grid.squeeze(),
    hdi_bounds[:, 0],
    hdi_bounds[:, 1],
    color="navy",
    alpha=0.2,
    label="95% HDI"
)

plt.title("Bat Speed Aging Curve using GP + Heckman Correction")
plt.xlabel("Age")
plt.ylabel("Bat Speed (mph)")
plt.xticks(np.arange(20, 42, step=4))
plt.legend()
plt.show()