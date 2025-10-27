import pandas as pd
import pymc as pm  
import numpy as np
import arviz as az 
import matplotlib.pyplot as plt 
from pymc.gp.util import plot_gp_dist 

df = pd.read_csv("bat_speed_df.csv") 

bat_speed = df["avg_swing_speed"].values 
age = df["player_age"].values
min_age = df["min_age"].values
max_age = df["max_age"].values
n_obs = len(bat_speed) 

with pm.Model() as bat_speed_hsgp: 
    ell = pm.Gamma("ell", alpha=2, beta=1) 
    eta = pm.HalfNormal("eta", sigma=2.5)  

    cov = eta**2 * pm.gp.cov.ExpQuad(1, ls=ell)  
    gp = pm.gp.HSGP(m=[10], c=2, cov_func=cov, parametrization="centered") 

    f = gp.prior("f", X=age[:, None])
    sigma = pm.HalfNormal("sigma", sigma=1) 
    y = pm.Normal("y", mu=f, sigma=sigma, observed=bat_speed) 

pm.model_to_graphviz(bat_speed_hsgp) 

with bat_speed_hsgp: 
    hsgp_trace = pm.sample(cores = 4, random_seed = 76, nuts_sampler="numpyro", target_accept=0.95)

az.summary(hsgp_trace, var_names=["eta", "ell", "sigma"], round_to=2)
az.plot_trace(hsgp_trace) 

with bat_speed_hsgp:
    pm.sample_posterior_predictive(hsgp_trace, extend_inferencedata=True, random_seed=76)

az.plot_ppc(hsgp_trace, num_pp_samples=200)  

fig = plt.figure(figsize=(10, 4))
ax = fig.gca() 

f_post = az.extract(hsgp_trace, var_names="f").transpose("sample", ...)
plot_gp_dist(ax, f_post, age) 
plt.title("Bat Speed Aging Curve using Gaussian Process")
plt.xlabel("Age")
plt.ylabel("Bat Speed (mph)")
plt.xticks(np.arange(20, 42, step=4)) 
