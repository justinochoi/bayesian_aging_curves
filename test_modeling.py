import pandas as pd
import pymc as pm  
import arviz as az 
import matplotlib.pyplot as plt 
from pymc.gp.util import plot_gp_dist 

# vanilla GP with reduced sample size (down to 100 from ~1800)
df = pd.read_csv("bat_speed_df.csv").sample(100)

bat_speed = df["avg_swing_speed"].values 
age = df["player_age"].values
min_age = df["min_age"].values
max_age = df["max_age"].values
n_obs = len(bat_speed) 

with pm.Model() as bat_speed_gp: 
    ell = pm.Gamma("ell", alpha=2, beta=1) 
    eta = pm.HalfNormal("eta", sigma=2.5)  

    cov = eta**2 * pm.gp.cov.ExpQuad(1, ls=ell) 
    gp = pm.gp.Latent(cov_func=cov) 

    f = gp.prior("f", X=age[:, None])
    sigma = pm.HalfNormal("sigma", sigma=1) 
    y = pm.Normal("y", mu=f, sigma=sigma, observed=bat_speed) 

pm.model_to_graphviz(bat_speed_gp)

# about 1:30 with reduced dataset 
with bat_speed_gp: 
    trace = pm.sample(cores = 4, random_seed=76, nuts_sampler='pymc') 

# diagonistics
az.plot_trace(trace) 
az.summary(trace) 

# posterior predictive check 
with bat_speed_gp:
    pm.sample_posterior_predictive(trace, extend_inferencedata=True, random_seed=76)

az.plot_ppc(trace, num_pp_samples=100) 

# plot posterior bat speed against observed age 
fig = plt.figure(figsize=(10, 4))
ax = fig.gca()

f_post = az.extract(trace, var_names="f").transpose("sample", ...)
plot_gp_dist(ax, f_post, age) 
ax.plot(age, bat_speed, "ok", label = "Observed Bat Speed")