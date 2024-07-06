# -*- coding: utf-8 -*-
"""
Created on Tue Jun 25 15:07:04 2024

@author: andre
"""
import numpy as np


# Initialize variables
nsteps = 252 # sample days 252 trading days in total
npaths = 20000
T = 1.0 # expiry of 
dt = T / nsteps
r = 0.2
sigma = 0.6
S0 = 100

# Initialize matrices
X_vector = np.zeros((nsteps, npaths))
S_vector = np.zeros((nsteps, npaths))

# Generate all random numbers at once
rng = np.random.default_rng()
random_numbers = rng.normal(size=(nsteps-1, npaths))

# Set initial values
X_vector[0, :] = 0
S_vector[0, :] = S0

# Generate paths for underlying following log-normal random walk
dX = (r - 0.5 * sigma ** 2) * dt + sigma * random_numbers * np.sqrt(dt)
X_vector[1:, :] = np.cumsum(dX, axis=0)
S_vector = S0 * np.exp(X_vector)

# Create observation grid by taking every 20th entry
obs_grid = np.arange(0, nsteps + 1, 20)
S_obs = S_vector[obs_grid, :]

# Calculate log returns for observation grid
log_returns_obs = np.diff(np.log(S_obs), axis=0)

# Calculate annualized realized variance for each path
realised_variances_obs = 252 * np.var(log_returns_obs, axis=0)

# Calculate the mean variance
mean_variance = np.mean(realised_variances_obs)
print("Variance Swap Price: ", mean_variance)
