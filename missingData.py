import pandas as pd
import datetime as dt
from datetime import datetime as dt
import numpy as np
from functools import reduce

# generate truth values from mu and sigma.
np.random.seed(1024)
mu = np.array([1, 2, 6])
Sigma = np.array([[118, 62, 44], [62, 49, 17], [44, 17, 21]])
n = 13  #number of data points.
X_truth = np.random.multivariate_normal(mu, Sigma, n)


def simulate_nan(X, nan_rate):
    # Create C matrix; entry is False if missing, and True if observed
    X_complete = X.copy()
    nr, nc = X_complete.shape
    C = np.random.random(nr * nc).reshape(nr, nc) > nan_rate

    # Check for which i's we have all components become missing
    checker = np.where(sum(C.T) == 0)[0]
    if len(checker) == 0:
        # Every X_i has at least one component that is observed,
        # which is what we want
        X_complete[C == False] = np.nan
    else:
        # Otherwise, randomly "revive" some components in such X_i's
        for index in checker:
            reviving_components = np.random.choice(
                nc,
                int(np.ceil(nc * np.random.random())),
                replace=False
            )
            C[index, np.ix_(reviving_components)] = True
        X_complete[C == False] = np.nan

    result = {
        'X': X_complete,
        'C': C,
        'nan_rate': nan_rate,
        'nan_rate_actual': np.sum(C == False) / (nr * nc)
    }

    return result


result = simulate_nan(X_truth, nan_rate = .4)   # missing data ratio to the truth.
X = result['X'].copy()
print (result['nan_rate_actual'])

def impute_em(X, max_iter=3000, eps=1e-08):
    nr, nc = X.shape
    C = np.isnan(X) == False

    # Collect M_i and O_i's
    one_to_nc = np.arange(1, nc + 1, step=1)
    M = one_to_nc * (C == False) - 1
    O = one_to_nc * C - 1

    # Generate Mu_0 and Sigma_0
    Mu = np.nanmean(X, axis=0)
    observed_rows = np.where(np.isnan(sum(X.T)) == False)[0]
    S = np.cov(X[observed_rows,].T)
    if np.isnan(S).any():
        S = np.diag(np.nanvar(X, axis=0))

    # Start updating
    Mu_tilde, S_tilde = {}, {}
    X_tilde = X.copy()
    no_conv = True
    iteration = 0
    while no_conv and iteration < max_iter:
        for i in range(nr):
            S_tilde[i] = np.zeros(nc ** 2).reshape(nc, nc)
            if set(O[i,]) != set(one_to_nc - 1):  # missing component exists
                M_i, O_i = M[i,][M[i,] != -1], O[i,][O[i,] != -1]
                S_MM = S[np.ix_(M_i, M_i)]
                S_MO = S[np.ix_(M_i, O_i)]
                S_OM = S_MO.T
                S_OO = S[np.ix_(O_i, O_i)]
                Mu_tilde[i] = Mu[np.ix_(M_i)] + S_MO @ np.linalg.inv(S_OO) @ (X_tilde[i, O_i] - Mu[np.ix_(O_i)])
                X_tilde[i, M_i] = Mu_tilde[i]
                S_MM_O = S_MM - S_MO @ np.linalg.inv(S_OO) @ S_OM
                S_tilde[i][np.ix_(M_i, M_i)] = S_MM_O
        Mu_new = np.mean(X_tilde, axis=0)

        S_new = np.cov(X_tilde.T, bias=1) + reduce(np.add, S_tilde.values()) / nr

        no_conv = np.linalg.norm(Mu - Mu_new) >= eps or np.linalg.norm(S - S_new, ord=2) >= eps
        Mu = Mu_new
        S = S_new
        iteration += 1

    result = {
        'mu': Mu,
        'Sigma': S,
        'X_imputed': X_tilde,
        'C': C,
        'iteration': iteration
    }

    return result


