"""
Created 08/02/2026
This file compares OLS and LASSO 
"""
import numpy as np
from sklearn.linear_model import LinearRegression, Lasso

# Generate toy regression dataset
np.random.seed(0)

n = 200  # nb of samples
d = 5    # nb of features

X = np.random.randn(n, d)
true_beta = np.array([1.5, -2.0, 0.0, 3.0, 0.0])  # some coefficients are zero
noise = np.random.randn(n) * 0.5

y = X @ true_beta + noise  # continuous target

# =====================
# OLS (Ordinary Least Squares)
# =====================
ols_model = LinearRegression()
ols_model.fit(X, y)

print("True beta:", true_beta)
print("\n--- OLS ---")
print("Coefficients:", ols_model.coef_)
print("Intercept:", ols_model.intercept_)

# =====================
# LASSO (L1 Regularization)
# =====================
lasso_model = Lasso(alpha=0.1)  # alpha controls regularization strength
lasso_model.fit(X, y)

print("\n--- LASSO (alpha=0.1) ---")
print("Coefficients:", lasso_model.coef_)
print("Intercept:", lasso_model.intercept_)

# Notice: LASSO may shrink some coefficients to exactly 0 (feature selection)