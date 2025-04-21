import numpy as np
from scipy.stats import t
import matplotlib.pyplot as plt

# 1. Generate synthetic training data
np.random.seed(42)
n_samples = 100
A_train = np.linspace(0, 10, n_samples).reshape(-1, 1)  # Single feature
true_slope = 2.5
b_train = true_slope * A_train.ravel() + np.random.normal(0, 1, n_samples)

# Add bias term (column of ones)
A_train = np.hstack([np.ones((n_samples, 1)), A_train])

# 2. Fit model using SVD
U, S, Vt = np.linalg.svd(A_train, full_matrices=False)
x = Vt.T @ np.diag(1/S) @ U.T @ b_train  # Model coefficients

# 3. Compute leverage for training data
H_train = U @ U.T
h_train = np.diag(H_train)

# 4. Prepare new data (including extrapolation points)
A_new = np.linspace(-2, 12, 50).reshape(-1, 1)  # Some points outside training range
A_new = np.hstack([np.ones((50, 1)), A_new])  # Add bias term

# 5. CORRECT leverage calculation for new data
def compute_leverage(A_new, U, S, Vt):
    V = Vt.T
    sigma_inv = np.diag(1/S)
    projection = A_new @ V @ sigma_inv
    return np.sum(projection**2, axis=1)

h_new = compute_leverage(A_new, U, S, Vt)

# 6. Make predictions and compute intervals
b_pred = A_new @ x
residuals = b_train - (A_train @ x)
RMSE = np.sqrt(np.mean(residuals**2))
dof = n_samples - A_train.shape[1]
t_critical = t.ppf(0.975, dof)

# Standard error and prediction intervals
SE_pred = RMSE * np.sqrt(1 + h_new)
PI = t_critical * SE_pred

# 7. Visualization
plt.figure(figsize=(10, 6))

# Training data
plt.scatter(A_train[:, 1], b_train, alpha=0.6, label='Training Data')

# Predictions
plt.plot(A_new[:, 1], b_pred, 'r-', label='Regression Line')

# Prediction intervals
plt.fill_between(
    A_new[:, 1], 
    b_pred - PI, 
    b_pred + PI, 
    color='orange', 
    alpha=0.3, 
    label='95% Prediction Interval'
)

plt.xlabel('Feature Value')
plt.ylabel('Target Value')
plt.title('Regression with Correct Leverage Calculation')
plt.legend()
plt.grid(True)
plt.show()

# Print some leverage values
print("Leverage values at key points:")
print(f"x = -2.0: h = {h_new[0]:.4f} (Extrapolation)")
print(f"x = 5.0: h = {h_new[30]:.4f} (Interpolation)") 
print(f"x = 12.0: h = {h_new[-1]:.4f} (Extrapolation)")