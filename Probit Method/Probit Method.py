
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
import statsmodels.api as sm
from statsmodels.discrete.discrete_model import Probit
import pandas as pd

data_path = "Armour_UHMWPE.xlsx"
df = pd.read_excel(data_path,sheet_name='Panel 1')

velocities = df["Velocity(m/s)"].to_numpy(dtype=float)
outcomes   = df["Outcome"].to_numpy(dtype=int)

X = sm.add_constant(velocities)
model = Probit(outcomes, X)
result = model.fit(disp=False)

a, b = result.params
cov = result.cov_params()

V10 = (norm.ppf(0.1)- a)/ b
V50 = - a/ b
V99 = (norm.ppf(0.99)- a)/ b

def gradient(zp, a, b, cov):
    dVa = -1 / b
    dVb = -(zp - a) / (b**2)
    grad = np.array([dVa, dVb])
    return grad.dot(cov).dot(grad.T)

std_V10 = np.sqrt(gradient(norm.ppf(0.1), a, b, cov))
std_V50 = np.sqrt(gradient(0, a, b, cov))
std_V99 = np.sqrt(gradient(norm.ppf(0.99), a, b, cov))

print("\n Ballistic Limits with Uncertainty ")
print(f"α = {a:.4f}, β = {b:.6f}")
print(f"V10 = {V10:.2f} ± {std_V10:.2f} m/s")
print(f"V50 = {V50:.2f} ± {std_V50:.2f} m/s")

v_grid = np.linspace(500, 1500, 100)
X_grid = sm.add_constant(v_grid)
p_fit = result.predict(X_grid)

plt.figure(figsize=(7,5))
plt.plot(v_grid, p_fit, color="black", linewidth=2, label="Probit fit")
plt.axvline(V50, color="orange", linestyle="--", label="V50")
plt.xlabel("Velocity (m/s)")
plt.ylabel("Probability of perforation")
plt.title("Probit Fit (Statsmodels): Probability of Perforation vs Velocity")
plt.legend()
plt.grid(True, linestyle="--", alpha=0.6)
plt.ylim(-0.05, 1.05)
plt.show()
