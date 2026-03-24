import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import LeaveOneOut
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.linear_model import LogisticRegression

data = pd.read_excel("Aluminium FSP.xlsx", sheet_name="Sheet3")
X = data[["velocity", "Thickness"]].values
y_outcome = data["outcome"].values
scaler = StandardScaler()
X_s = scaler.fit_transform(X)
perf = data[data["outcome"] == 1].copy()
X_r = perf[["velocity", "Thickness"]].values
y_r = perf["residual velocity"].values
X_r_s = scaler.transform(X_r)

mlp_residual = MLPRegressor(hidden_layer_sizes=(16,16),activation="relu",solver="adam",alpha=0.000001,learning_rate_init=0.1,max_iter=50000,random_state=42,early_stopping=True,validation_fraction=0.15,n_iter_no_change=50)
mlp_residual.fit(X_r_s, y_r)

clf = LogisticRegression(penalty="l2",C=1.0,solver="lbfgs",max_iter=5000)
clf.fit(X_s, y_outcome)
loo = LeaveOneOut()
y_true, y_pred = [], []

for train_idx, test_idx in loo.split(X_r_s):
    X_tr, X_te = X_r_s[train_idx], X_r_s[test_idx]
    y_tr, y_te = y_r[train_idx], y_r[test_idx]
    mlp_tmp = MLPRegressor(hidden_layer_sizes=(16,16),activation="relu",solver="adam",alpha=0.000001,learning_rate_init=0.1,max_iter=50000,random_state=42,early_stopping=True,validation_fraction=0.15,n_iter_no_change=50)
    mlp_tmp.fit(X_tr, y_tr)
    y_hat = mlp_tmp.predict(X_te)
    y_true.append(y_te[0])
    y_pred.append(y_hat[0])

print("LOOCV MAE (residual velocity):", mean_absolute_error(y_true, y_pred))
print("LOOCV R²  (residual velocity):", r2_score(y_true, y_pred))

def predict_residual_velocity(velocity, thickness, outcome):
    if outcome == 0:
        return 0.0
    X_new = np.array([[velocity, thickness]])
    X_new_s = scaler.transform(X_new)
    vr = mlp_residual.predict(X_new_s)[0]
    return vr

def perforation_probability(velocity, thickness):
    X_new = np.array([[velocity, thickness]])
    X_new_s = scaler.transform(X_new)
    return clf.predict_proba(X_new_s)[0, 1]

def compute_V50(thickness, v_min=0, v_max=1000, n=1000):
    velocities = np.linspace(v_min, v_max, n)
    probs = np.array([perforation_probability(v, thickness) for v in velocities])
    idx = np.argmin(np.abs(probs - 0.5))
    return velocities[idx], probs[idx], velocities, probs
  
thickness = 27
velocity = 800
residual_velocity = predict_residual_velocity(velocity, thickness, 1)
V50, P50, vel, prob = compute_V50(thickness)
print(f"\nEstimated Residual velocity = {residual_velocity:.2f} m/s at thickness = {thickness:.2f} mm and impact velocity = {velocity:.2f}")
print(f"\nEstimated V50 = {V50:.2f} m/s at thickness = {thickness:.2f} mm")
print(f"Perforation probability at V50 ≈ {P50:.2f}")
v_range = np.linspace(perf["velocity"].min(), perf["velocity"].max(), 1000)
X_plot = np.column_stack([v_range,np.full_like(v_range, thickness)])
X_plot_s = scaler.transform(X_plot)
v_res_pred = mlp_residual.predict(X_plot_s)
v_res_pred = np.maximum.accumulate(v_res_pred)

plt.figure(figsize=(8, 5))
plt.scatter(perf["velocity"], y_r, color="black", label="Experimental data")
plt.plot(v_range, v_res_pred, label="MLP prediction")
plt.xlabel("Impact Velocity (m/s)")
plt.ylabel("Residual Velocity (m/s)")
plt.title("Residual Velocity Prediction")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

plt.figure(figsize=(7,5))
plt.plot(vel, prob, label="Perforation probability")
plt.axhline(0.5, linestyle="--", color="red", label=f"P = {P50:.1f}")
plt.axvline(V50, linestyle="--", color="black", label=f"V50 ≈ {V50:.2f} m/s")
plt.xlabel("Impact Velocity (m/s)")
plt.ylabel("Probability of Perforation")
plt.title("V50 Estimation")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
