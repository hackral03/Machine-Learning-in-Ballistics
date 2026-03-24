
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, Model
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt

def lambert_jonas(vi, a, p, v_bl):
    vi = np.asarray(vi)
    out = np.zeros_like(vi, dtype=float)
    mask = vi > v_bl
    out[mask] = a * (vi[mask]**p - v_bl**p)**(1.0/p)
    return out

def make_dataset(num_samples=5000):
    X = []
    y = []
    for _ in range(num_samples):
        v_bl = np.random.uniform(100, 1000)
        vi = np.linspace(0, 1000, 1000)
        vr = lambert_jonas(vi, 1.0, 3.0, v_bl)
        vr += np.random.normal(0, 5, size=vr.shape)
        vr = np.clip(vr, 0, None)
        vr_scaled = vr / 1000.0
        X.append(vr_scaled.reshape(-1, 1))
        y.append(v_bl)
    return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32)

def build_cnn():
    inp = layers.Input(shape=(1000, 1))
    x = layers.Conv1D(32, 5, activation='relu', padding='same')(inp)
    x = layers.MaxPooling1D(2)(x)
    x = layers.Conv1D(64, 5, activation='relu', padding='same')(x)
    x = layers.MaxPooling1D(2)(x)
    x = layers.Conv1D(128, 5, activation='relu', padding='same')(x)
    x = layers.GlobalAveragePooling1D()(x)
    x = layers.Dense(64, activation='relu')(x)
    x = layers.Dense(32, activation='relu')(x)
    out = layers.Dense(1)(x)
    model = Model(inp, out)
    model.compile(optimizer=tf.keras.optimizers.Adam(1e-3),loss='mse',metrics=['mae'])
    return model
X, y = make_dataset(5000)
model = build_cnn()
model.summary()
model.fit(X, y, epochs=20, batch_size=32, validation_split=0.2)
vi = np.linspace(0, 1000, 1000)
true_vbl = 750
vr = lambert_jonas(vi, 1.0, 3.0, true_vbl)
vr_scaled = (vr / 1000.0).reshape(-1, 1)
pred_vbl = model.predict(vr_scaled[np.newaxis, ...])[0, 0]
vr_pred = lambert_jonas(vi, 1.0, 3.0, pred_vbl)
print("\nTrue Vbl:", true_vbl)
print("Predicted Vbl:", pred_vbl)
print("\nImpact Velocity vs Residual Velocity Table")
print(f"{'Vi (m/s)':>10} | {'Vr_true (m/s)':>15} | {'Vr_pred (m/s)':>15}")
print("-"*70)
for i in range(0, len(vi), 20):
    print(f"{vi[i]:10.2f} | {vr[i]:15.2f} | {vr_pred[i]:15.2f}")
plt.figure(figsize=(8,6))
plt.scatter(vi, vr_pred, s=12, alpha=0.6, label="CNN Points (Vi, Vr)")
plt.plot(vi, vr, color='blue', linewidth=1.5, label="Ballistic Curve")
plt.axvline(true_vbl, color='green', linestyle='--', linewidth=2,label=f"True Vbl = {true_vbl:.1f} m/s")

plt.axvline(pred_vbl, color='red', linestyle='-', linewidth=2,label=f"Predicted Vbl = {pred_vbl:.1f} m/s")
plt.xlabel("Impact Velocity (m/s)")
plt.ylabel("Residual Velocity (m/s)")
plt.title("CNN Input Points + True & Predicted Ballistic Limit")
plt.grid(True)
plt.legend()
plt.show()

