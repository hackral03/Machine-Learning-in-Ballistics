import numpy as np
import pandas as pd

data = pd.read_excel("SAP.xlsx", sheet_name="Sheet1")

rho_data  = data["Density"].values
sigma_data = data["Tensile Strength"].values
E_data     = data["Young's Modulus"].values
AD_data    = data["Areal Density"].values
V50_data   = data["V50"].values

def Ustar(sigma, E, rho):
    sigma_Pa = sigma * 88260 * rho
    E_Pa     = E     * 88260 * rho
    epsilon  = sigma_Pa / E_Pa
    c        = np.sqrt(E_Pa / rho)
    return (sigma_Pa * epsilon) / (2.0 * rho) * c

Ustar_data = []
for s, e, r in zip(sigma_data, E_data, rho_data):
  Ustar_data.append(Ustar(s, e, r))
Ustar_data = np.array(Ustar_data)
Ucbrt_data = Ustar_data ** (1.0 / 3.0)
V50star    = V50_data / Ucbrt_data
B_fit, logA = np.polyfit(np.log(AD_data), np.log(V50star), 1)
A_fit       = np.exp(logA)

V50_fitted  = A_fit * AD_data ** B_fit * Ucbrt_data

Prediction_type = int(input("Enter 1 to predict V50 or 2 to predict Areal Density: "))

sigma_new = float(input("Enter Tensile Strength (g/denier): "))
E_new     = float(input("Enter Young's Modulus (g/denier): "))
rho_new   = float(input("Enter Density (kg/m³): "))

Ustar_new = Ustar(sigma_new, E_new, rho_new)
Ucbrt_new = Ustar_new ** (1.0 / 3.0)

if Prediction_type == 1:
    AD_new   = float(input("Enter Areal Density (kg/m²): "))
    V50_pred = A_fit * (AD_new ** B_fit) * Ucbrt_new
    print(f"\nPredicted V50 = {V50_pred:.4f} m/s")

if Prediction_type == 2:
    V50_new = float(input("Enter V50 (m/s): "))
    AD_pred = (V50_new / (A_fit * Ucbrt_new)) ** (1.0 / B_fit)
    arg = V50_new / (A_fit * Ucbrt_new)
    print(f"\nPredicted Areal Density = {AD_pred:.4f} kg/m²")
