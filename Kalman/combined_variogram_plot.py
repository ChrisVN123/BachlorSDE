import numpy as np
import matplotlib.pyplot as plt

# --- Data ----------------------------------------------------------
# kalman_arima = np.array([
#     0.00180502, 0.00527584, 0.00859502, 0.0115511, 0.01536574, 0.01822247,
#     0.01988691, 0.02284669, 0.02610683, 0.02975721, 0.03605008, 0.04198879,
#     0.04837955, 0.051768,   0.06033683, 0.06862925, 0.07952207, 0.09734542,
#     0.120821,   0.1515595,  0.20623017, 0.27789125, 0.3160125
# ])

# ekf_arima = np.array([
#     0.00839499, 0.02674035, 0.04741965, 0.06728498, 0.08224002, 0.09127523,
#     0.10231449, 0.11407629, 0.12503129, 0.13619772, 0.15141494, 0.17942444,
#     0.22169197, 0.28292903, 0.35452641, 0.42538138, 0.4796247,  0.50405281,
#     0.47751418, 0.45878751, 0.41285431, 0.31835244, 0.28824299
# ])



garch_arima = np.array([4.16302790e-03,7.25562307e-03,9.38952845e-03,1.11751589e-02
,7.61680175e-03,7.20300222e-03,6.71722031e-03,4.08176078e-03
,4.06042221e-03,3.24439572e-03,5.93081743e-03,9.28470022e-03
,9.43156540e-03,9.42518268e-03,6.87082214e-03,6.36616491e-03
,9.58588705e-03,7.95790653e-03,1.03237193e-02,5.36566220e-03
,1.05147148e-03,1.99508175e-03,9.06997677e-04])

MLE_OU = np.array([
    0.0197217,  0.04699348, 0.07544843, 0.10091853, 0.1218283,  0.13778818,
    0.14909726, 0.15628625, 0.1605526,  0.16416484, 0.16940548, 0.17842681,
    0.1914987,  0.20625872, 0.2207194,  0.23306681, 0.24207547, 0.24661623,
    0.24681164, 0.24343238, 0.23847038, 0.23426148, 0.23473933
])

kalman_arima = np.array([
5.65639126e-03,1.48227726e-02,2.04041427e-02,2.08858999e-02
,1.95392367e-02,2.19993888e-02,2.33979999e-02,1.87939999e-02
,1.88069666e-02,1.87873213e-02,1.50581538e-02,7.53224998e-03
,5.57590908e-03,1.24223499e-02,2.13093888e-02,2.56951248e-02
,1.48059999e-02,1.47738332e-02,2.44908999e-02,2.26362500e-02
,1.02590000e-02,3.28850002e-03,7.20000069e-05,
])


ekf_arima = np.array([0.00346005,0.00930678,0.01298131,0.01331925,0.01288224,0.01458987
,0.01652494,0.01432375,0.01395111,0.01369044,0.01116181,0.00673481
,0.00544468,0.00905396,0.01371663,0.01450105,0.00729563,0.0080018
,0.0156023,0.01858326,0.01086897,0.00448439,0.00099094])


# --- Ensure equal length (optional safety) -------------------------
min_len = min(map(len, [kalman_arima, ekf_arima, garch_arima, MLE_OU]))
kalman_arima = kalman_arima[:min_len]
ekf_arima   = ekf_arima[:min_len]
garch_arima = garch_arima[:min_len]
MLE_OU      = MLE_OU[:min_len]

x = np.arange(1, min_len + 1)  # 1, 2, …, N

# --- Plot ----------------------------------------------------------
plt.figure(figsize=(10, 6))
plt.plot(x, kalman_arima, marker='o',  label='Kalman–OU')
plt.plot(x, ekf_arima,   marker='s',  label='EKF–CIR')
plt.plot(x, garch_arima, marker='^',  label='GARCH–ARIMA')
#plt.plot(x, MLE_OU,      marker='d',  label='MLE–OU')

plt.xlabel('Forecast horizon (1-hours steps)')
plt.ylabel('Semivariance')
plt.title('Model performance comparison of variograms')
plt.grid(True, linestyle='--', alpha=0.5)
plt.legend()
plt.tight_layout()
plt.show()
