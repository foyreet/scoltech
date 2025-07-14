import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import json
from sklearn.linear_model import LinearRegression

# === ЗАГРУЗКА ДАННЫХ ===
df = pd.read_csv("calibration_data.csv")
M_raw = df[["raw_x", "raw_y", "raw_z"]].to_numpy()
M_ref = df[["ref_x", "ref_y", "ref_z"]].to_numpy()
temperatures = df["temperature_C"].to_numpy()
angles = df["angle_deg"].to_numpy()

# === ТЕМПЕРАТУРНАЯ КОМПЕНСАЦИЯ ===
# Используем те же коэффициенты, что и в generate_data.py
bias_temp_coef = np.array([0.01, -0.015, 0.005])
sens_temp_coef = np.array([0.002, 0.001, -0.001])
T_ref = 25.0

bias_temp = (temperatures[:, None] - T_ref) * bias_temp_coef
sensitivity_temp = 1.0 + (temperatures[:, None] - T_ref) * sens_temp_coef

# Убираем температурное влияние
M_raw_corrected = (M_raw - bias_temp) / sensitivity_temp

# === КАЛИБРОВКА ===

# 1. Bias
b = np.mean(M_raw_corrected - M_ref, axis=0)

# 2. Центрирование
M_raw_centered = M_raw_corrected - b

# 3. Нормализация
scale = np.std(M_raw_centered, axis=0)
M_raw_scaled = M_raw_centered / scale

# 4. Матрица искажений A
A_scaled, _, _, _ = np.linalg.lstsq(M_raw_scaled, M_ref, rcond=None)
A = A_scaled / scale[:, None]

# 5. Линейная калибровка
M_calibrated = (M_raw_corrected - b) @ A

# === НЕЛИНЕЙНАЯ КОРРЕКЦИЯ ===
residuals = M_calibrated - M_ref
x, y, z = M_calibrated[:, 0], M_calibrated[:, 1], M_calibrated[:, 2]
X_nl = np.column_stack([x**2, y**2, z**2, x*y, x*z, y*z])
models = [LinearRegression().fit(X_nl, residuals[:, i]) for i in range(3)]
nonlinear_corr = np.column_stack([models[i].predict(X_nl) for i in range(3)])
M_calibrated -= nonlinear_corr

# === ОЦЕНКА КАЧЕСТВА ===
error = np.linalg.norm(M_calibrated - M_ref, axis=1)
axis_errors = np.abs(M_calibrated - M_ref)

print(f"Средняя ошибка: {np.mean(error):.3f}")
print(f"Максимальная ошибка: {np.max(error):.3f}")
print(f"Средняя ошибка по осям: {np.mean(axis_errors, axis=0)}")
print(f"Максимальная ошибка по осям: {np.max(axis_errors, axis=0)}")

# === ВИЗУАЛИЗАЦИЯ ===

# Графики откалиброванных vs эталонных
plt.figure(figsize=(10, 5))
for i, label in enumerate(['X', 'Y', 'Z']):
    plt.subplot(1, 3, i + 1)
    plt.plot(M_ref[:, i], label=f"Ref {label}")
    plt.plot(M_calibrated[:, i], '--', label=f"Calibrated {label}")
    plt.title(f"Аппроксимация по оси {label}")
    plt.legend()
plt.tight_layout()
plt.savefig("calibration_plot.png")
plt.close()

# График ошибки по углу
plt.figure()
plt.plot(angles, error)
plt.title("Ошибка вектора по углу")
plt.xlabel("Угол (°)")
plt.ylabel("Ошибка (нТл)")
plt.grid(True)
plt.savefig("error_vs_angle.png")
plt.close()

# График ошибки по температуре
plt.figure()
plt.plot(temperatures, error, 'o-')
plt.title("Ошибка вектора по температуре")
plt.xlabel("Температура (°C)")
plt.ylabel("Ошибка (нТл)")
plt.grid(True)
plt.savefig("error_vs_temperature.png")
plt.close()

# === СОХРАНЕНИЕ РЕЗУЛЬТАТОВ ===
calibration = {
    "bias": b.tolist(),
    "A": A.tolist(),
    "nonlinear_coeffs": [m.coef_.tolist() for m in models],
    "nonlinear_intercepts": [m.intercept_ for m in models],
    "temperature_ref": T_ref,
    "bias_temp_coef": bias_temp_coef.tolist(),
    "sens_temp_coef": sens_temp_coef.tolist()
}

with open("calibration_result.json", "w") as f:
    json.dump(calibration, f, indent=2)

print("Готово. Параметры сохранены в calibration_result.json")
