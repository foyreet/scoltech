import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import json
from sklearn.linear_model import LinearRegression

# Загрузка данных
df = pd.read_csv("calibration_data.csv")
M_raw = df[["raw_x", "raw_y", "raw_z"]].to_numpy()
M_ref = df[["ref_x", "ref_y", "ref_z"]].to_numpy()

# Калибровка

# 1. Вычисляем смещение (bias)
b = np.mean(M_raw - M_ref, axis=0)

# 2. Центрируем данные
M_raw_centered = M_raw - b

# 3. Масштабирование по осям для устойчивости
scale = np.std(M_raw_centered, axis=0)
M_raw_scaled = M_raw_centered / scale

# 4. Решаем M_raw_scaled @ A = M_ref
A_scaled, _, _, _ = np.linalg.lstsq(M_raw_scaled, M_ref, rcond=None)
A = A_scaled / scale[:, None]

# 5. Применяем линейную калибровку
M_calibrated = (M_raw - b) @ A

# Нелинейная коррекция
residuals = M_calibrated - M_ref
x, y, z = M_calibrated[:, 0], M_calibrated[:, 1], M_calibrated[:, 2]
X_nl = np.column_stack([x**2, y**2, z**2, x*y, x*z, y*z])
models = [LinearRegression().fit(X_nl, residuals[:, i]) for i in range(3)]
nonlinear_corr = np.column_stack([models[i].predict(X_nl) for i in range(3)])
M_calibrated -= nonlinear_corr

# Оценка качества
error = np.linalg.norm(M_calibrated - M_ref, axis=1)
print(f"Средняя ошибка: {np.mean(error):.3f}")
print(f"Максимальная ошибка: {np.max(error):.3f}")

axis_errors = np.abs(M_calibrated - M_ref)
print(f"Средняя ошибка по осям: {np.mean(axis_errors, axis=0)}")
print(f"Максимальная ошибка по осям: {np.max(axis_errors, axis=0)}")

# Визуализация
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

# Визуализация ошибки по углу
plt.figure()
plt.plot(df["angle_deg"], error)
plt.title("Ошибка по углу")
plt.xlabel("Угол (°)")
plt.ylabel("Ошибка (нТл)")
plt.grid(True)
plt.savefig("error_plot.png")
plt.close()

# Сохранение
calibration = {
    "bias": b.tolist(),
    "A": A.tolist(),
    "nonlinear_coeffs": [m.coef_.tolist() for m in models],
    "nonlinear_intercepts": [m.intercept_ for m in models]
}
with open("calibration_result.json", "w") as f:
    json.dump(calibration, f, indent=2)

print("Готово. Параметры сохранены в calibration_result.json")
