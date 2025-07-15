import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import json
import os
from sklearn.linear_model import Ridge

# === ЗАГРУЗКА ДАННЫХ ===
df = pd.read_csv("calibration_data.csv")
M_raw = df[["raw_x", "raw_y", "raw_z"]].to_numpy()
M_ref = df[["ref_x", "ref_y", "ref_z"]].to_numpy()
angles = df["angle_deg"].to_numpy()

# === БЛОК ТЕМПЕРАТУРНОЙ КОМПЕНСАЦИИ УБРАН ===
# Используем M_raw без коррекции температуры
M_raw_corrected = M_raw.copy()

# === КАЛИБРОВКА ===
b = np.mean(M_raw_corrected - M_ref, axis=0)
M_raw_centered = M_raw_corrected - b
scale = np.std(M_raw_centered, axis=0)
M_raw_scaled = M_raw_centered / scale
A_scaled, _, _, _ = np.linalg.lstsq(M_raw_scaled, M_ref, rcond=None)
A = A_scaled / scale[:, None]
M_calibrated = (M_raw_corrected - b) @ A

# === НЕЛИНЕЙНАЯ КОРРЕКЦИЯ ТОЛЬКО ДЛЯ ОСИ Z ===
x, y, z = M_calibrated[:, 0], M_calibrated[:, 1], M_calibrated[:, 2]

X_nl_z = np.column_stack([
    x, y, z,
    x**2, y**2, z**2,
    x*y, x*z, y*z,
    x**3, y**3, z**3,
    x**2 * y, x * y**2, x * z**2, y * z**2
])

residual_z = M_calibrated[:, 2] - M_ref[:, 2]
model_z = Ridge(alpha=0.1)
model_z.fit(X_nl_z, residual_z)
nonlinear_corr_z = model_z.predict(X_nl_z)
M_calibrated[:, 2] -= nonlinear_corr_z

# === ОЦЕНКА ===
error = np.linalg.norm(M_calibrated - M_ref, axis=1)
axis_errors = np.abs(M_calibrated - M_ref)
print(f"Средняя ошибка: {np.mean(error):.3f}")
print(f"Максимальная ошибка: {np.max(error):.3f}")
print(f"Средняя ошибка по осям: {np.mean(axis_errors, axis=0)}")
print(f"Максимальная ошибка по осям: {np.max(axis_errors, axis=0)}")

# === ВИЗУАЛИЗАЦИЯ ===
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

plt.figure()
plt.plot(angles, error)
plt.title("Ошибка вектора по углу")
plt.xlabel("Угол (°)")
plt.ylabel("Ошибка (нТл)")
plt.grid(True)
plt.savefig("error_vs_angle.png")
plt.close()

# === График ошибки по температуре убран ===

# === СОХРАНЕНИЕ ===
calibration = {
    "bias": b.tolist(),
    "A": A.tolist(),
    "nonlinear_z_model": {
        "coefficients": model_z.coef_.tolist(),
        "intercept": model_z.intercept_,
        "features": ["x", "y", "z",
            "x^2", "y^2", "z^2",
            "xy", "xz", "yz",
            "x^3", "y^3", "z^3",
            "x^2*y", "x*y^2", "x*z^2", "y*z^2"],
        "model_type": "Ridge(alpha=0.1)"
    }
    # Убраны параметры, связанные с температурой
}

with open("calibration_result.json", "w") as f:
    json.dump(calibration, f, indent=2)

print("Готово. Параметры сохранены в calibration_result.json")

os.makedirs("figures", exist_ok=True)

def plot_ellipses(m_raw, m_calibrated, fname_prefix="figures/ellipse"):
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    titles = ['XY', 'YZ', 'ZX']
    pairs = [(0, 1), (1, 2), (2, 0)]

    for ax, (i, j), title in zip(axes, pairs, titles):
        ax.scatter(m_raw[:, i], m_raw[:, j], alpha=0.3, s=5, label='Raw')
        ax.scatter(m_calibrated[:, i], m_calibrated[:, j], alpha=0.3, s=5, label='Calibrated')
        ax.set_xlabel(['X', 'Y', 'Z'][i])
        ax.set_ylabel(['X', 'Y', 'Z'][j])
        ax.set_title(f"{title} projection")
        ax.axis('equal')
        ax.legend()

    plt.tight_layout()
    plt.savefig(f"{fname_prefix}_projections.png")
    plt.close()

plot_ellipses(M_raw, M_calibrated)
