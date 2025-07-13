import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import json

# === ЗАГРУЗКА ДАННЫХ ===
df = pd.read_csv("calibration_data.csv")

M_raw = df[["raw_x", "raw_y", "raw_z"]].to_numpy()
M_ref = df[["ref_x", "ref_y", "ref_z"]].to_numpy()

# === КАЛИБРОВКА ===

# 1. Вычисляем смещение (bias)
b = np.mean(M_raw - M_ref, axis=0)

# 2. Центрируем данные
M_raw_centered = M_raw - b

# 3. Ищем матрицу A: решаем M_raw_centered @ A = M_ref
A, _, _, _ = np.linalg.lstsq(M_raw_centered, M_ref, rcond=None)

# 4. Применяем калибровку
M_calibrated = (M_raw - b) @ A


# === ОЦЕНКА КАЧЕСТВА ===

error = np.linalg.norm(M_calibrated - M_ref, axis=1)
print(f"Средняя ошибка: {np.mean(error):.3f}")
print(f"Максимальная ошибка: {np.max(error):.3f}")

# === ВИЗУАЛИЗАЦИЯ ===
plt.figure(figsize=(10, 5))
for i, label in enumerate(['X', 'Y', 'Z']):
    plt.subplot(1, 3, i + 1)
    plt.plot(M_ref[:, i], label=f"Ref {label}")
    plt.plot(M_calibrated[:, i], '--', label=f"Calibrated {label}")
    plt.title(f"Аппроксимация по оси {label}")
    plt.legend()
plt.tight_layout()
plt.show()

# === СОХРАНЕНИЕ ===
calibration = {
    "bias": b.tolist(),
    "A": A.tolist()
}
with open("calibration_result.json", "w") as f:
    json.dump(calibration, f, indent=2)

print("Готово. Параметры сохранены в calibration_result.json")
