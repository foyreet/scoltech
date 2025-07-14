import numpy as np
import pandas as pd
from scipy.spatial.transform import Rotation as R

# Константные параметры

bias = np.array([2.0, -1.5, 0.8])  # Смещение (bias)

A = np.array([                   # Матрица искажений (soft-iron)
    [1.05, 0.02, 0.01],
    [0.01, 0.98, -0.03],
    [-0.01, 0.02, 1.02]
])

sensitivity = np.array([0.95, 1.03, 1.00])  # Чувствительность по осям

noise_std = np.array([0.3, 0.3, 0.3])       # Шум по каждой оси

ref_noise_std = 0.1                         # Шум эталонного магнитометра

detection_threshold = 1.0                   # Порог чувствительности эталонного

rotate_axes = True                          # Применять поворот осей

# Задание направления и величины геомагнитного поля

B_magnitude = 49.5 * 1000 # нТл модуль вектора магнитного поля
theta_deg = 70.5 # угол между вектором поля и горизонтальной плоскости
phi_deg = 11.98 # азимут: угол по горизонту. Восток

theta = np.radians(theta_deg)
phi = np.radians(phi_deg)

B_real = np.array([
    B_magnitude * np.cos(theta) * np.cos(phi),
    B_magnitude * np.cos(theta) * np.sin(phi),
    B_magnitude * np.sin(theta)
])

# Эталонные данные 
angles_deg = np.arange(0, 360, 10)  
angles_rad = np.radians(angles_deg)

# Симуляция поворота на поворотном столе 
# Вращаем B_real вокруг оси Z (плоскость вращения)

M_ref = np.array([
    [
        B_real[0] * np.cos(a) - B_real[1] * np.sin(a),
        B_real[0] * np.sin(a) + B_real[1] * np.cos(a),
        B_real[2]
    ]
    for a in angles_rad
])

# Поворот осей

if rotate_axes:
    rot_matrix = R.from_euler('xyz', [2, -1, 3], degrees=True).as_matrix()
    M_ref = M_ref @ rot_matrix.T

# Добавляем шум к эталонным данным и порог детекции

ref_noise = np.random.normal(0, ref_noise_std, M_ref.shape)
M_ref_noisy = M_ref + ref_noise
M_ref_final = np.where(np.abs(M_ref_noisy) < detection_threshold, 0, M_ref_noisy)

# === Температурная модель убрана ===
# Используем константный bias и sensitivity без зависимости от температуры

bias_temp = np.tile(bias, (len(angles_deg), 1))            # просто bias
sensitivity_temp = np.tile(sensitivity, (len(angles_deg), 1))  # просто sensitivity

# === Калибруемый магнитометр ===
M_raw = M_ref_final * sensitivity_temp
M_raw = (M_raw @ A.T) + bias_temp

noise = np.random.normal(0, noise_std, M_ref.shape)
M_raw += noise

# 3. Сохраняем в CSV без температуры

df = pd.DataFrame({
    "angle_deg": angles_deg,
    "raw_x": M_raw[:, 0],
    "raw_y": M_raw[:, 1],
    "raw_z": M_raw[:, 2],
    "ref_x": M_ref_final[:, 0],
    "ref_y": M_ref_final[:, 1],
    "ref_z": M_ref_final[:, 2],
})
df.to_csv("calibration_data.csv", index=False)
