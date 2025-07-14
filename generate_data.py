import numpy as np
import pandas as pd
from scipy.spatial.transform import Rotation as R

# === ПАРАМЕТРЫ ===
bias = np.array([2.0, -1.5, 0.8])
A = np.array([[1.05, 0.02, 0.01], [0.01, 0.98, -0.03], [-0.01, 0.02, 1.02]])
sensitivity = np.array([0.95, 1.03, 1.00])
noise_std = np.array([0.3, 0.3, 0.3])
ref_noise_std = 0.1
detection_threshold = 1.0
rotate_axes = True

B_magnitude = 50  # нТл

# Сканируем по азимуту и углу наклона
phi_list = np.arange(0, 360, 20)
theta_list = np.arange(10, 80, 10)  # от 10° до 70°

records = []

for theta_deg in theta_list:
    for phi_deg in phi_list:
        theta = np.radians(theta_deg)
        phi = np.radians(phi_deg)

        # Истинный вектор поля
        B_real = np.array([
            B_magnitude * np.cos(theta) * np.cos(phi),
            B_magnitude * np.cos(theta) * np.sin(phi),
            B_magnitude * np.sin(theta)
        ])

        # Вращение на платформе вокруг Z (имитация поворота устройства)
        for angle_deg in range(0, 360, 30):
            angle_rad = np.radians(angle_deg)

            # Вращаем вокруг оси Z
            B_rot = np.array([
                B_real[0] * np.cos(angle_rad) - B_real[1] * np.sin(angle_rad),
                B_real[0] * np.sin(angle_rad) + B_real[1] * np.cos(angle_rad),
                B_real[2]
            ])

            # Поворот осей устройства (фиксированный)
            if rotate_axes:
                rot_matrix = R.from_euler('xyz', [2, -1, 3], degrees=True).as_matrix()
                B_rot = B_rot @ rot_matrix.T

            # Добавляем шум и отсечение
            ref = B_rot + np.random.normal(0, ref_noise_std, 3)
            ref = np.where(np.abs(ref) < detection_threshold, 0, ref)

            # Температура
            temperature = 20 + 10 * np.sin(np.radians(angle_deg))

            bias_temp_coef = np.array([0.01, -0.015, 0.005])
            sens_temp_coef = np.array([0.002, 0.001, -0.001])

            bias_temp = bias + (temperature - 25) * bias_temp_coef
            sensitivity_temp = sensitivity + (temperature - 25) * sens_temp_coef

            raw = ref * sensitivity_temp
            raw = raw @ A.T + bias_temp
            raw += np.random.normal(0, noise_std)

            records.append({
                "angle_deg": angle_deg,
                "temperature_C": temperature,
                "theta_deg": theta_deg,
                "phi_deg": phi_deg,
                "raw_x": raw[0],
                "raw_y": raw[1],
                "raw_z": raw[2],
                "ref_x": ref[0],
                "ref_y": ref[1],
                "ref_z": ref[2]
            })

# Сохраняем
df = pd.DataFrame(records)
df.to_csv("calibration_data.csv", index=False)
print(f"Генерация завершена: {len(df)} записей")
