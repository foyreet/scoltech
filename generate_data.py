import numpy as np
import pandas as pd
# Ввод параметров с клавиатуры

print("Введите bias (через пробел, 3 значения):")
bias = np.array(list(map(float, input().split())))

print("Введите значения матрицы A (в строках, 9 значений через пробел):")
A = np.array(list(map(float, input().split()))).reshape((3, 3))

print("Введите уровень шума (одно число):")
noise_std = float(input())

# Задание направления и величины геомагнитного поля

B_magnitude = 50 # нТл модуль вектора магнитного поля
theta_deg = 45 # угол между вектором поля и горизонтальной плоскостью
phi_deg = 90 # азимут: угол по горизонту. Восток

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
# Формула взяла из поворота вектора в 2D плоскости

M_ref = np.array([
    [
        B_real[0] * np.cos(a) - B_real[1] * np.sin(a),
        B_real[0] * np.sin(a) + B_real[1] * np.cos(a),
        B_real[2]
    ]
    for a in angles_rad
])

# 2. Искажения: bias + scale + soft-iron (матрица) + шум 

noise = np.random.normal(0, noise_std, M_ref.shape)  # шум

# Применяем искажения
M_raw = (M_ref @ A.T) + bias + noise

# 3. Сохраняем в CSV 

df = pd.DataFrame({
    "angle_deg": angles_deg,
    "raw_x": M_raw[:, 0],
    "raw_y": M_raw[:, 1],
    "raw_z": M_raw[:, 2],
    "ref_x": M_ref[:, 0],
    "ref_y": M_ref[:, 1],
    "ref_z": M_ref[:, 2],
})

df.to_csv("calibration_data.csv", index=False)
