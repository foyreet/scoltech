import numpy as np
import pandas as pd
from scipy.spatial.transform import Rotation as R
# Ввод параметров с клавиатуры

print("Введите bias (через пробел, 3 значения):")
bias = np.array(list(map(float, input().split())))

print("Введите матрицу искажений A (в строках, 9 значений через пробел):")
A = np.array(list(map(float, input().split()))).reshape((3, 3))

print("Введите чувствительность по каждой оси (3 значения):")
sensitivity = np.array(list(map(float, input().split())))

print("Введите шум по каждой оси (σx σy σz):")
noise_std = np.array(list(map(float, input().split())))

print("Введите уровень шума эталонного магнитометра:")
ref_noise_std = float(input())

print("Введите порог чувствительности эталонного магнитометра:")
detection_threshold = float(input())

print("Нужно ли применять поворот осей? (yes/no):")
rotate_axes = input().strip().lower() == "yes"



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

# Поворот осей

if rotate_axes:
    rot_matrix = R.from_euler('xyz', [2, -1, 3], degrees=True).as_matrix()
    M_ref = M_ref @ rot_matrix.T

# 2. Искажения: bias + scale + soft-iron (матрица) + шум 

noise = np.random.normal(0, ref_noise_std, M_ref.shape)  # шум

# Добавляем шум, деградацию и порог
# генерация шума распределённого нормально
ref_noise = np.random.normal(0, ref_noise_std, M_ref.shape) 
# добавление шума к эталонным данным
M_ref_noisy = M_ref + ref_noise
# Убираем слабые сигналы
M_ref_final = np.where(np.abs(M_ref_noisy) < detection_threshold, 0, M_ref_noisy)

# Калибруемый магнитометр
M_raw = M_ref_final * sensitivity  # применяем чувствительность
M_raw = (M_raw @ A.T) + bias  # применяем искажения
noise = np.random.normal(0, noise_std, M_ref.shape)
M_raw += noise  # добавляем шум

# 3. Сохраняем в CSV 

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
