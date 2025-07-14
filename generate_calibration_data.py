import numpy as np
import pandas as pd

# 1. Эталонные данные 
angles_deg = np.arange(0, 360, 10)  
angles_rad = np.radians(angles_deg)

# Эталон: вращение в горизонтальной плоскости + постоянное Z
ref_x = np.cos(angles_rad) * 50
ref_y = np.sin(angles_rad) * 50
ref_z = np.ones_like(ref_x) * 30
M_ref = np.vstack((ref_x, ref_y, ref_z)).T

# 2. Искажения: bias + scale + soft-iron (матрица) + шум 

bias = np.array([5, -3, 2]) 
A = np.array([
    [1.1, 0.05, 0.0],
    [0.02, 0.95, 0.01],
    [0.0, 0.0, 1.05]
])  

noise = np.random.normal(0, 0.3, M_ref.shape)  # шум

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
