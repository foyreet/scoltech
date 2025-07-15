
import numpy as np
import matplotlib.pyplot as plt
import os
from mpl_toolkits.mplot3d import Axes3D

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

def plot_3d_clouds(m_raw, m_calibrated, fname="figures/cloud_3d.png"):
    fig = plt.figure(figsize=(12, 5))

    ax1 = fig.add_subplot(121, projection='3d')
    ax1.scatter(*m_raw.T, alpha=0.3, s=5)
    ax1.set_title("Raw measurements")

    ax2 = fig.add_subplot(122, projection='3d')
    ax2.scatter(*m_calibrated.T, alpha=0.3, s=5)
    ax2.set_title("Calibrated measurements")

    plt.tight_layout()
    plt.savefig(fname)
    plt.close()

def angular_error(m_cal, m_ref):
    cos_angles = np.sum(m_cal * m_ref, axis=1) / (
        np.linalg.norm(m_cal, axis=1) * np.linalg.norm(m_ref, axis=1)
    )
    cos_angles = np.clip(cos_angles, -1, 1)
    angles_deg = np.arccos(cos_angles) * 180 / np.pi
    return angles_deg

def plot_angular_error(angles_deg, angle_labels, fname="figures/angular_error.png"):
    plt.figure()
    plt.plot(angle_labels, angles_deg)
    plt.xlabel("Угол (энкодер)")
    plt.ylabel("Ошибка направления (°)")
    plt.title("Угловая ошибка между калиброванным и эталонным вектором")
    plt.grid(True)
    plt.savefig(fname)
    plt.close()

def plot_vector_norms(m_raw, m_calibrated, m_ref, fname="figures/vector_norms.png"):
    plt.figure()
    plt.plot(np.linalg.norm(m_raw, axis=1), label="Raw")
    plt.plot(np.linalg.norm(m_calibrated, axis=1), label="Calibrated")
    plt.hlines(np.linalg.norm(m_ref, axis=1).mean(), 0, len(m_raw), colors='k', linestyles='dashed', label='Ref mean')
    plt.title("Норма магнитного вектора")
    plt.ylabel("нТл")
    plt.legend()
    plt.grid(True)
    plt.savefig(fname)
    plt.close()

def plot_error_histogram(error, fname="figures/error_hist.png"):
    plt.figure()
    plt.hist(error, bins=50, alpha=0.7)
    plt.xlabel("Ошибка (нТл)")
    plt.ylabel("Частота")
    plt.title("Распределение ошибки после калибровки")
    plt.grid(True)
    plt.savefig(fname)
    plt.close()

def save_all_visualizations(m_raw, m_calibrated, m_ref, error, angle_labels):
    plot_ellipses(m_raw, m_calibrated)
    plot_3d_clouds(m_raw, m_calibrated)
    plot_vector_norms(m_raw, m_calibrated, m_ref)
    plot_error_histogram(error)
    angles_deg = angular_error(m_calibrated, m_ref)
    plot_angular_error(angles_deg, angle_labels)
