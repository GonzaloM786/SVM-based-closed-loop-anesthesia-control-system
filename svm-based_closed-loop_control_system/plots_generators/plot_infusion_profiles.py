import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from cycler import cycler
import scienceplots

# Estilos
plt.style.use(["science", "ieee", "grid"])
sns.set(style="whitegrid", context="paper", font_scale=1.2)

# Paleta std_colors (alta visibilidad, estilo científico)
std_colors = ['#0C5DA5', '#00B945', '#FF9500', '#FF2C00', '#845B97', '#474747', '#9e9e9e']
plt.rcParams['axes.prop_cycle'] = cycler('color', std_colors)

# Rutas de archivo
path_unfiltered = "../data/Smooth_infusion_data_non_filtered_0_40_90_0.9_54_40.0.csv"
path_filtered = "../data/Smooth_infusion_data_0_40_90_0.9_54_40.0.csv"

# Cargar datos
raw_data = pd.read_csv(path_unfiltered, header=None)
filtered_data = pd.read_csv(path_filtered, header=None)

# Usar la primera fila de simulación
infusion_raw = raw_data.iloc[0].values
infusion_filtered = filtered_data.iloc[0].values

# Eje temporal (en minutos)
sampling_interval = 1  # segundos
total_time_min = len(infusion_raw) / 60
time_min = np.linspace(0, total_time_min, len(infusion_raw))

# Crear figura
fig, ax = plt.subplots(figsize=(6.5, 3.5))

# Trazar señales
ax.plot(time_min, infusion_raw, label="Raw infusion profile", color=std_colors[0], alpha=0.7, linewidth=1)
ax.plot(time_min, infusion_filtered, label="Filtered infusion profile", color=std_colors[2], linewidth=1.5)
ax.set_xlim(0, 90)

# Etiquetas
ax.set_xlabel("Time (min)")
ax.set_ylabel("Infusion Rate (mg/min)")

# Leyenda arriba centrada
ax.legend(loc="upper center", bbox_to_anchor=(0.5, 1.18), ncol=2, frameon=False)

# Limpiar bordes
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.grid(True)

# Guardar figura
plt.tight_layout()
plt.savefig("../../assets/infusion_profile_comparison.pdf", bbox_inches="tight", dpi=300)
print("Saved: ../../assets/infusion_profile_comparison.pdf")
plt.close()