import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from cycler import cycler
import scienceplots

# Estilos científicos
plt.style.use(["science", "ieee", "grid"])
sns.set(style="whitegrid", context="paper", font_scale=1.9)

# Paleta std-colors
std_colors = ['#0C5DA5', '#00B945', '#FF9500', '#FF2C00', '#845B97', '#474747', '#9e9e9e']
plt.rcParams['axes.prop_cycle'] = cycler('color', std_colors)

# Cargar datos
df = pd.read_csv('../data/schnider_simulation.csv')

# Crear figura y ejes
fig, axes = plt.subplots(2, 2, figsize=(14, 8))
axes = axes.flatten()

panel_labels = ['(a)', '(b)', '(c)', '(d)']

for i in range(len(axes)):
    axes[i].set_xlim(0, 60)
    axes[i].spines['top'].set_visible(False)
    axes[i].spines['right'].set_visible(False)
    axes[i].text(-0.1, 1.1, panel_labels[i], transform=axes[i].transAxes,
                 fontsize=17, fontweight='bold', va='top', ha='left')

# === Panel A ===
line1, = axes[0].plot(df['Time_min'], df['C1_Plasma'], label='C1 (Plasma)', color=std_colors[0])
line2, = axes[0].plot(df['Time_min'], df['C2_RapidPeripheral'], label='C2 (Rapid Peripheral)', color=std_colors[1])
line3, = axes[0].plot(df['Time_min'], df['C3_SlowPeripheral'], label='C3 (Slow Peripheral)', color=std_colors[2])
line4, = axes[0].plot(df['Time_min'], df['Ce_EffectSite'], label='Ce (Effect Site)', color=std_colors[3])
axes[0].set_title('Schnider PK Model - Compartment Concentrations', weight='bold')
axes[0].set_xlabel('Time (min)')
axes[0].set_ylabel('Concentration (mg/L)')
axes[0].grid(True)

# === Panel B ===
axes[1].plot(df['Time_min'], df['BIS'], color=std_colors[4])
axes[1].set_title('Schnider PD Model - BIS Index', weight='bold')
axes[1].set_xlabel('Time (min)')
axes[1].set_ylabel('BIS')
axes[1].grid(True)

# === Panel C ===
axes[2].plot(df['Time_min'], df['C1_Plasma'], label='C1 (Plasma)', color=std_colors[0])
axes[2].plot(df['Time_min'], df['Ce_EffectSite'], label='Ce (Effect Site)', color=std_colors[3])  # rojo igual
axes[2].set_title('Comparison Plasma vs Effect Site', weight='bold')
axes[2].set_xlabel('Time (min)')
axes[2].set_ylabel('Concentration (mg/L)')
axes[2].grid(True)

# === Panel D ===
axes[3].plot(df['Time_min'], df['InfusionRate'], color=std_colors[6])
axes[3].set_title('Infusion Profile Administered', weight='bold')
axes[3].set_xlabel('Time (min)')
axes[3].set_ylabel('Infusion rate (mg/min)')
axes[3].grid(True)

# === Leyenda global ===
handles = [line1, line2, line3, line4]
labels = [h.get_label() for h in handles]
fig.legend(handles, labels, loc='upper center', ncol=4, bbox_to_anchor=(0.5, 1.04), frameon=False)

# Guardar
plt.tight_layout(rect=[0, 0, 1, 0.97])  # deja espacio para leyenda
plt.savefig('../../assets/schnider_simulation_overview.pdf', bbox_inches='tight')
plt.close()