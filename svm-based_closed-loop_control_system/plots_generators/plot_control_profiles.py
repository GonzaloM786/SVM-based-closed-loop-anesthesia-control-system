import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from cycler import cycler
import scienceplots
import os

# Estilos científicos
plt.style.use(["science", "ieee", "grid"])
sns.set(style="whitegrid", context="paper", font_scale=1.9)

# Paleta de colores estándar
std_colors = ['#0C5DA5', '#00B945', '#FF9500', '#FF2C00', '#845B97', '#474747', '#9e9e9e']
plt.rcParams['axes.prop_cycle'] = cycler('color', std_colors)

# Definición de experimentos y controladores
experiments = [
    "ref_40", "ref_50", "ref_60",        # Consignas fijas
    "triangular",            # Consigna variable
    "noise",        # Ruido gaussiano
    "infusion_loss"          # Pérdida de infusión
]

controllers = ["SVM", "PID"]

# Ruta a los CSVs
data_path = "../data"
output_path = "../../assets/experiments"

# Etiquetas más amigables para los títulos
exp_titles = {
    "ref_40": "Fixed Reference = 40",
    "ref_50": "Fixed Reference = 50",
    "ref_60": "Fixed Reference = 60",
    "triangular": "Variable Reference (Triangular)",
    "noise": "Gaussian Noise in Feedback",
    "infusion_loss": "Infusion Loss Scenario"
}

# Asegura que la carpeta de salida exista
os.makedirs(output_path, exist_ok=True)

# Iterar sobre todos los experimentos y controladores
for exp in experiments:
    for ctrl in controllers:
        # Construir nombre de archivo
        filename = f"BIS_data_{ctrl}_{exp}.csv"
        file_path = os.path.join(data_path, filename)

        # Cargar CSV
        df = pd.read_csv(file_path)
        df['Time_min'] = df['Time_s'] / 60

        # Crear figura
        fig, axes = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
        fig.subplots_adjust(hspace=0.3)
        panel_labels = ['(a)', '(b)']

        # === Subfigura A: BIS ===
        axes[0].plot(df['Time_min'], df['BIS'], label='BIS', color=std_colors[0])
        if 'Reference' in df.columns:
            axes[0].plot(df['Time_min'], df['Reference'], label='Reference', color=std_colors[2], linestyle='--')
        else:
            if exp == "ref_40":
                ref_val = 40
            elif exp == "ref_60":
                ref_val = 60
            else:
                ref_val = 50
            axes[0].hlines(ref_val, xmin=0, xmax=df['Time_min'].max(),
                        label=f'Reference = {ref_val}', color=std_colors[2], linestyle='--')

        axes[0].set_ylabel('BIS')
        axes[0].text(-0.05, 1.2, panel_labels[0], transform=axes[0].transAxes,
                    fontsize=17, fontweight='bold', va='top', ha='left')
        axes[0].set_xlim(0, 60)
        axes[0].set_ylim(0, 100)
        axes[0].grid(True)
        axes[0].legend(loc='upper center', bbox_to_anchor=(0.5, 1.4), ncol=2, frameon=False)
        axes[0].spines['top'].set_visible(False)
        axes[0].spines['right'].set_visible(False)
        if ctrl == "SVM":
            axes[0].set_title('BIS Real vs Predicted', weight='bold')
        else:
            axes[0].set_title('BIS Real', weight='bold')

        # === Subfigura B: Infusion ===
        axes[1].plot(df['Time_min'], df['InfusionRate'], color=std_colors[1])
        axes[1].set_xlabel('Time (min)')
        axes[1].set_ylabel('Infusion rate (mg/min)')
        axes[1].text(-0.05, 1.2, panel_labels[1], transform=axes[1].transAxes,
                    fontsize=17, fontweight='bold', va='top', ha='left')
        axes[1].set_xlim(0, 60)
        axes[1].set_ylim(0, 30)
        axes[1].grid(True)
        axes[1].spines['top'].set_visible(False)
        axes[1].spines['right'].set_visible(False)
        axes[1].set_title('Anesthesia Infusion Rate', weight='bold')

        # Guardar como PDF
        output_filename = f"{ctrl}_{exp}_experiment.pdf"
        fig.savefig(os.path.join(output_path, output_filename), bbox_inches='tight')
        plt.close()
