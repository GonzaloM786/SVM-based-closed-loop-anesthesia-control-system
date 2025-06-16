import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from cycler import cycler
import scienceplots

# Estilos científicos
plt.style.use(["science", "ieee", "grid"])
sns.set(style="whitegrid", context="paper", font_scale=1.9)

# Definir paleta de colores std_colors de scienceplots
std_colors = ['#0C5DA5', '#00B945', '#FF9500', '#FF2C00', '#845B97', '#474747', '#9e9e9e']
plt.rcParams['axes.prop_cycle'] = cycler('color', std_colors)

# Cargar datos
df = pd.read_csv("../simulink/results/control_metrics.csv")

# Asegurar que la columna 'Reference' es categórica
df["Reference"] = df["Reference"].astype(str)

# Orden lógico para las referencias
reference_order = ["40", "50", "60", "Triangular", "Gaussian noise", "Infusion loss"]

# Métricas y títulos
metrics = ["Undershoot", "SettlingTime_20", "SettlingTime_5", "MSE", "IAE", "ISE"]
titles = {
    "Undershoot": "Undershoot (\%)",
    "SettlingTime_20": r"Settling Time ($\pm$20\%) [min]",
    "SettlingTime_5": r"Settling Time ($\pm$5\%) [min]",
    "MSE": "Mean Squared Error (MSE)",
    "IAE": "Integral of Absolute Error (IAE)",
    "ISE": "Integral of Squared Error (ISE)",
}

# Etiquetas de panel
panel_labels = ['(a)', '(b)', '(c)', '(d)', '(e)', '(f)']

# Crear figura
fig, axes = plt.subplots(3, 2, figsize=(13, 10))
axes = axes.flatten()

# Paleta fija para controladores usando std_colors
palette = {"SVM-based": std_colors[0], "PID": std_colors[2]}

# Graficar cada métrica
for i, metric in enumerate(metrics):
    ax = axes[i]
    sns.barplot(
        data=df,
        x="Reference",
        y=metric,
        hue="Controller",
        palette=palette,
        order=reference_order,
        ax=ax,
    )
    ax.text(-0.1, 1.1, panel_labels[i], transform=ax.transAxes,
            fontsize=17, fontweight='bold', va='top', ha='left')
    ax.set_title(titles[metric], fontsize=17, fontweight='bold')
    ax.set_xlabel("Experiment Condition")
    ax.set_ylabel("")
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.legend().set_visible(False)

    if i < 4:
        ax.set_xlabel("")
        ax.set_xticklabels([])
    else:
        ax.set_xticklabels(
            ['Ref 40', 'Ref 50', 'Ref 60', 'Ref Triangular', 'Gaussian Noise', 'Infusion Loss'],
            rotation=30, ha='right'
        )

# Leyenda global arriba
handles, labels = axes[-1].get_legend_handles_labels()
fig.legend(handles, labels, loc='upper center', ncol=2, bbox_to_anchor=(0.5, 1.04), frameon=False)

# Ajustes finales y guardado
plt.tight_layout()
plt.savefig("../../assets/control_metrics_summary.pdf", bbox_inches='tight')
print("Saved: ../../assets/control_metrics_summary.pdf")
plt.close()