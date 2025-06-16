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

# Cargar archivos
val_df = pd.read_csv('../data/svr_validation_predictions.csv')
test_df = pd.read_csv('../data/svr_test_predictions.csv')

panel_labels = ['(a)', '(b)']

# Función para graficar resultados
def plot_results(df, save_path=None):
    time = df['Time']
    y_real = df['Y_real']
    y_pred = df['Y_pred']
    y_pred_pert = df['Y_pred_pert']
    delta_bis = df['DeltaBIS']

    _, axs = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    axs[0].set_xlim(0, 5330)
    axs[1].set_xlim(0, 5330)

    # Subplot 1: BIS real vs predicho
    axs[0].plot(time, y_real, label='Real', color=std_colors[0], linewidth=1.5)
    axs[0].plot(time, y_pred, label='Predicted', linestyle='--', color=std_colors[1], linewidth=1.5)
    axs[0].plot(time, y_pred_pert, label='Perturbed', linestyle='--', color=std_colors[2], linewidth=1.3, alpha=0.7)
    axs[0].set_ylabel('BIS')
    axs[0].legend()
    axs[0].set_title('Real vs Predicted BIS')
    axs[0].text(-0.07, 1.1, panel_labels[0], transform=axs[0].transAxes,
                 fontsize=17, fontweight='bold', va='top', ha='left')
    axs[0].spines['top'].set_visible(False)
    axs[0].spines['right'].set_visible(False)

    # Subplot 2: Delta BIS
    axs[1].plot(time, delta_bis, color=std_colors[6], linewidth=1)
    axs[1].fill_between(time, delta_bis, 0, where=delta_bis < 0, color=std_colors[3], alpha=0.4, label='Correct (↓ BIS)')
    axs[1].fill_between(time, delta_bis, 0, where=delta_bis >= 0, color=std_colors[4], alpha=0.4, label='Incorrect (↑ BIS)')
    axs[1].set_xlabel('Time')
    axs[1].set_ylabel(r'$\Delta$BIS (Perturbed - Predicted)')
    axs[1].legend()
    axs[1].set_title(r'$\Delta$BIS after Perturbation')
    axs[1].text(-0.07, 1.1, panel_labels[1], transform=axs[1].transAxes,
                 fontsize=17, fontweight='bold', va='top', ha='left')
    axs[1].spines['top'].set_visible(False)
    axs[1].spines['right'].set_visible(False)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
        print(f"Gráfica guardada en {save_path}")
    else:
        plt.show()


# Generar gráficos
plot_results(val_df, save_path="../../assets/validation_plot.pdf")
plot_results(test_df, save_path="../../assets/test_plot.pdf")
