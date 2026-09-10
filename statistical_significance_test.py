"""
Statistical significance testing + visualization for Bayes-VCF vs. baselines.

Tests:  Wilcoxon signed-rank, paired t-test, Friedman, Cohen's d, Bonferroni
Plots:  6 publication-quality figures saved to ./figures/
"""

import os, numpy as np
from scipy import stats
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.patches import Patch

# ── Global style ─────────────────────────────────────────────────────
plt.rcParams.update({
    'font.family': 'serif', 'font.size': 10,
    'axes.titlesize': 12, 'axes.labelsize': 11,
    'legend.fontsize': 8, 'figure.dpi': 200,
    'savefig.dpi': 300, 'savefig.bbox': 'tight',
})
OUT = 'figures'
os.makedirs(OUT, exist_ok=True)

# ── Utility ──────────────────────────────────────────────────────────
def wilcoxon_signed_rank(x, y, alpha=0.05, alternative='two-sided'):
    try:
        # The published category values have two decimal places.  Rounding the
        # paired differences prevents binary floating-point noise from turning
        # intended ties into different ranks (see scipy.stats.wilcoxon notes).
        diff = np.around(np.asarray(x, dtype=float) - np.asarray(y, dtype=float), decimals=8)
        stat, p = stats.wilcoxon(diff, alternative=alternative)
        return stat, p, p < alpha
    except ValueError:
        return np.nan, np.nan, False

def holm_adjust(p_values):
    """Holm-adjust p-values while preserving their original order."""
    p_values = np.asarray(p_values, dtype=float)
    order = np.argsort(p_values)
    adjusted = np.empty_like(p_values)
    running_max = 0.0
    m = len(p_values)
    for rank, index in enumerate(order):
        running_max = max(running_max, min(1.0, (m - rank) * p_values[index]))
        adjusted[index] = running_max
    return adjusted

def cohens_d(x, y):
    diff = np.array(x, dtype=float) - np.array(y, dtype=float)
    return np.mean(diff) / np.std(diff, ddof=1) if np.std(diff, ddof=1) > 0 else 0.0

def effect_size_label(d):
    d_abs = abs(d)
    if d_abs < 0.2: return "negligible"
    elif d_abs < 0.5: return "small"
    elif d_abs < 0.8: return "medium"
    else: return "large"

def bonferroni(p_values, alpha=0.05):
    m = len(p_values)
    return alpha / m, [p < alpha / m for p in p_values]

def p_stars(p):
    if p < 0.001: return '***'
    elif p < 0.01: return '**'
    elif p < 0.05: return '*'
    else: return 'ns'

# ── Color palette ────────────────────────────────────────────────────
BASELINE_COLORS = {
    'APRIL-GAN':   '#E8A87C',
    'CLIP-AD':     '#95E1D3',
    'AnomalyCLIP': '#F38181',
    'AdaCLIP':     '#AA96DA',
    'Bayes-PFL':   '#A8D8EA',
    'Bayes-VCF':   '#FC5185',
}
VCF_COLOR = '#FC5185'
PFL_COLOR = '#364F6B'

# ═══════════════════════════════════════════════════════════════════════
#  DATA
# ═══════════════════════════════════════════════════════════════════════

ydfid_classes = ['CL1','CL2','CL3','CL4','CL10','CL12',
                 'SL1','SL8','SL9','SL10','SL11','SL13','SL16',
                 'SP3','SP5','SP19','SP24']

ydfid_auroc_px = {
    'APRIL-GAN':   [99.90,99.20,99.00,91.90,81.80,88.80,80.50,93.30,90.20,90.20,71.30,77.60,74.60,91.90,95.90,76.70,86.50],
    'CLIP-AD':     [99.90,99.20,99.90,92.90,91.80,96.30,90.20,97.50,99.90,87.80,84.40,91.20,67.80,98.80,98.70,94.00,98.20],
    'AnomalyCLIP': [100.00,99.90,97.70,91.30,84.20,89.90,77.60,98.00,97.00,93.70,81.20,75.30,76.00,95.00,98.50,87.50,91.70],
    'AdaCLIP':     [99.98,98.59,98.79,93.82,82.04,96.31,87.03,98.67,96.54,79.15,53.65,78.61,91.76,91.44,95.15,91.88,96.86],
    'Bayes-PFL':   [99.98,99.69,99.87,93.02,89.36,95.85,88.11,97.65,96.92,93.18,77.35,83.90,86.08,99.01,98.68,91.94,95.54],
    'Bayes-VCF':   [99.98,99.47,99.64,95.84,92.43,96.44,90.89,97.89,96.31,92.80,81.76,92.91,92.66,99.21,99.06,94.26,94.43],
}

ydfid_ap_px = {
    'APRIL-GAN':   [31.60,32.20,69.30,39.60,19.30,19.10,12.50,20.80,49.30,20.60,5.70,13.50,6.40,44.20,47.50,18.60,26.10],
    'CLIP-AD':     [13.00,12.60,53.10,33.80,18.40,33.50,17.80,24.50,62.70,11.70,23.10,23.10,2.70,35.80,33.90,37.90,38.10],
    'AnomalyCLIP': [22.60,15.70,66.20,36.40,31.20,28.30,9.20,30.10,68.00,29.80,12.70,11.00,6.20,44.10,46.10,30.20,37.40],
    'AdaCLIP':     [25.48,27.50,66.96,31.20,25.89,23.99,15.82,25.30,46.76,14.44,10.84,3.38,7.55,35.19,42.10,24.14,35.08],
    'Bayes-PFL':   [30.15,30.91,62.30,38.55,31.07,33.02,16.32,29.14,51.08,26.69,20.36,21.95,10.93,55.91,55.13,33.14,43.15],
    'Bayes-VCF':   [34.20,28.90,66.47,40.55,36.16,35.69,20.05,27.49,53.66,24.75,23.72,24.51,11.15,56.02,53.70,46.30,43.36],
}

ydfid_aupro = {
    'APRIL-GAN':   [74.2,49.2,43.4,37.1,2.8,17.7,24.8,13.6,18.3,10.8,7.3,8.7,14.3,34.5,25.5,14.2,12.8],
    'CLIP-AD':     [99.8,94.8,97.9,85.4,61.3,89.5,83.0,96.1,99.4,62.7,82.4,94.0,46.8,95.7,94.3,71.7,86.9],
    'AnomalyCLIP': [99.9,99.7,84.0,81.9,49.7,85.0,74.3,96.4,92.6,80.1,53.1,77.1,68.7,88.6,95.2,64.3,76.3],
    'AdaCLIP':     [38.97,39.27,52.23,40.02,17.62,29.71,26.94,27.31,31.70,29.17,46.54,71.72,68.81,39.95,27.06,23.65,21.94],
    'Bayes-PFL':   [99.93,96.49,95.45,74.45,64.24,91.05,82.21,92.46,91.03,85.37,75.22,91.24,77.14,98.39,94.82,74.29,80.19],
    'Bayes-VCF':   [99.93,96.56,94.75,77.96,68.88,92.45,83.33,90.26,92.47,79.90,77.12,88.71,84.83,98.68,96.97,76.49,81.30],
}

mvtec_runs_raw = [
    [91.80,48.30,49.09,34.33,92.30,96.70,92.93,0.628],
    [92.20,46.03,46.70,32.33,91.08,95.25,92.23,0.598],
    [92.39,47.85,48.88,33.89,92.01,95.94,93.29,0.545],
    [91.83,46.79,48.11,33.45,93.40,97.07,93.62,0.567],
    [87.23,27.32,31.58,19.64,83.41,91.35,89.23,0.532],
    [91.35,43.35,45.18,30.46,92.08,96.43,93.03,0.461],
    [92.16,47.77,48.59,33.65,93.33,96.89,93.39,0.516],
    [91.95,44.27,46.38,31.84,91.49,95.86,92.50,0.559],
    [91.54,41.94,44.91,30.52,90.77,95.13,91.97,0.565],
    [91.64,46.93,48.65,33.78,91.82,95.96,92.44,0.574],
    [92.07,45.43,46.80,32.29,91.75,96.00,92.75,0.589],
    [91.63,40.53,43.11,29.08,91.29,95.10,92.62,0.577],
    [91.43,45.24,47.25,32.63,92.08,95.44,92.37,0.559],
    [92.63,48.33,49.09,34.12,91.54,96.22,92.58,0.489],
    [91.74,45.00,46.42,32.10,90.74,95.38,91.95,0.000],
    [92.64,49.32,50.09,34.96,92.41,96.61,93.36,0.637],
    [92.69,48.87,49.69,34.69,92.33,96.52,93.12,0.601],
    [92.80,48.54,49.20,34.15,91.99,96.18,92.33,0.000],
    [92.81,49.63,50.03,35.02,90.48,95.47,92.24,0.000],
]

mvtec_baselines = {
    'APRIL-GAN':   {'AUROC-SP':86.1,'F1Max-SP':90.4,'AP-SP':93.5,'AUROC-PX':87.6,'F1Max-PX':44.0,'AP-PX':40.8},
    'CLIP-AD':     {'AUROC-SP':89.8,'F1Max-SP':91.1,'AP-SP':95.3,'AUROC-PX':89.8,'F1Max-PX':70.6,'AP-PX':40.0},
    'AnomalyCLIP': {'AUROC-SP':91.5,'F1Max-SP':92.8,'AP-SP':96.2,'AUROC-PX':91.1,'F1Max-PX':81.4,'AP-PX':34.5},
    'AdaCLIP':     {'AUROC-SP':92.0,'F1Max-SP':92.7,'AP-SP':96.4,'AUROC-PX':86.8,'F1Max-PX':33.8,'AP-PX':38.1},
    'Bayes-PFL':   {'AUROC-SP':92.3,'F1Max-SP':93.1,'AP-SP':96.7,'AUROC-PX':91.8,'F1Max-PX':87.4,'AP-PX':48.3},
}

visa_baselines = {
    'APRIL-GAN':   {'AUROC-SP':78.0,'F1Max-SP':78.7,'AP-SP':81.4,'AUROC-PX':94.2,'F1Max-PX':86.8,'AP-PX':25.7},
    'CLIP-AD':     {'AUROC-SP':79.8,'F1Max-SP':79.2,'AP-SP':84.3,'AUROC-PX':95.0,'F1Max-PX':86.9,'AP-PX':26.3},
    'AnomalyCLIP': {'AUROC-SP':82.1,'F1Max-SP':80.4,'AP-SP':85.4,'AUROC-PX':95.5,'F1Max-PX':87.0,'AP-PX':21.3},
    'AdaCLIP':     {'AUROC-SP':83.0,'F1Max-SP':81.6,'AP-SP':84.9,'AUROC-PX':95.1,'F1Max-PX':71.3,'AP-PX':29.2},
    'Bayes-PFL':   {'AUROC-SP':87.0,'F1Max-SP':84.1,'AP-SP':89.2,'AUROC-PX':95.6,'F1Max-PX':88.9,'AP-PX':29.8},
}

vcf_mvtec_thesis = {'AUROC-SP':(93.41,0.18),'F1Max-SP':(93.30,0.04),'AP-SP':(96.70,0.15),
                    'AUROC-PX':(92.69,0.07),'F1Max-PX':(88.50,0.22),'AP-PX':(49.30,0.13)}

vcf_visa_thesis = {'AUROC-SP':(88.10,0.19),'F1Max-SP':(85.00,0.14),'AP-SP':(90.10,0.15),
                   'AUROC-PX':(95.70,0.12),'F1Max-PX':(89.50,0.17),'AP-PX':(30.20,0.18)}

ablation_mvtec = {
    'AUC-SP':  {'Exper1':93.10,'Exper2':94.04,'Exper3':94.22,'Exper4':94.35,'Exper5':95.12,'Ours':95.60},
    'AUC-PX':  {'Exper1':93.30,'Exper2':94.18,'Exper3':94.05,'Exper4':93.85,'Exper5':94.86,'Ours':95.40},
    'AUPRO-PX':{'Exper1':86.67,'Exper2':86.91,'Exper3':87.02,'Exper4':86.83,'Exper5':87.18,'Ours':87.34},
}

ablation_dilation = {
    'YDFID-1 AUC-SP': {'[1,3,3]':88.50,'[2,3,3]':88.20,'[1,2,3]':95.60,'[3,3,3]':87.90},
    'YDFID-1 AUC-PX': {'[1,3,3]':93.82,'[2,3,3]':93.51,'[1,2,3]':95.40,'[3,3,3]':93.20},
    'MVTec AUC-SP':   {'[1,3,3]':92.90,'[2,3,3]':92.50,'[1,2,3]':93.40,'[3,3,3]':92.10},
    'MVTec AUC-PX':   {'[1,3,3]':92.35,'[2,3,3]':92.12,'[1,2,3]':92.69,'[3,3,3]':91.80},
}

ablation_compression = {
    'YDFID-1 AUC-SP': {'1/2':95.60,'1/4':88.70,'1/8':88.01},
    'YDFID-1 AUC-PX': {'1/2':95.40,'1/4':93.85,'1/8':93.25},
    'MVTec AUC-SP':   {'1/2':93.41,'1/4':92.81,'1/8':92.12},
    'MVTec AUC-PX':   {'1/2':92.69,'1/4':92.36,'1/8':91.75},
}

ydfid_3runs = {
    'run1': {
        'AUROC-PX':[99.98,91.41,96.44,98.75,99.64,92.30,90.89,92.43,81.76,90.86,88.06,97.29,96.31,92.46,94.44,99.41,98.65],
        'AUROC-SP':[99.60,78.73,94.93,99.25,93.43,100.00,93.10,98.10,86.21,88.22,98.39,99.23,99.65,86.88,97.53,99.85,98.38],
        'AP-PX':   [34.12,34.30,34.49,29.97,66.07,41.91,22.06,27.43,20.75,26.92,12.65,29.99,53.90,42.17,45.60,57.07,54.99],
        'AP-SP':   [94.80,38.06,87.66,87.67,75.83,100.0,77.75,94.61,52.73,59.04,83.83,93.93,90.38,71.07,91.10,98.57,92.00],
        'F1Max-SP':[88.89,45.45,79.07,90.91,70.59,100.0,76.92,93.02,54.55,60.00,81.08,88.89,85.71,75.00,86.67,94.12,85.71],
    },
    'run2': {
        'AUROC-PX':[99.98,92.43,96.02,99.47,99.46,95.84,90.01,92.80,69.98,92.91,92.66,97.02,96.22,94.26,94.28,99.14,99.06],
        'AUROC-SP':[99.73,80.79,96.88,99.25,95.56,100.00,91.55,96.44,95.48,90.91,98.80,99.42,99.77,85.41,96.32,100.00,98.95],
        'AP-PX':   [32.43,35.53,32.85,30.23,61.64,40.44,21.95,24.39,19.87,21.34,12.29,27.83,54.13,41.16,44.12,53.83,50.70],
        'AP-SP':   [96.58,49.16,92.13,87.67,84.19,100.0,80.12,93.00,72.91,65.41,86.01,95.28,90.94,70.29,91.39,100.00,94.11],
        'F1Max-SP':[94.12,48.00,85.71,90.91,84.21,100.0,76.92,93.02,71.43,65.00,87.50,90.32,93.33,75.00,86.11,100.00,88.89],
    },
    'run3': {
        'AUROC-PX':[99.98,91.09,95.34,99.59,99.67,94.53,87.55,91.12,72.23,91.83,91.64,96.35,94.44,93.54,91.40,98.92,99.01],
        'AUROC-SP':[99.60,81.50,95.39,99.25,95.56,99.90,93.62,95.59,91.45,90.82,98.85,99.52,99.71,87.27,95.73,100.00,99.40],
        'AP-PX':   [34.20,36.16,35.69,28.90,66.47,40.55,20.05,24.75,23.72,24.51,11.15,27.49,53.66,46.30,43.36,56.02,53.70],
        'AP-SP':   [95.56,49.07,90.48,87.67,82.84,98.66,82.18,93.14,67.98,70.29,87.84,96.26,92.14,73.46,91.39,100.00,96.43],
        'F1Max-SP':[94.12,50.00,84.21,90.91,77.78,96.00,80.00,93.02,66.67,73.17,90.32,93.75,83.33,80.00,87.32,100.00,91.89],
    },
}

subclass_ranges = {'CL':(0,6), 'SL':(6,13), 'SP':(13,17)}

# ═══════════════════════════════════════════════════════════════════════
#  FIGURE 1: YDFID-1 per-class paired comparison box plot
# ═══════════════════════════════════════════════════════════════════════

def fig1_ydfid_per_class_boxplot():
    """Box plot: Bayes-VCF vs all baselines across YDFID-1 classes."""
    baselines = ['APRIL-GAN','CLIP-AD','AnomalyCLIP','AdaCLIP','Bayes-PFL','Bayes-VCF']
    display_names = ['APRIL-GAN','CLIP-AD','AnomalyCLIP','AdaCLIP','Bayes-PFL','VCF']
    metrics = [('AUROC-PX (%)', ydfid_auroc_px), ('AP-PX (%)', ydfid_ap_px), ('AUPRO-PX (%)', ydfid_aupro)]

    raw_p_values = []
    for _, data in metrics:
        _, p_value, _ = wilcoxon_signed_rank(
            data['Bayes-VCF'], data['Bayes-PFL'], alternative='two-sided'
        )
        raw_p_values.append(p_value)
    adjusted_p_values = holm_adjust(raw_p_values)

    # Final full-width print size used in the manuscript.
    fig, axes = plt.subplots(3, 1, figsize=(5.76, 5.08))
    fig.suptitle('YDFID-1 category-level pixel-localization performance (17 pattern categories)',
                 fontweight='bold', y=0.995, fontsize=9.7)

    for metric_index, (ax, (metric_name, data)) in enumerate(zip(axes, metrics)):
        positions = list(range(len(baselines)))
        boxes = []
        for bl in baselines:
            boxes.append(data[bl])

        bp = ax.boxplot(boxes, positions=positions, widths=0.55, patch_artist=True,
                        medianprops={'color':'black','linewidth':1.2},
                        whiskerprops={'linewidth':1}, capprops={'linewidth':1})

        colors = [BASELINE_COLORS[bl] for bl in baselines]
        for patch, color in zip(bp['boxes'], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.85)

        # Overlay individual points
        for i, bl in enumerate(baselines):
            x_jitter = np.random.default_rng(42).uniform(-0.1, 0.1, len(data[bl]))
            ax.scatter(np.full(len(data[bl]), i) + x_jitter, data[bl],
                       s=9, c='black', alpha=0.45, zorder=3, edgecolors='none')

        ax.set_xticks(positions)
        ax.set_xticklabels(display_names, rotation=12, ha='right', fontsize=7.1)
        ax.tick_params(axis='y', labelsize=7.1)
        ax.set_ylabel(metric_name, fontsize=7.8)
        ax.set_title(metric_name, loc='left', fontweight='bold', fontsize=8.5, pad=8)
        ax.grid(axis='y', alpha=0.3, linestyle='--')

        # Significance annotations: Bayes-VCF vs Bayes-PFL
        vcf = np.array(data['Bayes-VCF'])
        pfl = np.array(data['Bayes-PFL'])
        p_val = raw_p_values[metric_index]
        p_holm = adjusted_p_values[metric_index]
        d = cohens_d(vcf, pfl)
        annotation_color = '#8B0000' if p_holm < 0.05 else '#4A4A4A'
        ax.text(
            1.0,
            1.025,
            f'VCF vs Bayes-PFL: two-sided p={p_val:.4f}\nHolm p={p_holm:.4f}; paired d={d:.2f}',
            transform=ax.transAxes,
            ha='right',
            va='bottom',
            fontsize=6.5,
            color=annotation_color,
            fontweight='bold',
            linespacing=1.0,
        )

    fig.text(
        0.5,
        0.008,
        'Two-sided Wilcoxon signed-rank tests; Holm adjustment across three metrics. '
        'VCF category values: seed 333; n=17.',
        ha='center',
        va='bottom',
        fontsize=6.3,
        color='#333333',
    )
    plt.tight_layout(rect=[0, 0.04, 1, 0.96], h_pad=1.5)
    fig.savefig(f'{OUT}/fig1_ydfid_per_class_boxplot.png', dpi=600, facecolor='white')
    plt.close(fig)
    print(f'  [saved] {OUT}/fig1_ydfid_per_class_boxplot.png')


# ═══════════════════════════════════════════════════════════════════════
#  FIGURE 2: YDFID-1 paired difference scatter (VCF vs PFL)
# ═══════════════════════════════════════════════════════════════════════

def fig2_ydfid_paired_diff():
    """Paired scatter + diff bar: Bayes-VCF minus Bayes-PFL per class."""
    metrics_map = {'AUROC-PX': ydfid_auroc_px, 'AP-PX': ydfid_ap_px, 'AUPRO': ydfid_aupro}
    diffs = {}
    pvals = {}

    for m_name, data in metrics_map.items():
        vcf = np.array(data['Bayes-VCF'])
        pfl = np.array(data['Bayes-PFL'])
        diffs[m_name] = vcf - pfl
        # Per-class Wilcoxon is not meaningful with n=1, so use overall paired test
        _, pvals[m_name], _ = wilcoxon_signed_rank(vcf, pfl, alternative='greater')

    fig, axes = plt.subplots(1, 3, figsize=(16, 5.5))
    fig.suptitle('YDFID-1: Per-class Improvement of Bayes-VCF over Bayes-PFL', fontweight='bold', y=1.02)

    subclass_colors = {'CL': '#E76F51', 'SL': '#2A9D8F', 'SP': '#264653'}
    for ax, (m_name, diff) in zip(axes, diffs.items()):
        x = np.arange(len(ydfid_classes))
        bar_colors = []
        for cls in ydfid_classes:
            for sc, (s, e) in subclass_ranges.items():
                if cls in ydfid_classes[s:e]:
                    bar_colors.append(subclass_colors[sc])
                    break
        bars = ax.bar(x, diff, color=bar_colors, edgecolor='white', linewidth=0.5, alpha=0.9)
        ax.axhline(y=0, color='gray', linewidth=0.8, linestyle='-')
        ax.axhline(y=np.mean(diff), color='red', linewidth=1.2, linestyle='--',
                   label=f'Mean Δ={np.mean(diff):+.2f}')
        ax.set_xticks(x)
        ax.set_xticklabels(ydfid_classes, rotation=45, ha='right', fontsize=7)
        ax.set_ylabel(f'Δ {m_name}')
        ax.set_title(f'{m_name} (p={pvals[m_name]:.4f})', fontweight='bold')
        ax.legend(fontsize=7)
        ax.grid(axis='y', alpha=0.3, linestyle='--')

    legend_elements = [Patch(facecolor=subclass_colors[sc], label=sc) for sc in ['CL','SL','SP']]
    fig.legend(handles=legend_elements, loc='lower center', ncol=3, fontsize=9, frameon=False)

    plt.tight_layout(rect=[0, 0.06, 1, 1])
    fig.savefig(f'{OUT}/fig2_ydfid_paired_difference.png', facecolor='white')
    plt.close(fig)
    print(f'  [saved] {OUT}/fig2_ydfid_paired_difference.png')


# ═══════════════════════════════════════════════════════════════════════
#  FIGURE 3: MVTec-AD multi-run distribution + CI
# ═══════════════════════════════════════════════════════════════════════

def fig3_mvtec_multirun():
    """Multi-run Bayes-VCF performance on MVTec-AD with 95% CI."""
    runs = np.array(mvtec_runs_raw)
    valid = runs[runs[:, 7] > 0.001]
    col_map = {0:'AUROC-PX',1:'AP-PX',2:'F1Max-PX',3:'IoU',
               4:'AUROC-SP',5:'AP-SP',6:'F1Max-SP'}

    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    axes_flat = axes.flatten()
    fig.suptitle('MVTec-AD: Bayes-VCF Multi-run Performance Distribution (n={})'.format(len(valid)),
                 fontweight='bold', y=1.01)

    for idx, (col, name) in enumerate(col_map.items()):
        ax = axes_flat[idx]
        vals = valid[:, col]
        mean_v = np.mean(vals)
        std_v = np.std(vals, ddof=1)
        ci = stats.t.interval(0.95, len(vals)-1, loc=mean_v, scale=std_v/np.sqrt(len(vals)))

        # KDE + rug
        kde_x = np.linspace(min(vals)-2*std_v, max(vals)+2*std_v, 200)
        kde = stats.gaussian_kde(vals)
        ax.fill_between(kde_x, kde(kde_x), alpha=0.25, color=VCF_COLOR)
        ax.plot(kde_x, kde(kde_x), color=VCF_COLOR, linewidth=2)
        ax.plot(vals, np.zeros_like(vals), '|', color='black', markersize=8, alpha=0.6)

        # Mean + CI
        ax.axvline(mean_v, color='red', linewidth=1.5, linestyle='--', label=f'Mean={mean_v:.2f}')
        ax.axvspan(ci[0], ci[1], alpha=0.12, color='red', label=f'95% CI [{ci[0]:.2f},{ci[1]:.2f}]')

        # Thesis value
        if name in vcf_mvtec_thesis:
            t_mean, t_std = vcf_mvtec_thesis[name]
            ax.axvline(t_mean, color='blue', linewidth=1, linestyle=':', label=f'Thesis {t_mean:.2f}')

        ax.set_xlabel(name)
        ax.set_ylabel('Density')
        ax.legend(fontsize=6.5, loc='upper left')
        ax.grid(alpha=0.3, linestyle='--')

    # Last subplot: summary bar chart with error bars
    ax = axes_flat[7]
    names_plot = ['AUROC-SP','F1Max-SP','AP-SP','AUROC-PX','F1Max-PX','AP-PX']
    means_plot = [np.mean(valid[:, c]) for c in [4,6,5,0,2,1]]
    stds_plot = [np.std(valid[:, c], ddof=1) for c in [4,6,5,0,2,1]]
    colors_plot = ['#364F6B']*3 + ['#FC5185']*3
    bars = ax.bar(names_plot, means_plot, color=colors_plot, edgecolor='white', alpha=0.85)
    ax.errorbar(names_plot, means_plot, yerr=np.array(stds_plot)*1.96/np.sqrt(len(valid)),
                fmt='none', ecolor='black', capsize=5, linewidth=1.2)
    ax.set_ylabel('Score (%)')
    ax.set_title('Mean ± 95% CI', fontweight='bold')
    for bar, m in zip(bars, means_plot):
        ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.3, f'{m:.1f}',
                ha='center', fontsize=7, fontweight='bold')
    ax.set_ylim(min(means_plot)-5, max(means_plot)+5)
    ax.grid(axis='y', alpha=0.3, linestyle='--')

    plt.tight_layout()
    fig.savefig(f'{OUT}/fig3_mvtec_multirun_distribution.png', facecolor='white')
    plt.close(fig)
    print(f'  [saved] {OUT}/fig3_mvtec_multirun_distribution.png')


# ═══════════════════════════════════════════════════════════════════════
#  FIGURE 4: Cross-dataset comparison (Bayes-VCF vs all baselines)
# ═══════════════════════════════════════════════════════════════════════

def fig4_cross_dataset_comparison():
    """Grouped bar: VCF vs baselines on YDFID-1, MVTec-AD, VisA."""
    datasets = ['YDFID-1', 'MVTec-AD', 'VisA']
    baselines_all = ['APRIL-GAN','CLIP-AD','AnomalyCLIP','AdaCLIP','Bayes-PFL','Bayes-VCF']

    # Aggregated values (use mean of per-class for YDFID, thesis table for MVTec/VisA)
    data = {
        'YDFID-1': {
            'AUROC-PX': {bl: np.mean(ydfid_auroc_px[bl]) for bl in baselines_all},
            'AP-PX':    {bl: np.mean(ydfid_ap_px[bl]) for bl in baselines_all},
            'AUPRO':    {bl: np.mean(ydfid_aupro[bl]) for bl in baselines_all},
        },
        'MVTec-AD': {
            'AUROC-SP': {**{bl: mvtec_baselines[bl]['AUROC-SP'] for bl in baselines_all[:5]},
                         'Bayes-VCF': vcf_mvtec_thesis['AUROC-SP'][0]},
            'AUROC-PX': {**{bl: mvtec_baselines[bl]['AUROC-PX'] for bl in baselines_all[:5]},
                         'Bayes-VCF': vcf_mvtec_thesis['AUROC-PX'][0]},
            'AP-PX':    {**{bl: mvtec_baselines[bl]['AP-PX'] for bl in baselines_all[:5]},
                         'Bayes-VCF': vcf_mvtec_thesis['AP-PX'][0]},
        },
        'VisA': {
            'AUROC-SP': {**{bl: visa_baselines[bl]['AUROC-SP'] for bl in baselines_all[:5]},
                         'Bayes-VCF': vcf_visa_thesis['AUROC-SP'][0]},
            'AUROC-PX': {**{bl: visa_baselines[bl]['AUROC-PX'] for bl in baselines_all[:5]},
                         'Bayes-VCF': vcf_visa_thesis['AUROC-PX'][0]},
            'AP-PX':    {**{bl: visa_baselines[bl]['AP-PX'] for bl in baselines_all[:5]},
                         'Bayes-VCF': vcf_visa_thesis['AP-PX'][0]},
        },
    }

    panels = [('YDFID-1','AUROC-PX'),('YDFID-1','AP-PX'),('YDFID-1','AUPRO'),
              ('MVTec-AD','AUROC-SP'),('MVTec-AD','AUROC-PX'),('MVTec-AD','AP-PX'),
              ('VisA','AUROC-SP'),('VisA','AUROC-PX'),('VisA','AP-PX')]

    fig, axes = plt.subplots(3, 3, figsize=(17, 11))
    fig.suptitle('Cross-dataset Comparison: Bayes-VCF vs. State-of-the-Art Methods',
                 fontweight='bold', y=1.01, fontsize=14)

    for idx, (ds, metric) in enumerate(panels):
        ax = axes[idx // 3][idx % 3]
        vals = [data[ds][metric][bl] for bl in baselines_all]
        colors = [BASELINE_COLORS[bl] for bl in baselines_all]
        bars = ax.bar(baselines_all, vals, color=colors, edgecolor='white', linewidth=0.5, alpha=0.9)

        # Highlight Bayes-VCF
        bars[-1].set_edgecolor('black')
        bars[-1].set_linewidth(2)

        ax.set_title(f'{ds} — {metric}', fontweight='bold', fontsize=10)
        ax.set_xticklabels(baselines_all, rotation=25, ha='right', fontsize=7)
        ax.grid(axis='y', alpha=0.3, linestyle='--')

        # Add value labels on bars for VCF
        for bar, v in zip(bars, vals):
            if bar == bars[-1]:
                ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.8,
                        f'{v:.1f}', ha='center', fontsize=7, fontweight='bold', color='red')

        # Significance marker
        pfl_val = data[ds][metric]['Bayes-PFL']
        vcf_val = data[ds][metric]['Bayes-VCF']
        if vcf_val > pfl_val:
            ax.annotate(f'Δ{vcf_val-pfl_val:+.1f}', xy=(4.5, max(vals)+1.5),
                        fontsize=7, ha='center', color='red', fontweight='bold')

    # Remove empty subplot if any
    for idx in range(len(panels), 9):
        axes[idx // 3][idx % 3].set_visible(False)

    plt.tight_layout()
    fig.savefig(f'{OUT}/fig4_cross_dataset_comparison.png', facecolor='white')
    plt.close(fig)
    print(f'  [saved] {OUT}/fig4_cross_dataset_comparison.png')


# ═══════════════════════════════════════════════════════════════════════
#  FIGURE 5: Effect size heatmap + significance matrix
# ═══════════════════════════════════════════════════════════════════════

def fig5_effect_size_heatmap():
    """Cohen's d + p-value heatmap for YDFID-1 pairwise comparisons."""
    baselines = ['APRIL-GAN','CLIP-AD','AnomalyCLIP','AdaCLIP','Bayes-PFL','Bayes-VCF']
    metrics_data = {'AUROC-PX': ydfid_auroc_px, 'AP-PX': ydfid_ap_px, 'AUPRO': ydfid_aupro}

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle("YDFID-1: Pairwise Cohen's d Effect Size & Wilcoxon Significance",
                 fontweight='bold', y=1.02, fontsize=13)

    for ax, (m_name, data) in zip(axes, metrics_data.items()):
        n = len(baselines)
        d_matrix = np.zeros((n, n))
        p_matrix = np.zeros((n, n))

        for i, bl_i in enumerate(baselines):
            for j, bl_j in enumerate(baselines):
                if i == j:
                    d_matrix[i, j] = 0
                    p_matrix[i, j] = 1.0
                else:
                    d_matrix[i, j] = cohens_d(np.array(data[bl_i]), np.array(data[bl_j]))
                    _, p_matrix[i, j], _ = wilcoxon_signed_rank(
                        np.array(data[bl_i]), np.array(data[bl_j]))

        # Mask upper triangle
        mask = np.triu(np.ones_like(d_matrix, dtype=bool), k=1)
        d_display = np.ma.array(d_matrix, mask=mask)

        im = ax.imshow(d_display, cmap='RdYlGn', aspect='auto', vmin=-2.5, vmax=2.5)
        ax.set_xticks(range(n))
        ax.set_yticks(range(n))
        ax.set_xticklabels(baselines, rotation=35, ha='right', fontsize=8)
        ax.set_yticklabels(baselines, fontsize=8)
        ax.set_title(m_name, fontweight='bold')

        # Annotate cells
        for i in range(n):
            for j in range(n):
                if i > j:
                    d_val = d_matrix[i, j]
                    p_val = p_matrix[i, j]
                    text = f'd={d_val:+.2f}\n{p_stars(p_val)}'
                    ax.text(j, i, text, ha='center', va='center', fontsize=7,
                            color='white' if abs(d_val) > 1.2 else 'black', fontweight='bold')

        plt.colorbar(im, ax=ax, shrink=0.82, label="Cohen's d")

    plt.tight_layout()
    fig.savefig(f'{OUT}/fig5_effect_size_heatmap.png', facecolor='white')
    plt.close(fig)
    print(f'  [saved] {OUT}/fig5_effect_size_heatmap.png')


# ═══════════════════════════════════════════════════════════════════════
#  FIGURE 6: Ablation — progressive gain + hyperparameter analysis
# ═══════════════════════════════════════════════════════════════════════

def fig6_ablation():
    """Ablation studies: module contribution + hyperparameter sensitivity."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Ablation Studies: Module Contribution & Hyperparameter Sensitivity',
                 fontweight='bold', y=1.01, fontsize=13)

    # ── (a) Module ablation: progressive gain ──
    ax = axes[0, 0]
    methods = ['Exper1\n(Baseline)','Exper2\n+Voga','Exper3\n+Cira','Exper4\n+Loss','Exper5\n+Voga+Cira','Ours\n(VCF)']
    x = np.arange(len(methods))
    width = 0.25

    for i, (metric, vals_dict) in enumerate(ablation_mvtec.items()):
        vals = [vals_dict[k] for k in ['Exper1','Exper2','Exper3','Exper4','Exper5','Ours']]
        offset = (i - 1) * width
        bars = ax.bar(x + offset, vals, width, label=metric, alpha=0.85, edgecolor='white', linewidth=0.5)

    ax.set_xticks(x)
    ax.set_xticklabels(methods, fontsize=8)
    ax.set_ylabel('Score (%)')
    ax.set_title('(a) Module Ablation on MVTec-AD', fontweight='bold')
    ax.legend(fontsize=8)
    ax.grid(axis='y', alpha=0.3, linestyle='--')

    # ── (b) Dilation rate ablation ──
    ax = axes[0, 1]
    dilations = ['[1,3,3]','[2,3,3]','[1,2,3]','[3,3,3]']
    d_x = np.arange(len(dilations))
    d_width = 0.2
    d_colors = ['#E76F51','#F4A261','#2A9D8F','#264653']

    for i, (metric, vals) in enumerate(ablation_dilation.items()):
        vals_list = [vals[d] for d in dilations]
        offset = (i - 1.5) * d_width
        ax.bar(d_x + offset, vals_list, d_width, label=metric, alpha=0.85,
               edgecolor='white', linewidth=0.5)

    ax.set_xticks(d_x)
    ax.set_xticklabels(dilations, fontsize=9)
    ax.set_ylabel('Score (%)')
    ax.set_title('(b) Dilation Rate Ablation', fontweight='bold')
    ax.legend(fontsize=7.5)
    ax.grid(axis='y', alpha=0.3, linestyle='--')

    # Highlight best
    best_x = 2  # [1,2,3]
    ax.axvspan(best_x - 0.45, best_x + 0.45, alpha=0.1, color='green')
    ax.annotate('Best', xy=(best_x, ax.get_ylim()[1]*0.99), fontsize=9,
                ha='center', color='green', fontweight='bold')

    # ── (c) Channel compression ablation ──
    ax = axes[1, 0]
    comps = ['1/2','1/4','1/8']
    c_x = np.arange(len(comps))
    c_width = 0.2

    for i, (metric, vals) in enumerate(ablation_compression.items()):
        vals_list = [vals[c] for c in comps]
        offset = (i - 1.5) * c_width
        ax.bar(c_x + offset, vals_list, c_width, label=metric, alpha=0.85,
               edgecolor='white', linewidth=0.5)

    ax.set_xticks(c_x)
    ax.set_xticklabels(comps, fontsize=10)
    ax.set_ylabel('Score (%)')
    ax.set_title('(c) Channel Compression Ratio Ablation', fontweight='bold')
    ax.legend(fontsize=7.5)
    ax.grid(axis='y', alpha=0.3, linestyle='--')

    best_x_c = 0  # 1/2
    ax.axvspan(best_x_c - 0.45, best_x_c + 0.45, alpha=0.1, color='green')
    ax.annotate('Best', xy=(best_x_c, ax.get_ylim()[1]*0.99), fontsize=9,
                ha='center', color='green', fontweight='bold')

    # ── (d) Summary: Bayes-VCF advantage decomposition ──
    ax = axes[1, 1]
    components = ['Voga\n(Table 5: Exp2-Exp1)', 'Cira\n(Table 5: Exp3-Exp1)',
                  'Loss\n(Table 5: Exp4-Exp1)', 'Voga+Cira\n(Table 5: Exp5-Exp1)',
                  'Full VCF\n(Table 5: Ours-Exp1)']
    gains_auc_sp = [+0.94, +1.12, +1.25, +2.02, +2.50]
    gains_auc_px = [+0.88, +0.75, +0.55, +1.56, +2.10]
    gains_aupro  = [+0.24, +0.35, +0.16, +0.51, +0.67]

    x_comp = np.arange(len(components))
    w = 0.25
    ax.bar(x_comp - w, gains_auc_sp, w, label='AUC-SP', color='#E76F51', alpha=0.85, edgecolor='white')
    ax.bar(x_comp, gains_auc_px, w, label='AUC-PX', color='#2A9D8F', alpha=0.85, edgecolor='white')
    ax.bar(x_comp + w, gains_aupro, w, label='AUPRO-PX', color='#264653', alpha=0.85, edgecolor='white')
    ax.axhline(y=0, color='gray', linewidth=0.8)
    ax.set_xticks(x_comp)
    ax.set_xticklabels(components, fontsize=8)
    ax.set_ylabel('Gain over Baseline (pp)')
    ax.set_title('(d) Component Contribution Decomposition', fontweight='bold')
    ax.legend(fontsize=8)
    ax.grid(axis='y', alpha=0.3, linestyle='--')

    plt.tight_layout()
    fig.savefig(f'{OUT}/fig6_ablation_studies.png', facecolor='white')
    plt.close(fig)
    print(f'  [saved] {OUT}/fig6_ablation_studies.png')


# ═══════════════════════════════════════════════════════════════════════
#  FIGURE 7: Cross-run consistency
# ═══════════════════════════════════════════════════════════════════════

def fig7_cross_run_consistency():
    """Three-run stability check for YDFID-1."""
    runs = ydfid_3runs
    metrics = ['AUROC-PX','AUROC-SP','AP-PX','AP-SP','F1Max-SP']
    run_labels = ['Run 1','Run 2','Run 3']
    colors = ['#E76F51','#2A9D8F','#264653']

    fig, axes = plt.subplots(2, 3, figsize=(17, 9))
    axes_flat = axes.flatten()
    fig.suptitle('YDFID-1: Cross-run Reproducibility (3 Independent Runs)',
                 fontweight='bold', y=1.01, fontsize=13)

    for idx, metric in enumerate(metrics):
        ax = axes_flat[idx]
        x = np.arange(len(ydfid_classes))
        w = 0.25
        for ri, rname in enumerate(['run1','run2','run3']):
            offset = (ri - 1) * w
            ax.bar(x + offset, runs[rname][metric], w, label=run_labels[ri],
                   color=colors[ri], alpha=0.85, edgecolor='white', linewidth=0.3)

        ax.set_xticks(x)
        ax.set_xticklabels(ydfid_classes, rotation=45, ha='right', fontsize=7)
        ax.set_ylabel(metric)
        ax.set_title(metric, fontweight='bold')
        ax.legend(fontsize=7)
        ax.grid(axis='y', alpha=0.3, linestyle='--')

    # Last subplot: run-wise mean ± std summary
    ax = axes_flat[5]
    run_means = {m: [np.mean(runs[r][m]) for r in ['run1','run2','run3']] for m in metrics}
    run_stds = {m: [np.std(runs[r][m], ddof=1) for r in ['run1','run2','run3']] for m in metrics}
    x_sum = np.arange(len(metrics))
    w_sum = 0.25
    for ri, (rname, c) in enumerate(zip(['run1','run2','run3'], colors)):
        means = [run_means[m][ri] for m in metrics]
        stds = [run_stds[m][ri] for m in metrics]
        offset = (ri - 1) * w_sum
        ax.bar(x_sum + offset, means, w_sum, yerr=stds, label=run_labels[ri],
               color=c, alpha=0.85, edgecolor='white', linewidth=0.5,
               capsize=4, error_kw={'linewidth': 1})

    ax.set_xticks(x_sum)
    ax.set_xticklabels(metrics, fontsize=9)
    ax.set_ylabel('Score (%)')
    ax.set_title('Mean ± Std Across Runs', fontweight='bold')
    ax.legend(fontsize=8)
    ax.grid(axis='y', alpha=0.3, linestyle='--')

    plt.tight_layout()
    fig.savefig(f'{OUT}/fig7_cross_run_consistency.png', facecolor='white')
    plt.close(fig)
    print(f'  [saved] {OUT}/fig7_cross_run_consistency.png')


# ═══════════════════════════════════════════════════════════════════════
#  TEXT OUTPUTS (same as before)
# ═══════════════════════════════════════════════════════════════════════

def print_all_tests():
    """Console statistical test output."""
    print("\n" + "█" * 82)
    print("█  STATISTICAL SIGNIFICANCE ANALYSIS: Bayes-VCF")
    print("█" * 82)

    # ── Test 1: YDFID-1 per-class Wilcoxon ──
    print("\n" + "=" * 82)
    print("TEST 1: YDFID-1 — Wilcoxon signed-rank (per-class, Bayes-VCF vs baselines)")
    print("=" * 82)
    baselines = ['APRIL-GAN','CLIP-AD','AnomalyCLIP','AdaCLIP','Bayes-PFL']
    all_pvals = []
    for metric_name, data in [('AUROC-PX',ydfid_auroc_px),('AP-PX',ydfid_ap_px),('AUPRO',ydfid_aupro)]:
        print(f"\n  {metric_name}  |  VCF mean={np.mean(data['Bayes-VCF']):.2f}")
        vcf = np.array(data['Bayes-VCF'])
        for bl in baselines:
            bl_vals = np.array(data[bl])
            stat, p, sig = wilcoxon_signed_rank(vcf, bl_vals, alternative='greater')
            d = cohens_d(vcf, bl_vals)
            all_pvals.append(p)
            diff = np.mean(vcf)-np.mean(bl_vals)
            print(f"    vs {bl:<14s}  W={stat:6.1f}  p={p:.6f}  {'***' if sig else '   '}  "
                  f"d={d:+.3f} ({effect_size_label(d):>9s})  Δ={diff:+.2f}")
    adj_a, sigs = bonferroni([p for p in all_pvals if not np.isnan(p)])
    print(f"\n  Bonferroni adjusted α = {adj_a:.6f} (m={len(sigs)})")

    # ── Test 2: MVTec-AD multi-run ──
    print("\n" + "=" * 82)
    print("TEST 2: MVTec-AD — Multi-run analysis")
    print("=" * 82)
    runs = np.array(mvtec_runs_raw)
    valid = runs[runs[:,7]>0.001]
    print(f"  Valid runs: {len(valid)}/{len(runs)}")
    col_map = {0:'AUROC-PX',1:'AP-PX',2:'F1Max-PX',3:'IoU',4:'AUROC-SP',5:'AP-SP',6:'F1Max-SP'}
    for col, name in col_map.items():
        v = valid[:,col]
        ci = stats.t.interval(0.95, len(v)-1, loc=np.mean(v), scale=np.std(v,ddof=1)/np.sqrt(len(v)))
        print(f"  {name:<14s} mean={np.mean(v):.2f}±{np.std(v,ddof=1):.2f}  CI=[{ci[0]:.2f},{ci[1]:.2f}]")
        if name in vcf_mvtec_thesis:
            t_mean, t_std = vcf_mvtec_thesis[name]
            t_stat, t_p = stats.ttest_1samp(v, t_mean)
            print(f"           vs thesis {t_mean:.2f}±{t_std:.2f}: t={t_stat:+.3f}, p={t_p:.4f}  "
                  f"{'(consistent)' if t_p>=0.05 else '(DIFFERS)'}")

    # ── Test 3: Sub-class ──
    print("\n" + "=" * 82)
    print("TEST 3: YDFID-1 — Sub-class analysis (VCF vs PFL)")
    print("=" * 82)
    for metric_name, data in [('AUROC-PX',ydfid_auroc_px),('AP-PX',ydfid_ap_px),('AUPRO',ydfid_aupro)]:
        print(f"\n  {metric_name}:")
        for sc,(s,e) in subclass_ranges.items():
            vcf = np.array(data['Bayes-VCF'][s:e]); pfl = np.array(data['Bayes-PFL'][s:e])
            stat,p,sig = wilcoxon_signed_rank(vcf,pfl,alternative='greater')
            d = cohens_d(vcf,pfl)
            print(f"    {sc}: VCF={np.mean(vcf):.2f} PFL={np.mean(pfl):.2f} Δ={np.mean(vcf)-np.mean(pfl):+.2f}  "
                  f"W={stat:.1f} p={p:.4f} d={d:+.3f} {'*' if sig else ''}")

    # ── Test 4: Cross-run consistency ──
    print("\n" + "=" * 82)
    print("TEST 4: YDFID-1 — Cross-run consistency (ANOVA)")
    print("=" * 82)
    for metric in ['AUROC-PX','AUROC-SP','AP-PX','AP-SP','F1Max-SP']:
        r1,r2,r3 = [np.array(ydfid_3runs[r][metric]) for r in ['run1','run2','run3']]
        f_stat,p_anova = stats.f_oneway(r1,r2,r3)
        print(f"  {metric}: F={f_stat:.3f} p={p_anova:.4f}  "
              f"r1={np.mean(r1):.2f} r2={np.mean(r2):.2f} r3={np.mean(r3):.2f}  "
              f"{'(consistent)' if p_anova>=0.05 else '(DIFFERS)'}")

    # ── Test 5: Ablation Friedman ──
    print("\n" + "=" * 82)
    print("TEST 5: Ablation — Friedman test")
    print("=" * 82)
    methods_list = ['Exper1','Exper2','Exper3','Exper4','Exper5','Ours']
    metrics_list = ['AUC-SP','AUC-PX','AUPRO-PX']
    data_mat = np.array([[ablation_mvtec[m][e] for e in methods_list] for m in metrics_list])
    f_stat,f_p = stats.friedmanchisquare(*[data_mat[:,j] for j in range(len(methods_list))])
    print(f"  Friedman χ²={f_stat:.3f}, p={f_p:.6f}  {'***' if f_p<0.05 else 'ns'}")

    # ── Test 6: Summary ──
    print("\n" + "=" * 82)
    print("TEST 6: SUMMARY — Bayes-VCF vs Bayes-PFL")
    print("=" * 82)
    print(f"  {'Dataset':<12s} {'Metric':<14s} {'PFL':>8s} {'VCF':>8s} {'Δ':>8s} {'p-val':>10s} {'Cohen d':>8s}")
    for metric_name, data in [('AUROC-PX',ydfid_auroc_px),('AP-PX',ydfid_ap_px),('AUPRO',ydfid_aupro)]:
        vcf=np.array(data['Bayes-VCF']); pfl=np.array(data['Bayes-PFL'])
        _,p,_=wilcoxon_signed_rank(vcf,pfl,alternative='greater')
        d=cohens_d(vcf,pfl)
        print(f"  {'YDFID-1':<12s} {metric_name:<14s} {np.mean(pfl):8.2f} {np.mean(vcf):8.2f} {np.mean(vcf)-np.mean(pfl):+8.2f} {p:10.6f} {d:+8.3f}")
    for name,pfl_v,vcf_v in [('AUROC-SP',92.30,93.41),('F1Max-SP',93.10,93.30),('AUROC-PX',91.80,92.69),('F1Max-PX',87.40,88.50),('AP-PX',48.30,49.30)]:
        print(f"  {'MVTec-AD':<12s} {name:<14s} {pfl_v:8.2f} {vcf_v:8.2f} {vcf_v-pfl_v:+8.2f} {'N/A':>10s} {'N/A':>8s}")
    for name,pfl_v,vcf_v in [('AUROC-SP',87.0,88.10),('AUROC-PX',95.6,95.70),('AP-PX',29.8,30.20)]:
        print(f"  {'VisA':<12s} {name:<14s} {pfl_v:8.2f} {vcf_v:8.2f} {vcf_v-pfl_v:+8.2f} {'N/A':>10s} {'N/A':>8s}")

    print("\n" + "█" * 82)
    print("█  Analysis complete. Figures saved to ./figures/")
    print("█" * 82)


# ═══════════════════════════════════════════════════════════════════════
if __name__ == '__main__':
    print_all_tests()
    print("\n  Generating figures...")
    fig1_ydfid_per_class_boxplot()
    # fig2_ydfid_paired_diff()
    # fig3_mvtec_multirun()
    # fig4_cross_dataset_comparison()
    # fig5_effect_size_heatmap()
    # fig6_ablation()
    # fig7_cross_run_consistency()
    print("\n  Done. All figures saved to ./figures/")
