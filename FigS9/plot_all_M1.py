
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import matplotlib.ticker as ticker
import itertools


AVG_FILE   = "avg_num_sparse_mean_10initial.npy"   # K x S
STEPS_FILE = "record_steps_10initial.npy"          # K
MTBR_INDEX = 0                           
THRESHOLD  = 0.1                       
SAVE_PATH  = "evolution_10initial.pdf"  
DPI        = 600


GRID = np.linspace(0.0, 1.0, 11)

def decode_m1_probs(global_idx, offset_custom=8):

    if global_idx < offset_custom:
        return None
    m = global_idx - offset_custom
    i0 = m // 1331
    m  = m % 1331
    i1 = m // 121
    m  = m % 121
    i2 = m // 11
    i3 = m % 11
    return (float(GRID[i0]), float(GRID[i1]), float(GRID[i2]), float(GRID[i3]))

def strategy_label(global_idx,
                   custom_names=None,
                   decimals=1,
                   short=True,
                   offset_custom=8):

    if custom_names is None:
        custom_names = {
            0: 'MTBR',
            1: 'Holds a grudge',
            2: 'Fool me once',
            3: 'OmegaTFT',
            4: 'Gradual',
            5: 'CURE',
            6: 'AON-2',
            7: 'Reactive-2',
        }
    if global_idx in custom_names:
        return custom_names[global_idx]
    p = decode_m1_probs(global_idx, offset_custom=offset_custom)
    if p is None:
        return f"custom_{global_idx}"
    p = tuple(round(x, decimals) for x in p)
    if short:

        return f"({', '.join(map(str, p))})"
    else:
        return f"M1(CC={p[0]}, CD={p[1]}, DC={p[2]}, DD={p[3]})"

get_label = strategy_label


avg_num = np.load(AVG_FILE)        # (K, S)
steps   = np.load(STEPS_FILE)      # (K,)
K, S = avg_num.shape

# counts -> fraction
tot = np.sum(avg_num, axis=1, keepdims=True)  # (K,1)
tot = np.maximum(tot, 1e-12)
frac = avg_num / tot                          # (K, S)


mask = steps > 0
steps_plot = steps[mask]
frac_plot  = frac[mask]


peaks = np.nanmax(frac_plot, axis=0)   # (S,)
selected = np.where(peaks >= THRESHOLD)[0].tolist()
if MTBR_INDEX not in selected:
    selected.append(MTBR_INDEX)


plt.rcParams['font.family'] = 'Arial'
plt.rc('font', size=11)

fig, ax = plt.subplots(figsize=(6.2, 4.6))  
for spine in ax.spines.values():
    spine.set_linewidth(0.6)

ax.tick_params(axis='both', direction='in', width=0.6, length=4)
ax.set_xlabel('Generations', fontsize=11)
ax.set_ylabel('Fraction of strategies', fontsize=11)
ax.set_title('Evolution of MTBR in populations of memory-1 strategies', fontsize=11)


ax.set_xscale('log')
xmin = steps_plot.min()
xmax = steps_plot.max()
ax.set_xlim(max(1.0, xmin), xmax * 1.02)
exp_min = int(np.floor(np.log10(max(1.0, xmin))))
exp_max = int(np.ceil(np.log10(xmax)))
xticks = [10**e for e in range(exp_min, exp_max + 1)]
ax.set_xticks(xticks)
ax.xaxis.set_major_formatter(ticker.LogFormatterMathtext())
ax.set_ylim(-0.02, 1.02)


mtbr_color = 'tab:green'

color_pool = ['tab:blue','tab:orange','tab:red','tab:purple',
              'tab:brown','tab:pink','tab:gray','tab:olive','tab:cyan']
color_cycle = itertools.cycle(color_pool)


draw_order = ([MTBR_INDEX] if MTBR_INDEX in selected else []) + [i for i in selected if i != MTBR_INDEX]

legend_handles = []
for idx in draw_order:
    if idx == MTBR_INDEX:
        lbl = get_label(idx)
        ax.plot(steps_plot, frac_plot[:, idx],
                color=mtbr_color, linewidth=1.8, alpha=1.0, zorder=5, label=lbl)
        legend_handles.append(Line2D([0],[0], color=mtbr_color, linewidth=1.8, label=lbl))
    else:
        c = next(color_cycle)
        lbl = get_label(idx)
        ax.plot(steps_plot, frac_plot[:, idx],
                color=c, linewidth=1.0, alpha=0.95, zorder=3, label=lbl)
        legend_handles.append(Line2D([0],[0], color=c, linewidth=1.2, label=lbl))


ax.legend(handles=legend_handles, loc='best', fontsize=9, frameon=False)

plt.tight_layout(pad=1.0)
if SAVE_PATH:
    plt.savefig(SAVE_PATH, bbox_inches='tight')
# plt.show()
