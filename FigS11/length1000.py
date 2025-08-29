from matplotlib import pyplot as plt
import numpy as np

def plot_evolution(ax, x, y, label, color, linestyle, alpha=1.0):
    # max_episode = 50
    # x = [value * 224 for value in range(max_episode)]
    ax.plot(x, y, color=color, linestyle=linestyle, linewidth=1.5, alpha=alpha, label=label)

def plot_evolution_repeat(ax, x, yi, label_stra, label, color, linestyle,  alpha=0.1):
    # max_episode = 50
    return
    repeattimes = 8
    for irepeat in range(repeattimes):
        # x = [value * 224 for value in range(max_episode)]
        ax.plot(x, yi[irepeat][label_stra], color=color, linestyle=linestyle, linewidth=0.3, alpha=alpha, label=label)

plt.rcParams['font.family'] = 'Arial'
plt.rc('font', size=11)
max_episode = 10000

y = np.load('length1000.npy', allow_pickle=True).tolist()






# fig, axs = plt.subplots(2, 2, figsize=(6, 6))

# subplot_size = (4, 3)
# fig, axs = plt.subplots(1, 3, figsize=(subplot_size[0] * 3, subplot_size[1] * 1), gridspec_kw={'width_ratios': [subplot_size[0]] * 3, 'height_ratios': [subplot_size[1]] * 1})
# # fig, axs = plt.subplots(2, 2, figsize=(6, 6))
# fig.subplots_adjust(wspace=0.2,hspace=0.2)

fig, axs = plt.subplots(1, 1, figsize=(3, 3))  # 每个子图 3x3，共 3 列 → 总宽度9，高度3
fig.subplots_adjust(wspace=0.2, hspace=0.2)   # 控制子图间距
# 左上角
ax_main = axs
ax_main.tick_params(axis='both', direction='in', width=0.5)
ax_main.set_title(r'Interaction length = 1000', fontsize=11)
ax_main.set_xlabel('Generations', fontsize=11)
ax_main.set_ylabel('Fraction of strategies', fontsize=11)
ax_main.set_xlim(-224 ,224*max_episode/2)

import numpy as np
import matplotlib.ticker as mticker

# 1) 关闭科学计数法 + 偏移
ax_main.ticklabel_format(style='plain', axis='x', useOffset=False)  # 有些版本已足够
fmt = mticker.ScalarFormatter(useOffset=False)
fmt.set_scientific(False)
ax_main.xaxis.set_major_formatter(fmt)

# 2) 有些版本仍会显示右上角的 "1e6" offset，直接隐藏它
ax_main.get_xaxis().get_offset_text().set_visible(False)


ax_main.set_ylim(-0.02,1.02)
ax_main.spines['top'].set_linewidth(0.5)  # 边框粗细
ax_main.spines['right'].set_linewidth(0.5)
ax_main.spines['bottom'].set_linewidth(0.5)
ax_main.spines['left'].set_linewidth(0.5)


x = [value * 224 for value in range(max_episode)]

plot_evolution(ax_main, x, y[0], 'MTBR', 'green', '-', alpha=1.0)

plot_evolution(ax_main, x, y[2], 'GTFT0.3', 'blue', '-', alpha=1.0)

plot_evolution(ax_main, x, y[7], 'Gradual TFT', 'red', '-', alpha=1.0)
plot_evolution(ax_main, x, [sum(values) for values in zip(y[1], y[3], y[4], y[5], y[6], y[8], y[9], y[10], y[11], y[12], y[13], y[14], y[15])]
               , 'Other strategies', 'gray', '-', alpha=1.0)



from matplotlib.lines import Line2D

legend_elements = [
    Line2D([0], [0], color='green', linestyle='-', label='MTBR'),
    Line2D([0], [0], color='blue', linestyle='-', label='GTFT0.3'),
    Line2D([0], [0], color='red', linestyle='-', label='Gradual TFT'),
    Line2D([0], [0], color='gray', linestyle='-', label='Other strategies')
]

ax_main.legend(handles=legend_elements, loc='best', fontsize=9, frameon=False)










# plt.savefig('test1.pdf', format='pdf', dpi=300)
plt.tight_layout()
plt.show()
