# -*- coding: utf-8 -*-
"""
Created on Fri Jan  5 23:14:06 2024

@author: 63284
"""

from matplotlib import pyplot as plt
import numpy as np

def plot_evolution(ax, x, y, label, color, linestyle, alpha=1.0):
    # max_episode = 50
    # x = [value * 224 for value in range(max_episode)]
    ax.plot(x, y, color=color, linestyle=linestyle, linewidth=1.5, alpha=alpha, label=label)

def plot_evolution_repeat(ax, x, yi, label_stra, label, color, linestyle,  alpha=0.1):
    # max_episode = 50
    repeattimes = 50
    for irepeat in range(repeattimes):
        # x = [value * 224 for value in range(max_episode)]
        ax.plot(x, yi[irepeat][label_stra], color=color, linestyle=linestyle, linewidth=0.3, alpha=alpha, label=label)

plt.rcParams['font.family'] = 'Arial'
plt.rc('font', size=11)


# 有MTBR
y_0001 = np.load('mutation/mu0001.npy', allow_pickle=True).tolist()

yi_0001 = np.load('mutation/mu0001_50.npy', allow_pickle=True).tolist()

# 无MTBR
y_001 = np.load('mutation/mu001.npy', allow_pickle=True).tolist()

yi_001 = np.load('mutation/mu001_50.npy', allow_pickle=True).tolist()


y_01 = np.load('mutation/mu01.npy', allow_pickle=True).tolist()
yi_01 = np.load('mutation/mu01_50.npy', allow_pickle=True).tolist()

# fig, axs = plt.subplots(2, 2, figsize=(6, 6))

# subplot_size = (4, 3)
# fig, axs = plt.subplots(1, 3, figsize=(subplot_size[0] * 3, subplot_size[1] * 1), gridspec_kw={'width_ratios': [subplot_size[0]] * 3, 'height_ratios': [subplot_size[1]] * 1})
# # fig, axs = plt.subplots(2, 2, figsize=(6, 6))
# fig.subplots_adjust(wspace=0.2,hspace=0.2)

fig, axs = plt.subplots(1, 3, figsize=(8, 3))  # 每个子图 3x3，共 3 列 → 总宽度9，高度3
fig.subplots_adjust(wspace=0.2, hspace=0.2)   # 控制子图间距
# 左上角
ax_0001 = axs[0]
ax_0001.tick_params(axis='both', direction='in', width=0.5)
ax_0001.set_title(r'Mutation rate $\mu$ = 0.001', fontsize=11)
ax_0001.set_xlabel('Generations', fontsize=11)
ax_0001.set_ylabel('Fraction of strategies', fontsize=11)
ax_0001.set_xlim(-224,22400)
ax_0001.set_ylim(-0.02,1.02)
ax_0001.spines['top'].set_linewidth(0.5)  # 边框粗细
ax_0001.spines['right'].set_linewidth(0.5)
ax_0001.spines['bottom'].set_linewidth(0.5)
ax_0001.spines['left'].set_linewidth(0.5)


x = [value * 224 for value in range(100)]

plot_evolution(ax_0001, x, y_0001[0], 'MTBR', 'green', '-', alpha=1.0)

plot_evolution(ax_0001, x, y_0001[2], 'GTFT0.3', 'blue', '-', alpha=1.0)

plot_evolution(ax_0001, x, y_0001[7], 'Gradual TFT', 'red', '-', alpha=1.0)
plot_evolution(ax_0001, x, [sum(values) for values in zip(y_0001[1], y_0001[3], y_0001[4], y_0001[5], y_0001[6], y_0001[8], y_0001[9], y_0001[10], y_0001[11], y_0001[12], y_0001[13], y_0001[14], y_0001[15])]
               , 'Other strategies', 'gray', '-', alpha=1.0)

plot_evolution_repeat(ax_0001, x, yi_0001, 0, 'MTBR', color='green',linestyle="-")
plot_evolution_repeat(ax_0001, x, yi_0001, 2, 'GTFT0.3', color='blue', linestyle="-")
plot_evolution_repeat(ax_0001, x, yi_0001, 7, 'Gradual TFT', color='red', linestyle="-")

from matplotlib.lines import Line2D

legend_elements = [
    Line2D([0], [0], color='green', linestyle='-', label='MTBR'),
    Line2D([0], [0], color='blue', linestyle='-', label='GTFT0.3'),
    Line2D([0], [0], color='red', linestyle='-', label='Gradual TFT'),
    Line2D([0], [0], color='gray', linestyle='-', label='Other strategies')
]

ax_0001.legend(handles=legend_elements, loc='best', fontsize=9, frameon=False)

# 右上角
ax_001 = axs[1]
ax_001.tick_params(axis='both', direction='in', width=0.5)
ax_001.set_title(r'Mutation rate $\mu$ = 0.01', fontsize=11)
ax_001.set_xlabel('Generations', fontsize=11)
# ax_001.set_ylabel('Fraction of strategies', fontsize=11)
ax_001.set_xlim(-224,22400)
ax_001.set_ylim(-0.02,1.02)
ax_001.spines['top'].set_linewidth(0.5)  # 边框粗细
ax_001.spines['right'].set_linewidth(0.5)
ax_001.spines['bottom'].set_linewidth(0.5)
ax_001.spines['left'].set_linewidth(0.5)
x = [value * 224 for value in range(100)]

plot_evolution(ax_001, x, y_001[0], 'MTBR', 'green', '-', alpha=1.0)

plot_evolution(ax_001, x, y_001[2], 'GTFT0.3', 'blue', '-', alpha=1.0)

plot_evolution(ax_001, x, y_001[7], 'Gradual TFT', 'red', '-', alpha=1.0)
plot_evolution(ax_001, x, [sum(values) for values in zip(y_001[1], y_001[3], y_001[4], y_001[5], y_001[6], y_001[8], y_001[9], y_001[10], y_001[11], y_001[12], y_001[13], y_001[14], y_001[15])],
               'Other strategies', 'gray', '-', alpha=1.0)

plot_evolution_repeat(ax_001, x, yi_001, 0, 'MTBR', color='green', linestyle="-")
plot_evolution_repeat(ax_001, x, yi_001, 2, 'GTFT0.3', color='blue', linestyle="-")
plot_evolution_repeat(ax_001, x, yi_001, 7, 'Gradual TFT', color='red', linestyle="-")






# 左下角
ax_01 = axs[2]
ax_01.tick_params(axis='both', direction='in', width=0.5)
ax_01.set_title(r'Mutation rate $\mu$ = 0.1', fontsize=11)
ax_01.set_xlabel('Generations', fontsize=11)
# ax_01.set_ylabel('Fraction of strategies', fontsize=11)
ax_01.set_xlim(-224,22400)
ax_01.set_ylim(-0.02,1.02)
ax_01.spines['top'].set_linewidth(0.5)  # 边框粗细
ax_01.spines['right'].set_linewidth(0.5)
ax_01.spines['bottom'].set_linewidth(0.5)
ax_01.spines['left'].set_linewidth(0.5)
x = [value * 224 for value in range(100)]

plot_evolution(ax_01, x, y_01[0], 'MTBR', 'green', '-', alpha=1.0)

plot_evolution(ax_01, x, y_01[2], 'GTFT0.3', 'blue', '-', alpha=1.0)

plot_evolution(ax_01, x, y_01[7], 'Gradual TFT', 'red', '-', alpha=1.0)
plot_evolution(ax_01, x, [sum(values) for values in zip(y_01[1], y_01[3], y_01[4], y_01[5], y_01[6], y_01[8], y_01[9], y_01[10], y_01[11], y_01[12], y_01[13], y_01[14], y_01[15])],
               'Other strategies', 'gray', '-', alpha=1.0)

plot_evolution_repeat(ax_01, x, yi_01, 0, 'MTBR', color='green', linestyle="-")
plot_evolution_repeat(ax_01, x, yi_01, 2, 'GTFT0.3', color='blue', linestyle="-")
plot_evolution_repeat(ax_01, x, yi_01, 7, 'Gradual TFT', color='red', linestyle="-")







# 画abcd


ax_0001.text(-0.2, 1.08, 'a', transform=axs[0].transAxes,
            fontsize=11, fontweight='bold', va='center', ha='center')
ax_001.text(-0.2, 1.08, 'b', transform=axs[1].transAxes,
            fontsize=11, fontweight='bold', va='center', ha='center')
ax_01.text(-0.2, 1.08, 'c', transform=axs[2].transAxes,
            fontsize=11, fontweight='bold', va='center', ha='center')



# plt.savefig('test1.pdf', format='pdf', dpi=300)
plt.tight_layout()
plt.show()