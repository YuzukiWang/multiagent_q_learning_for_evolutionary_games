
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
        ax.plot(x, yi[irepeat][label_stra], color=color, linestyle=linestyle, linewidth=1.5, alpha=alpha, label=label)

plt.rcParams['font.family'] = 'Arial'
plt.rc('font', size=11)



y = np.load('15_interaction_length.npy', allow_pickle=True).tolist()

yi = np.load('15_interaction_length_50_times.npy', allow_pickle=True).tolist()


y_no_MTBR = np.load('25_interaction_length.npy', allow_pickle=True).tolist()

yi_no_MTBR = np.load('25_interaction_length_50_times.npy', allow_pickle=True).tolist()


y_random = np.load('stochastic_avg_20_interaction_length.npy', allow_pickle=True).tolist()
yi_random = np.load('stochastic_avg_20_interaction_length_50_times.npy', allow_pickle=True).tolist()

# fig, axs = plt.subplots(2, 2, figsize=(6, 6))

subplot_size = (3, 3)
fig, axs = plt.subplots(2, 2, figsize=(subplot_size[0] * 2, subplot_size[1] * 2), gridspec_kw={'width_ratios': [subplot_size[0]] * 2, 'height_ratios': [subplot_size[1]] * 2})
# fig, axs = plt.subplots(2, 2, figsize=(6, 6))
fig.subplots_adjust(wspace=0.2,hspace=0.2)

ax_main = axs[0, 0]
ax_main.tick_params(axis='both', direction='in', width=0.5)
ax_main.set_title('Interaction length = 15', fontsize=11)
ax_main.set_xlabel('Generations', fontsize=11)
ax_main.set_ylabel('Fraction of strategies', fontsize=11)
ax_main.set_xlim(-224,11200)
ax_main.set_ylim(-0.02,1.02)
ax_main.spines['top'].set_linewidth(0.5)  
ax_main.spines['right'].set_linewidth(0.5)
ax_main.spines['bottom'].set_linewidth(0.5)
ax_main.spines['left'].set_linewidth(0.5)
x = [value * 224 for value in range(50)]

plot_evolution(ax_main, x, y[0], 'MTBR', 'black', '-', alpha=1.0)
plot_evolution(ax_main, x, y[1], 'TFT', '#0000FF', '-', alpha=1.0)
plot_evolution(ax_main, x, y[3], 'WSLS', '#00FFFF', '-', alpha=1.0)
plot_evolution(ax_main, x, y[4], 'Holds a grudge', '#FF0000', '-', alpha=1.0)
plot_evolution(ax_main, x, y[5], 'Fool me once', '#00FF80', '-', alpha=1.0)
plot_evolution(ax_main, x, y[6], 'Omega TFT', '#FFFF00', '-', alpha=1.0)
plot_evolution(ax_main, x, y[7], 'Gradual TFT', '#FF00FF', '-', alpha=1.0)

color_zd = plt.cm.viridis(np.linspace(0, 1, 15))
linestyle_zd = ['--', '-.', ':']
for i, label in enumerate(['ZDExtort2', 'ZDExtort2v2', 'ZDExtort3', 'ZDExtort4', 'ZDGen2', 'ZDGTFT2', 'ZDMischief', 'ZDSet2']):
    plot_evolution(ax_main, x, y[8 + i], label, color_zd[i % 8], linestyle_zd[i % 3], alpha=1.0)

plot_evolution_repeat(ax_main, x, yi, 0, 'MTBR', color='black',linestyle="-")
plot_evolution_repeat(ax_main, x, yi, 7, 'Gradual TFT', color='#FF00FF', linestyle="-")


ax_no_MTBR = axs[0, 1]
ax_no_MTBR.tick_params(axis='both', direction='in', width=0.5)
ax_no_MTBR.set_title('Interaction length = 25', fontsize=11)
ax_no_MTBR.set_xlabel('Generations', fontsize=11)
ax_no_MTBR.set_ylabel('Fraction of strategies', fontsize=11)
ax_no_MTBR.set_xlim(-224,11200)
ax_no_MTBR.set_ylim(-0.02,1.02)
ax_no_MTBR.spines['top'].set_linewidth(0.5)  
ax_no_MTBR.spines['right'].set_linewidth(0.5)
ax_no_MTBR.spines['bottom'].set_linewidth(0.5)
ax_no_MTBR.spines['left'].set_linewidth(0.5)
plot_evolution(ax_no_MTBR, x, y_no_MTBR[0], 'MTBR', 'black', '-', alpha=1.0)
plot_evolution(ax_no_MTBR, x, y_no_MTBR[1], 'TFT', '#0000FF', '-', alpha=1.0)
plot_evolution(ax_no_MTBR, x, y_no_MTBR[3], 'WSLS', '#00FFFF', '-', alpha=1.0)
plot_evolution(ax_no_MTBR, x, y_no_MTBR[4], 'Holds a grudge', '#FF0000', '-', alpha=1.0)
plot_evolution(ax_no_MTBR, x, y_no_MTBR[5], 'Fool me once', '#00FF80', '-', alpha=1.0)
plot_evolution(ax_no_MTBR, x, y_no_MTBR[6], 'Omega TFT', '#FFFF00', '-', alpha=1.0)
plot_evolution(ax_no_MTBR, x, y_no_MTBR[7], 'Gradual TFT', '#FF00FF', '-', alpha=1.0)

color_zd = plt.cm.viridis(np.linspace(0, 1, 15))
linestyle_zd = ['--', '-.', ':']
for i, label in enumerate(['ZDExtort2', 'ZDExtort2v2', 'ZDExtort3', 'ZDExtort4', 'ZDGen2', 'ZDGTFT2', 'ZDMischief', 'ZDSet2']):
    plot_evolution(ax_no_MTBR, x, y_no_MTBR[8 + i], label, color_zd[i % 8], linestyle_zd[i % 3], alpha=1.0)

plot_evolution_repeat(ax_no_MTBR, x, yi_no_MTBR, 0, 'MTBR', color='black',linestyle="-")
plot_evolution_repeat(ax_no_MTBR, x, yi_no_MTBR, 7, 'Gradual TFT', color='#FF00FF', linestyle="-")





ax_random = axs[1, 0]
ax_random.tick_params(axis='both', direction='in', width=0.5)
ax_random.set_title('Stochastic interaction lengths', fontsize=11)
ax_random.set_xlabel('Generations', fontsize=11)
ax_random.set_ylabel('Fraction of strategies', fontsize=11)
ax_random.set_xlim(-224, 11200)
ax_random.set_ylim(-0.02, 1.02)
ax_random.spines['top'].set_linewidth(0.5)  
ax_random.spines['right'].set_linewidth(0.5)
ax_random.spines['bottom'].set_linewidth(0.5)
ax_random.spines['left'].set_linewidth(0.5)

x = [value * 224*5/3.0 for value in range(30)]
plot_evolution(ax_random, x, y_random[0][:30], 'MTBR', 'black', '-', alpha=1.0)
plot_evolution(ax_random, x, y_random[1][:30], 'TFT', '#0000FF', '-', alpha=1.0)
plot_evolution(ax_random, x, y_random[3][:30], 'WSLS', '#00FFFF', '-', alpha=1.0)
plot_evolution(ax_random, x, y_random[4][:30], 'Holds a grudge', '#FF0000', '-', alpha=1.0)
plot_evolution(ax_random, x, y_random[5][:30], 'Fool me once', '#00FF80', '-', alpha=1.0)
plot_evolution(ax_random, x, y_random[6][:30], 'Omega TFT', '#FFFF00', '-', alpha=1.0)
plot_evolution(ax_random, x, y_random[7][:30], 'Gradual TFT', '#FF00FF', '-', alpha=1.0)

color_zd = plt.cm.viridis(np.linspace(0, 1, 15))
linestyle_zd = ['--', '-.', ':']
for i, label in enumerate(['ZDExtort2', 'ZDExtort2v2', 'ZDExtort3', 'ZDExtort4', 'ZDGen2', 'ZDGTFT2', 'ZDMischief', 'ZDSet2']):
    plot_evolution(ax_random, x, y_random[8 + i][:30], label, color_zd[i % 8], linestyle_zd[i % 3], alpha=1.0)
    
yi_random_30 = [{k: v[:30] for k, v in d.items()} for d in yi_random]

plot_evolution_repeat(ax_random, x, yi_random_30, 0, 'MTBR', color='black',linestyle="-")
plot_evolution_repeat(ax_random, x, yi_random_30, 7, 'Gradual TFT', color='#FF00FF', linestyle="-")










ax_legend = axs[1, 1]
ax_legend.axis('off')

color_zd = plt.cm.viridis(np.linspace(0, 1, 15))
linestyle_zd = ['--', '-.', ':']
legend_labels = [
    ("MTBR", 'black', '-', None),
    ("TFT", '#0000FF', '-', None),
    ("WSLS", '#00FFFF', '-', None),
    ("Holds a grudge", '#FF0000', '-', None),
    ("Fool me once", '#00FF80', '-', None),
    ("OmegaTFT", '#FFFF00', '-', None),
    ("GradualTFT", '#FF00FF', '-', None),
    ("ZDExtort2", color_zd[0], linestyle_zd[0%3], None),
    ("ZDExtort2v2", color_zd[1], linestyle_zd[1%3], None),
    ("ZDExtort3", color_zd[2], linestyle_zd[2%3], None),
    ("ZDExtort4", color_zd[3], linestyle_zd[3%3], None),
    ("ZDGen2", color_zd[4], linestyle_zd[4%3], None),
    ("ZDGTFT2", color_zd[5], linestyle_zd[5%3], None),
    ("ZDMischief", color_zd[6], linestyle_zd[6%3], None),
    ("ZDSet2", color_zd[7], linestyle_zd[7%3], None)
]

legend_legend1 = ax_legend.legend(
    handles=[
        plt.Line2D([0], [0], color=color, linestyle=linestyle, linewidth=1.5, label=label, marker='None')
        for label, color, linestyle, _ in legend_labels[:7]
    ],
    loc='center left',
    fontsize=10,
    frameon=False,
    bbox_to_anchor=(-0.15, 0.5)
)
legend_legend2 = ax_legend.legend(
    handles=[
        plt.Line2D([0], [0], color=color, linestyle=linestyle, linewidth=1.5, label=label, marker='None')
        for label, color, linestyle, _ in legend_labels[7:]
    ],
    loc='center right',
    fontsize=10,
    frameon=False,
    bbox_to_anchor=(1.12, 0.5)
)
ax_legend.add_artist(legend_legend1)
ax_legend.add_artist(legend_legend2)


label_ax_a = fig.add_axes([axs[0, 0].get_position().x0, axs[0, 0].get_position().y1, 0.01, 0.01], zorder=10)
label_ax_b = fig.add_axes([axs[0, 1].get_position().x0, axs[0, 1].get_position().y1, 0.01, 0.01], zorder=10)
label_ax_c = fig.add_axes([axs[1, 0].get_position().x0, axs[1, 0].get_position().y1, 0.01, 0.01], zorder=10)
label_ax_d = fig.add_axes([axs[1, 1].get_position().x0, axs[1, 1].get_position().y1, 0.01, 0.01], zorder=10)

label_ax_a.text(-8.8, 7.5, 'a', fontsize=11, fontweight='bold', ha='center', va='center')
label_ax_b.text(-2.5, 7.5, 'b', fontsize=11, fontweight='bold', ha='center', va='center')
label_ax_c.text(-8.8, 0.5, 'c', fontsize=11, fontweight='bold', ha='center', va='center')
# label_ax_d.text(-2.5, 0.5, 'd', fontsize=12, fontweight='bold', ha='center', va='center')

label_ax_a.axis('off')
label_ax_b.axis('off')
label_ax_c.axis('off')
label_ax_d.axis('off')

# plt.savefig('test1.pdf', format='pdf', dpi=300)
plt.tight_layout()
plt.show()