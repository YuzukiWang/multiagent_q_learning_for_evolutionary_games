import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import matplotlib.patches as patches
import numpy as np

from matplotlib import rc
rc('font', family='Arial')
plt.rc('font', size=11)
plt.rcParams['axes.linewidth'] = 0.5


subplot_size = (3, 3)
fig, axs = plt.subplots(2, 2, figsize=(subplot_size[0] * 2, subplot_size[1] * 2), gridspec_kw={'width_ratios': [subplot_size[0]] * 2, 'height_ratios': [subplot_size[1]] * 2})
fig.subplots_adjust(wspace=0.18, hspace=0.24)


ax_schematic = axs[0, 0]
ax_schematic.axis('off')

main_axes_schematic = inset_axes(ax_schematic, width="100%", height="100%", loc='center')
main_axes_schematic.set_aspect('equal', adjustable='box')
for spine in main_axes_schematic.spines.values():
    spine.set_linewidth(0.5)

rect_hg = patches.Rectangle((-2, 0), 2, 2, linewidth=0.5, edgecolor='none', facecolor='#add8e6')  
rect_sd = patches.Rectangle((0, 0), 2, 2, linewidth=0.5, edgecolor='none', facecolor='#e0e5df')  
rect_sh = patches.Rectangle((-2, -2), 2, 2, linewidth=0.5, edgecolor='none', facecolor='#bfee90')  
rect_pd = patches.Rectangle((0, -2), 2, 2, linewidth=0.5, edgecolor='none', facecolor='#ffb6c1')  
main_axes_schematic.add_patch(rect_hg)
main_axes_schematic.add_patch(rect_sd)
main_axes_schematic.add_patch(rect_sh)
main_axes_schematic.add_patch(rect_pd)
main_axes_schematic.text(-1, 1, 'HG', fontsize=11, ha='center', va='center')
main_axes_schematic.text(1, 1, 'SD', fontsize=11, ha='center', va='center')
main_axes_schematic.text(-1, -1, 'SH', fontsize=11, ha='center', va='center')
main_axes_schematic.text(1, -1, 'PD', fontsize=11, ha='center', va='center')
plt.xticks([-2, -1, 0, 1, 2], fontsize=11)
plt.yticks([-2, -1, 0, 1, 2], fontsize=11)
plt.xlabel(r'$\mathit{T}-\mathit{R}$', fontsize=11, labelpad=1.5)
plt.ylabel(r'$\mathit{S}-\mathit{P}$', fontsize=11, labelpad=-5)
plt.tick_params(axis='both', direction='in', width=0.5)



for i in range(2):
    for j in range(2):
        if i == 0 and j == 0:
            continue
        elif i == 0 and j == 1:
            data_filename = "WM.npy"
            #text_label = 'Well-mixed population'
            text_label = 'Well-mixed'
            final_result = np.load(data_filename)
        elif i == 1 and j == 0:
            data_filename = "lattice.npy"
            text_label = 'Lattice'
            final_result = np.load(data_filename)
            final_result *= 0.01
        elif i == 1 and j == 1:
            data_filename = "SF.npy"
            text_label = 'Scale-free'
            final_result = np.load(data_filename)

        ax = axs[i, j]
        ax.axis('off')
        main_axes = inset_axes(ax, width="100%", height="100%", loc='center')
        main_axes.set_aspect('equal', adjustable='box')
        for spine in main_axes.spines.values():
            spine.set_linewidth(0.5)
        img = main_axes.imshow(final_result, cmap='YlGnBu_r', interpolation='bicubic', vmin=0, vmax=1)

        yticks_percent = [0, 0.25, 0.5, 0.75, 1]
        yticklabels = ['-1', '0', '1', '2', '3']
        main_axes.set_yticks([main_axes.get_ylim()[0] + p * (main_axes.get_ylim()[1] - main_axes.get_ylim()[0]) for p in yticks_percent])
        main_axes.set_yticklabels(yticklabels, fontsize=11)

        xticks_percent = [0, 0.25, 0.5, 0.75, 1]
        xticklabels = ['1', '2', '3', '4', '5']
        main_axes.set_xticks([main_axes.get_xlim()[0] + p * (main_axes.get_xlim()[1] - main_axes.get_xlim()[0]) for p in xticks_percent])
        main_axes.set_xticklabels(xticklabels, fontsize=11)

        main_axes.set_xlabel('T', fontsize=11, labelpad=1.5, fontstyle='italic')
        main_axes.set_ylabel('S', fontsize=11, labelpad=-1.5, fontstyle='italic')
        main_axes.tick_params(axis='both', direction='in', which='both', bottom=True, top=False, left=True, right=False, width=0.5)


        #mid_y = final_result.shape[0] // 2
        #mid_x = final_result.shape[1] // 2
        main_axes.axhline(y=9.5, color='gray', linestyle='--', linewidth=0.5)
        main_axes.axvline(x=9.5, color='gray', linestyle='--', linewidth=0.5)

        fig.text(axs[i, j].get_position().x0 + axs[i, j].get_position().width / 2, axs[i, j].get_position().y0 + axs[i, j].get_position().height * 0.1, text_label, fontsize=11, ha='center', va='bottom')


cax_height = axs[0, 1].get_position().y1 - axs[1, 1].get_position().y0
bias5 = (axs[0, 1].get_position().y1 - axs[0, 1].get_position().y0) * 0
cax = fig.add_axes([0.93, axs[1, 1].get_position().y0 + bias5, 0.02, cax_height - 2 * bias5])  # [left, bottom, width, height]
cbar = fig.colorbar(img, cax=cax, shrink=0.72, ticks=[0, 1])
cbar.ax.tick_params(axis='y', length=0, width=0.5)
cbar.set_label('Fraction of MTBR agents', rotation=90, fontsize=11, labelpad=-3)


label_ax_a = fig.add_axes([axs[0, 0].get_position().x0, axs[0, 0].get_position().y1, 0.01, 0.01], zorder=10)
label_ax_b = fig.add_axes([axs[0, 1].get_position().x0, axs[0, 1].get_position().y1, 0.01, 0.01], zorder=10)
label_ax_c = fig.add_axes([axs[1, 0].get_position().x0, axs[1, 0].get_position().y1, 0.01, 0.01], zorder=10)
label_ax_d = fig.add_axes([axs[1, 1].get_position().x0, axs[1, 1].get_position().y1, 0.01, 0.01], zorder=10)

label_ax_a.text(-3.5, 2, 'a', fontsize=11, fontweight='bold', ha='center', va='center')
label_ax_b.text(-3.5, 2, 'b', fontsize=11, fontweight='bold', ha='center', va='center')
label_ax_c.text(-3.5, 2, 'c', fontsize=11, fontweight='bold', ha='center', va='center')
label_ax_d.text(-3.5, 2, 'd', fontsize=11, fontweight='bold', ha='center', va='center')

label_ax_a.axis('off')
label_ax_b.axis('off')
label_ax_c.axis('off')
label_ax_d.axis('off')

# plt.savefig('Figure5.svg')
plt.show()