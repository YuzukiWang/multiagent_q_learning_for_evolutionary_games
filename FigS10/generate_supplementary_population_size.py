import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import matplotlib.patches as patches
import numpy as np

from matplotlib import rc
rc('font', family='Arial')
plt.rc('font', size=12)
plt.rcParams['axes.linewidth'] = 0.5

s_list = [0.25,0.5,0.75,1,1.25,1.5,1.75,2,2.25,2.5]
N_list = [15,30,45,60,75,90,120,150,180,225,270,315,360,450,540]


data_filename = "selection_intensity.npy"
final_result = np.load(data_filename)
list1 = [row[0] * 224 for row in final_result] 
list2 = [row[1] for row in final_result]


data_filename = "population_size.npy"
final_result = np.load(data_filename)
list3 = [row[0] * 0.5 for row in final_result] 
list3 = [x * y for x, y in zip(list3, N_list)] 
list4 = [row[1] for row in final_result]




subplot_size = (3, 3)
# fig, axs = plt.subplots(1, 2, figsize=(subplot_size[0] * 2, subplot_size[1] * 1), gridspec_kw={'width_ratios': [subplot_size[0]] * 2, 'height_ratios': [subplot_size[1]] * 1})
fig, axs = plt.subplots(1, 2, figsize=(6, 3))
fig.subplots_adjust(wspace=0.88,hspace=0.3)



axs[0].plot(s_list, list1, label='Absorption time', color='b',marker='o')
axs[0].set_ylabel('Absorption time')
axs[0].set_ylim(-600, 30600)

axs[0].set_box_aspect(1)
axs[0].set_xlim(0.15 ,2.6)
axs[0].set_yticks([0,10000, 20000,30000])

axs[0].set_yticklabels(['0', '1', '2', '3'])
axs[0].text(0.05, 1.05, r'$\times 10^4$', transform=axs[0].transAxes, ha='center', va='bottom')



ax2 = axs[0].twinx()
ax2.plot(s_list, list2, label='BRTM occupancy', color='r', marker='x')
ax2.set_ylabel('BRTM occupancy')
ax2.set_ylim(0.9082, 1.0018)
ax2.set_box_aspect(1)
ax2.set_yticks([0.91,0.94, 0.97,1.00])
ax2.set_yticklabels(['0.91',  '0.94', '0.97', '1.00'])
axs[0].tick_params(axis='both', direction='in', width=0.5)
ax2.tick_params(axis='both', direction='in', width=0.5)

axs[0].legend(loc='best',fontsize=10,frameon=False)
ax2.legend(loc='best',fontsize=10,frameon=False)

axs[0].set_xticks([0.5, 1.0, 1.5, 2.0, 2.5])
axs[0].set_xticklabels(['0.5', '1.0', '1.5', '2.0', '2.5'])

axs[0].set_xlabel('Selection intensity, $\mathit{δ}$')







axs[1].plot(N_list, list3, label='Absorption time', color='b',marker='o')
axs[1].set_xscale('log')
axs[1].set_ylabel('Absorption time')
axs[1].set_ylim(-300, 15300)
axs[1].set_yticks([0,5000, 10000,15000])
axs[1].set_yticklabels(['0', '0.5', '1', '1.5'])
axs[1].set_box_aspect(1)
axs[1].tick_params(axis='both', direction='in', width=0.5)



from matplotlib.ticker import FixedLocator, NullFormatter
axs[1].xaxis.set_minor_locator(FixedLocator([15, 50, 150, 500]))
axs[1].xaxis.set_minor_formatter(NullFormatter())

ax3 = axs[1].twinx()
ax3.plot(N_list, list4, label='BRTM occupancy', color='r', marker='x')
ax3.set_yticks([0.55,0.70, 0.85,1.00])
ax3.set_yticklabels(['0.55',  '0.70', '0.85', '1.00'])
# ax3.set_xscale('log')
ax3.set_ylabel('BRTM occupancy')
ax3.set_ylim(0.55-0.009, 1.009)
ax3.set_box_aspect(1)

ax3.tick_params(axis='both', direction='in', width=0.5)
axs[1].text(0.05, 1.05, r'$\times 10^4$', transform=axs[1].transAxes, ha='center', va='bottom')

axs[1].set_xticks([15, 50, 150, 500])
axs[1].set_xticklabels(['15', '50', '150', '500'])

label_ax_a = fig.add_axes([axs[0].get_position().x0, axs[0].get_position().y1, 0.01, 0.01], zorder=10)
label_ax_b = fig.add_axes([axs[1].get_position().x0, axs[0].get_position().y1, 0.01, 0.01], zorder=10)

label_ax_a.text(-5, 7.5, 'a', fontsize=12, fontweight='bold', ha='center', va='center')
label_ax_b.text(-7.1, 7.5, 'b', fontsize=12, fontweight='bold', ha='center', va='center')

label_ax_a.axis('off')
label_ax_b.axis('off')


axs[1].set_xlabel('Population size, $\mathit{N}$')





# plt.xticks([ 0, 1, 2, 3], fontsize=12)
# plt.yticks([-2, -1, 0, 1, 2], fontsize=12)





plt.show()
