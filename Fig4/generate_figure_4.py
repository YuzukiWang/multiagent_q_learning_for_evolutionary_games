from matplotlib import pyplot as plt
import numpy as np

def panel_a(selection_intensity):
    plt.rcParams['font.family'] = 'Arial'
    plt.rc('font', size=17)

    states_over_time = np.load(f'figure4_data/states_over_time_withoutMTBR_{selection_intensity}_7500.npy')

    x = [value * 10 for value in range(1001)]
    fig, ax = plt.subplots(figsize=(5, 5))

    for index in range(1,101):
        temp_matrix = np.loadtxt(f'figure4_data/WithoutMTBR_delta{selection_intensity}_{index}.txt')
        
        ax.plot(x, temp_matrix[:,5], color='red', linewidth = 0.5, alpha = 0.08)
        ax.plot(x, temp_matrix[:,11], color='blue', linewidth = 0.5, alpha = 0.08)

    x_time = [value * 0.01 for value in range(1000000)]

    ax.plot(x_time, states_over_time[:,5], color='red', linewidth = 2.5, label = 'General TFT')
    ax.plot(x_time, states_over_time[:,11], color='blue', linewidth = 2.5, label = 'ZDTFT2')
    ax.plot(x_time, states_over_time[:,0]+states_over_time[:,1]+states_over_time[:,2]+states_over_time[:,3]+states_over_time[:,4]+states_over_time[:,6]+states_over_time[:,7]+states_over_time[:,8]+states_over_time[:,9]+states_over_time[:,10]
            +states_over_time[:,12]+states_over_time[:,13], color='grey', linewidth = 1.5, label = 'Other strategies')

    ax.ticklabel_format(style='sci', axis='x', scilimits=(0, 0))

    ax.set_ylim(-0.02,1.0)
    ax.set_xlim(0,4000)
    ax.set_title('Without MTBR', fontsize=17.5)
    ax.set_xlabel(r'Time', fontsize=17.5)
    ax.set_ylabel(r'Fraction of strategies', fontsize=17.5)
    plt.legend()

    #plt.savefig('panel_a.svg')


    plt.show()

def panel_b(selection_intensity):
    plt.rcParams['font.family'] = 'Arial'
    plt.rc('font', size=17)

    states_over_time = np.load(f'figure4_data/states_over_time_withMTBR_{selection_intensity}_7500.npy')

    x = [value * 10 for value in range(1001)]
    fig, ax = plt.subplots(figsize=(5, 5))

    for index in range(1,101):
        temp_matrix = np.loadtxt(f'figure4_data/WithMTBR_delta{selection_intensity}_{index}.txt')
        if temp_matrix[1000,6] >0.5 or temp_matrix[150,6] >0.5:
            continue

        ax.plot(x, temp_matrix[:,0], color='green', linewidth = 0.5, alpha = 0.08)
        ax.plot(x, temp_matrix[:,6], color='red', linewidth = 0.5, alpha = 0.08)
        ax.plot(x, temp_matrix[:,12], color='blue', linewidth = 0.5, alpha = 0.08)


    x_time = [value * 0.01 for value in range(1000000)]

    ax.plot(x_time, states_over_time[:,0], color='green', linewidth = 2.5, label = 'MTBR')
    ax.plot(x_time, states_over_time[:,6], color='red', linewidth = 2.5, label = 'General TFT')
    ax.plot(x_time, states_over_time[:,12], color='blue', linewidth = 2.5, label = 'ZDTFT2')
    ax.plot(x_time, states_over_time[:,1]+states_over_time[:,2]+states_over_time[:,3]+states_over_time[:,4]+states_over_time[:,5]+states_over_time[:,7]+states_over_time[:,8]+states_over_time[:,9]+states_over_time[:,10]
            +states_over_time[:,11]+states_over_time[:,13]+states_over_time[:,14], color='grey', linewidth = 1.5, label = 'Other strategies')

    ax.ticklabel_format(style='sci', axis='x', scilimits=(0, 0))

    ax.set_ylim(-0.02,1.02)
    ax.set_xlim(0,4000)
    ax.set_title('With MTBR', fontsize=17.5)
    ax.set_xlabel(r'Time', fontsize=17.5)
    ax.set_ylabel(r'Fraction of strategies', fontsize=17.5)
    plt.legend()

    #plt.savefig('panel_b.svg')

    plt.show()

def panel_c(selection_intensity):
    plt.rcParams['font.family'] = 'Arial'
    plt.rc('font', size=17)
    fig, ax = plt.subplots(figsize=(5, 5))

    payoff_withMTBR_numerical = np.loadtxt(f'figure4_data/payoff_withMTBR_numerical_{selection_intensity}.txt')
    payoff_withoutMTBR_numerical = np.loadtxt(f'figure4_data/payoff_withoutMTBR_numerical_{selection_intensity}.txt')

    x_simulation = [value * 10 for value in range(1001)]

    PayoffMatrix_withMTBR = np.loadtxt('figure4_data/payoffMatrix.txt')
    PayoffMatrix_withoutMTBR = PayoffMatrix_withMTBR[1:,1:]
    for index in range(1,101):
        state_withMTBR_simulation = np.loadtxt(f'figure4_data/WithMTBR_delta{selection_intensity}_{index}.txt')
        state_withoutMTBR_simulation = np.loadtxt(f'figure4_data/WithoutMTBR_delta{selection_intensity}_{index}.txt')

        payoff_withMTBR_simulation = np.array([state_withMTBR_simulation[i, :] @ PayoffMatrix_withMTBR @ state_withMTBR_simulation[i, :].T for i in range(state_withMTBR_simulation.shape[0])])
        payoff_withoutMTBR_simulation = np.array([state_withoutMTBR_simulation[i, :] @ PayoffMatrix_withoutMTBR @ state_withoutMTBR_simulation[i, :].T for i in range(state_withoutMTBR_simulation.shape[0])])

        ax.plot(x_simulation, payoff_withMTBR_simulation, color='red', linewidth=0.5, alpha = 0.08)
        ax.plot(x_simulation, payoff_withoutMTBR_simulation, color='blue', linewidth=0.5, alpha = 0.08)

    x_numerical = [value * 0.01 for value in range(1000000)]
    ax.plot(x_numerical, payoff_withMTBR_numerical, color='red', linewidth=2.5, label='With MTBR')
    ax.plot(x_numerical, payoff_withoutMTBR_numerical, color='blue', linewidth=2.5, label='Without MTBR')

    # 设置横轴为科学计数法
    ax.ticklabel_format(style='sci', axis='x', scilimits=(0, 0))

    ax.set_ylim(2.85, 2.95)
    ax.set_xlim(0, 4000)
    ax.set_yscale('log')

    ax.yaxis.set_ticks([])

    custom_ticks = [2.86, 2.88, 2.90, 2.92, 2.94]
    ax.set_yticks(custom_ticks)
    ax.set_yticklabels([f'{tick:.2f}' for tick in custom_ticks])

    ax.set_xlabel(r'Time', fontsize=17.5)
    ax.set_ylabel(r'Global average payoff', fontsize=17.5)
    plt.legend()

    #plt.savefig('panel_c.svg')
    plt.show()

def panel_d():
    plt.rcParams['font.family'] = 'Arial'
    plt.rc('font', size=17)

    state_vector_final = []
    for deltaindex in range(1,22):
        vector_deltaindex = np.zeros((1,14))
        for index in range(1,51):
            temp_matrix = np.loadtxt(f'figure4_data/selection_intensity/WithoutMTBR_deltaindex{deltaindex}_{index}.txt')
            vector_deltaindex += temp_matrix[-1,:]
        vector_deltaindex /= 50
        state_vector_final.append(vector_deltaindex)

    state_vector_final = np.vstack(state_vector_final)

    fig, ax = plt.subplots(figsize=(5, 5))

    delta_set = np.array([10**(-1), 10**(-0.9), 10**(-0.8), 10**(-0.7), 10**(-0.6), 10**(-0.5), 10**(-0.4), 10**(-0.3), 10**(-0.2), 10**(-0.1), 1,
                        10**(0.1), 10**(0.2), 10**(0.3), 10**(0.4), 10**(0.5), 10**(0.6), 10**(0.7), 10**(0.8), 10**(0.9), 10])


    ax.scatter(
        delta_set[::2],
        state_vector_final[:,5][::2],
        facecolor='none',
        marker='s',
        s=50,
        edgecolor='red',
        linewidths=1.1,
        label='Gradual TFT'
    )

    ax.scatter(
        delta_set[::2],
        state_vector_final[:,11][::2],
        facecolor='none',
        marker='s',
        s=50,
        edgecolor='blue',
        linewidths=1.1,
        label='ZDGTFT2'
    )

    states_over_time = np.loadtxt('figure4_data/WithoutMTBR_selectionIntensity_numerical.txt')


    ax.plot(delta_set, states_over_time[:,5], color='red', linewidth = 1.5, label='Theoretical')
    ax.plot(delta_set, states_over_time[:,11], color='blue', linewidth = 1.5, label='Simulation')

    ax.ticklabel_format(style='sci', axis='x', scilimits=(0, 0))
    ax.set_xscale('log')
    ax.set_ylim(-0.04,1.04)
    ax.set_title('Without MTBR', fontsize=17.5)
    ax.set_xlabel(r'Selection intensity', fontsize=17.5)
    ax.set_ylabel(r'Fraction of strategies', fontsize=17.5)
    plt.legend()
    #plt.savefig('panel_d.svg')
    plt.show()

def panel_e():
    plt.rcParams['font.family'] = 'Arial'
    plt.rc('font', size=17)

    state_vector_final = []
    for deltaindex in range(1,22):
        vector_deltaindex = np.zeros((1,15))
        for index in range(1,51):
            temp_matrix = np.loadtxt(f'figure4_data/selection_intensity/WithMTBR_deltaindex{deltaindex}_{index}.txt')
            vector_deltaindex += temp_matrix[-1,:]
        vector_deltaindex /= 50
        state_vector_final.append(vector_deltaindex)

    state_vector_final = np.vstack(state_vector_final)

    fig, ax = plt.subplots(figsize=(5, 5))

    delta_set = np.array([10**(-1), 10**(-0.9), 10**(-0.8), 10**(-0.7), 10**(-0.6), 10**(-0.5), 10**(-0.4), 10**(-0.3), 10**(-0.2), 10**(-0.1), 1,
                        10**(0.1), 10**(0.2), 10**(0.3), 10**(0.4), 10**(0.5), 10**(0.6), 10**(0.7), 10**(0.8), 10**(0.9), 10])


    ax.scatter(
        delta_set[::2],
        state_vector_final[:,0][::2],
        facecolor='none',
        marker='s',
        s=50,
        edgecolor='green',
        linewidths=1.1,
        label='MTBR'
    )

    ax.scatter(
        delta_set[::2],
        state_vector_final[:,6][::2],
        facecolor='none',
        marker='s',
        s=50,
        edgecolor='red',
        linewidths=1.1,
        label='Gradual TFT'
    )

    ax.scatter(
        delta_set[::2],
        state_vector_final[:,12][::2],
        facecolor='none',
        marker='s',
        s=50,
        edgecolor='blue',
        linewidths=1.1,
        label='ZDGTFT2'
    )


    states_over_time = np.loadtxt('figure4_data/WithMTBR_selectionIntensity_numerical.txt')

    ax.plot(delta_set, states_over_time[:,0], color='green', linewidth = 1.5, label='Theoretical')
    ax.plot(delta_set, states_over_time[:,6], color='red', linewidth = 1.5, label='Simulation')
    ax.plot(delta_set, states_over_time[:,12], color='blue', linewidth = 1.5, alpha = 0.5, label='Simulation')

    ax.ticklabel_format(style='sci', axis='x', scilimits=(0, 0))
    ax.set_xscale('log')
    ax.set_ylim(-0.04,1.04)
    ax.set_title('With MTBR', fontsize=17.5)
    ax.set_xlabel(r'Selection intensity', fontsize=17.5)
    ax.set_ylabel(r'Fraction of strategies', fontsize=17.5)
    plt.legend()
    #plt.savefig('panel_e.svg')
    plt.show()

def panel_f():
    plt.rcParams['font.family'] = 'Arial'
    plt.rc('font', size=17)

    PayoffMatrix_withMTBR = np.loadtxt('figure4_data/payoffMatrix.txt')
    PayoffMatrix_withoutMTBR = PayoffMatrix_withMTBR[1:,1:]

    state_vector_final_withoutMTBR = []
    for deltaindex in range(1,22):
        vector_deltaindex = np.zeros((1,14))
        for index in range(1,51):
            temp_matrix = np.loadtxt(f'figure4_data/selection_intensity/WithoutMTBR_deltaindex{deltaindex}_{index}.txt')
            vector_deltaindex += temp_matrix[-1,:]
        vector_deltaindex /= 50
        state_vector_final_withoutMTBR.append(vector_deltaindex)

    state_vector_final_withoutMTBR = np.vstack(state_vector_final_withoutMTBR)

    state_vector_final_withMTBR = []
    for deltaindex in range(1,22):
        vector_deltaindex = np.zeros((1,15))
        for index in range(1,51):
            temp_matrix = np.loadtxt(f'figure4_data/selection_intensity/withMTBR_deltaindex{deltaindex}_{index}.txt')
            vector_deltaindex += temp_matrix[-1,:]
        vector_deltaindex /= 50
        state_vector_final_withMTBR.append(vector_deltaindex)

    state_vector_final_withMTBR = np.vstack(state_vector_final_withMTBR)

    payoff_withMTBR_nu = np.zeros(21)
    payoff_withoutMTBR_nu = np.zeros(21)

    for index in range(21):
        payoff_withMTBR_nu[index] = state_vector_final_withMTBR[index,:] @ PayoffMatrix_withMTBR @ state_vector_final_withMTBR[index,:].T
        payoff_withoutMTBR_nu[index] = state_vector_final_withoutMTBR[index,:] @ PayoffMatrix_withoutMTBR @ state_vector_final_withoutMTBR[index,:].T


    fig, ax = plt.subplots(figsize=(5, 5))

    delta_set = np.array([10**(-1), 10**(-0.9), 10**(-0.8), 10**(-0.7), 10**(-0.6), 10**(-0.5), 10**(-0.4), 10**(-0.3), 10**(-0.2), 10**(-0.1), 1,
                        10**(0.1), 10**(0.2), 10**(0.3), 10**(0.4), 10**(0.5), 10**(0.6), 10**(0.7), 10**(0.8), 10**(0.9), 10])

    ax.scatter(
        delta_set[::2],
        payoff_withoutMTBR_nu[::2],
        facecolor='none',
        marker='s',
        s=50,
        edgecolor='blue',
        linewidths=1.1,
        label='Without MTBR'
    )

    ax.scatter(
        delta_set[::2],
        payoff_withMTBR_nu[::2],
        facecolor='none',
        marker='s',
        s=50,
        edgecolor='red',
        linewidths=1.1,
        label='With MTBR'
    )

    states_over_time_withoutMTBR = np.loadtxt('figure4_data/WithoutMTBR_selectionIntensity_numerical.txt')
    states_over_time_withMTBR = np.loadtxt('figure4_data/WithMTBR_selectionIntensity_numerical.txt')

    payoff_withMTBR_si = np.zeros(21)
    payoff_withoutMTBR_si = np.zeros(21)

    for index in range(21):
        payoff_withMTBR_si[index] = states_over_time_withMTBR[index,:] @ PayoffMatrix_withMTBR @ states_over_time_withMTBR[index,:].T
        payoff_withoutMTBR_si[index] = states_over_time_withoutMTBR[index,:] @ PayoffMatrix_withoutMTBR @ states_over_time_withoutMTBR[index,:].T

    ax.plot(delta_set, payoff_withMTBR_si, color='red', linewidth = 1.5, label='Theoretical')
    ax.plot(delta_set, payoff_withoutMTBR_si, color='blue', linewidth = 1.5, label='Simulation')


    ax.set_xscale('log')
    ax.set_ylim(2.89,2.94)
    ax.set_xlabel(r'Selection intensity', fontsize=17.5)
    ax.set_ylabel(r'Global average payoff', fontsize=17.5)
    plt.legend()

    #plt.savefig('panel_f.svg')

    plt.show()

#panel_a(0.5)
#panel_b(0.5)
#panel_c(0.5)
#panel_d()
#panel_e()
#panel_f()
