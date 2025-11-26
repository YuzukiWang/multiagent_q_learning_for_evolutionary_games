import numpy as np

def get_Ab(PayoffMatrix, num_of_strategies, num_of_players):
    # 初始化 A 矩阵
    A = np.zeros((num_of_strategies, num_of_strategies))

    # 填充 A 矩阵
    for i in range(1, num_of_strategies):  # 行遍历，从第2行开始
        for j in range(num_of_strategies):  # 列遍历
            A[i-1, j] = PayoffMatrix[0, j] - PayoffMatrix[i, j]
    
    # 最后一行全为 1
    A[-1, :] = 1

    # 初始化 b 向量
    b = np.zeros(num_of_strategies)
    
    # 填充 b 向量
    for i in range(1, num_of_strategies):
        b[i-1] = (PayoffMatrix[0, 0] - PayoffMatrix[i, i]) / num_of_players
    
    # 最后一项为 1
    b[-1] = 1

    return A, b

def payoff_processing(mode):
    filename = 'payoffMatrix_new.txt'
    PayoffMatrix_original = np.loadtxt(filename)
    #filename = 'matrix_data.csv'
    #PayoffMatrix_original = np.genfromtxt(filename, delimiter=',')

    if mode == 1: #with MBTR
        #num_of_strategies = 15
        #num_of_players = 500*15
        num_of_strategies = 16
        num_of_players = 500*16
        PayoffMatrix = PayoffMatrix_original

    elif mode ==2: #without MBTR
        #num_of_strategies = 14
        #num_of_players = 500*14
        num_of_strategies = 15
        num_of_players = 500*15
        PayoffMatrix = PayoffMatrix_original[1:, 1:]

    return PayoffMatrix, num_of_strategies, num_of_players

def interior_eq(A,b):
    if np.linalg.det(A) != 0:
        return np.linalg.solve(A, b)
    else:
        print("No interior equilibrim point")
        return None
    
def Jacobian_matrix(PayoffMatrix, num_of_strategies, num_of_players, state_vector):
    
    diagonal_vector = np.diag(PayoffMatrix)
    diagonal_matrix = np.diag(diagonal_vector)

    U_vector = num_of_players / (num_of_players-1) * ( PayoffMatrix @ state_vector - 1/num_of_players * (diagonal_matrix @ state_vector) )
    U_average = U_vector.T @ state_vector
    U_minus_average = U_vector - U_average
    U_matrix = np.diag(U_minus_average)

    a_mm_matrix = np.tile(diagonal_vector, (num_of_strategies, 1)) / num_of_players

    akm_plus_amk = PayoffMatrix + PayoffMatrix.T

    temp1 = PayoffMatrix + a_mm_matrix - np.tile((akm_plus_amk @ state_vector).T, (num_of_strategies,1))
    temp2 = temp1 * num_of_players / (num_of_players-1)

    temp3 = temp2 * np.tile(state_vector, (1, num_of_strategies))

    J = temp3 + U_matrix

    return J

def numerical(PayoffMatrix, num_of_strategies, num_of_players, selection_intensity, time_step, time):
    state_vector = np.full(num_of_strategies, 1/num_of_strategies)
    diagonal_vector = np.diag(PayoffMatrix)
    diagonal_matrix = np.diag(diagonal_vector)

    for _ in range(int(time/time_step)):
        U_vector = num_of_players / (num_of_players-1) * ( PayoffMatrix @ state_vector - 1/num_of_players * (diagonal_matrix @ state_vector) )
        U_average = U_vector.T @ state_vector
        U_minus_average = U_vector - U_average
        state_vector = state_vector + time_step * num_of_players/(num_of_players-1) * selection_intensity/2 * (state_vector * U_minus_average )

    return state_vector

def numerical_new(PayoffMatrix, num_of_strategies, num_of_players, selection_intensity, time_step, time, deltaindex):
    state_vector = np.full(num_of_strategies, 1/num_of_strategies)
    diagonal_vector = np.diag(PayoffMatrix)
    diagonal_matrix = np.diag(diagonal_vector)
    
    # 初始化列表来存储每一个时间步的状态向量
    states_over_time = []
    
    for _ in range(int(time/time_step)):
        U_vector = num_of_players / (num_of_players-1) * (PayoffMatrix @ state_vector - 1/num_of_players * (diagonal_matrix @ state_vector))
        U_average = U_vector.T @ state_vector
        U_minus_average = U_vector - U_average
        state_vector = state_vector + time_step * num_of_players/(num_of_players-1) * selection_intensity/2 * (state_vector * U_minus_average)
        
        # 在每个时间步后保存当前的状态向量
        states_over_time.append(state_vector.copy())
    
    # 将所有时间步的状态向量保存到npy文件
    np.save(f'states_over_time_withoutMTBR_{deltaindex}_withGTFT3.npy', np.array(states_over_time))

    return state_vector

def numerical_new_with_noise(PayoffMatrix, num_of_strategies, num_of_players, selection_intensity, time_step, time):
    state_vector = np.full(num_of_strategies, 1/num_of_strategies)
    diagonal_vector = np.diag(PayoffMatrix)
    diagonal_matrix = np.diag(diagonal_vector)
    
    # 初始化列表来存储每一个时间步的状态向量
    states_over_time = []
    
    for _ in range(int(time/time_step)):
        U_vector = num_of_players / (num_of_players-1) * (PayoffMatrix @ state_vector - 1/num_of_players * (diagonal_matrix @ state_vector))
        U_average = U_vector.T @ state_vector
        U_minus_average = U_vector - U_average
        second_term_temp = state_vector * (1.-state_vector) /(num_of_players-1)
        second_term = np.sqrt(second_term_temp)
        state_vector = state_vector + time_step * (num_of_players/(num_of_players-1) * selection_intensity/2 * (state_vector * U_minus_average) + second_term_temp * np.random.normal(loc=0, scale=1))
        
        # 在每个时间步后保存当前的状态向量
        states_over_time.append(state_vector.copy())
    
    # 将所有时间步的状态向量保存到npy文件
    np.save('states_over_time_with_noise_withMTBR.npy', np.array(states_over_time))

    return state_vector

'''
def main_old(mode):
    PayoffMatrix, num_of_strategies, num_of_players = payoff_processing(mode)
    A,b = get_Ab(PayoffMatrix, num_of_strategies, num_of_players)
    x_interior = interior_eq(A,b)
    #print("A:",A)
    #print("B:",b)
    print("interior equilibrium point:", x_interior)
    print("U:",num_of_players / (num_of_players-1) * ( PayoffMatrix @ x_interior - 1/num_of_players * (np.diag(np.diag(PayoffMatrix)) @ x_interior) ))



main_old(mode)



'''

def main(mode):
    PayoffMatrix, num_of_strategies, num_of_players = payoff_processing(mode)
    num_of_stable_equilibria = 0
    for num_removed in range(num_of_strategies):  # 从删除0个开始一直到删除 num_of_strategies 个
        print(f"{num_removed} extinct strategies:")

        # 生成所有可以移除的组合
        combinations = generate_combinations(num_of_strategies, num_removed)

        for comb in combinations:
            remaining_indices = [i for i in range(num_of_strategies) if i not in comb]
            A, b = get_dynamic_Ab(PayoffMatrix, remaining_indices, num_of_players)
            try:
                x_solution = np.linalg.solve(A, b)
                x_solution = expand_solution(x_solution, comb, num_of_strategies)
                if np.all(x_solution>=0) & np.all(Jacobian_matrix(PayoffMatrix, num_of_strategies, num_of_players, x_solution)<=0):
                    print(f"Solution with x{comb}=0: {x_solution}")
                    num_of_stable_equilibria += 1
            except np.linalg.LinAlgError:
                #print(f"Singular matrix for indices {comb}, skipping.")
                break
    return num_of_stable_equilibria

def expand_solution(sub_x, comb, original_size):
    # 初始化为全零的列向量 (original_size, 1)
    full_x = np.zeros((original_size, 1))
    # 将 sub_x 填入到未移除的索引位置
    remaining_indices = [i for i in range(original_size) if i not in comb]
    full_x[remaining_indices, 0] = sub_x  # 将 sub_x 的值赋给相应位置
    
    return full_x

def generate_combinations(n, k):
    from itertools import combinations
    return list(combinations(range(n), k))

def get_dynamic_Ab(PayoffMatrix, remaining_indices, num_of_players):
    num_of_strategies = len(remaining_indices)
    
    # 初始化 A 矩阵
    A = np.zeros((num_of_strategies, num_of_strategies))

    # 填充 A 矩阵，动态使用剩余索引
    for i, row_index in enumerate(remaining_indices[1:]):  # 行遍历，从第2行开始
        for j, col_index in enumerate(remaining_indices):   # 列遍历
            A[i, j] = PayoffMatrix[remaining_indices[0], col_index] - PayoffMatrix[row_index, col_index]
    
    # 最后一行全为 1
    A[-1, :] = 1

    # 初始化 b 向量
    b = np.zeros(num_of_strategies)
    
    # 填充 b 向量，动态使用剩余索引
    for i, row_index in enumerate(remaining_indices[1:]):
        b[i] = (PayoffMatrix[remaining_indices[0], remaining_indices[0]] - PayoffMatrix[row_index, row_index]) / num_of_players
    
    # 最后一项为 1
    b[-1] = 1

    return A, b

#print(main(2))


mode = 2
selection_intensity = 3.0
time_step = 0.01
time = 10000
PayoffMatrix, num_of_strategies, num_of_players = payoff_processing(mode)

s_final = numerical_new(PayoffMatrix, num_of_strategies, num_of_players, selection_intensity, time_step, time, selection_intensity)

'''
mode = 1
time_step = 0.01
PayoffMatrix, num_of_strategies, num_of_players = payoff_processing(mode)
delta_set = np.array([10**(-1), 10**(-0.9), 10**(-0.8), 10**(-0.7), 10**(-0.6), 10**(-0.5), 10**(-0.4), 10**(-0.3), 10**(-0.2), 10**(-0.1), 1,
                      10**(0.1), 10**(0.2), 10**(0.3), 10**(0.4), 10**(0.5), 10**(0.6), 10**(0.7), 10**(0.8), 10**(0.9), 10])

state_vector = []
for deltaindex, selection_intensity in enumerate(delta_set):
    time = max(int(10000/selection_intensity),10000)
    #time = 10000
    s_final = numerical_new(PayoffMatrix, num_of_strategies, num_of_players, selection_intensity, time_step, time, deltaindex)
    state_vector.append(s_final)

state_vector = np.vstack(state_vector)
np.savetxt('WithMTBR_selectionIntensity_numerical_withGTFT3',state_vector)

'''

