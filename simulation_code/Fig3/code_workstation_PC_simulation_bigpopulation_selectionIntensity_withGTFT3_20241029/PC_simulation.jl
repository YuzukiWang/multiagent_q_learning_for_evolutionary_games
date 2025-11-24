using LinearAlgebra
using Random
using Distributions  # for Categorical
using ArgParse
using SparseArrays
using DelimitedFiles

function state_initialization(numOfStrategies)
    # 直接创建一个 numOfStrategies x 1 矩阵，所有元素初始化为 500
    matrix = fill(500, numOfStrategies)
    return matrix
end


function calculate_fitness(u, delta)
    f = exp.(u * delta)
    return f
end


function calculate_payoff(PayoffMatrix, x, num_nodes)
    u_temp = (PayoffMatrix*x .- diag(PayoffMatrix)) ./ (num_nodes-1)
    return u_temp
end


function update_state_PC(f, x, numOfStrategies,num_nodes)
    updated_one = rand(Categorical(vec(x./num_nodes)))  # select the player to be updated

    vec_new = zeros(numOfStrategies)  # 创建一个长度为 n 的全零向量
    @inbounds vec_new[updated_one] = 1  # 使用 @inbounds 关闭边界检查以提高速度

    e = x - vec_new
    e = e./(num_nodes-1)
    selected_element = rand(Categorical(vec(e)))

    p_replaced = f[selected_element]/(f[selected_element] + f[updated_one])

    if rand()<=p_replaced
        x[updated_one] -= 1
        x[selected_element] += 1
    end

    return x
end

function simulation(PayoffMatrix, numOfStrategies, delta)
    generation = max(floor(Int, (10000/delta)),10000)  # 总的代数
    
    num_nodes = 500 * numOfStrategies
    total_updates = generation * num_nodes  # 总的更新次数

    x = state_initialization(numOfStrategies)  # 初始化状态
    history = []  # 存储记录的状态

    # 记录初始状态（第0代）
    push!(history, copy(x./num_nodes))

    for t in 1:total_updates
        u = calculate_payoff(PayoffMatrix, x, num_nodes)
        f = calculate_fitness(u, delta)
        x = update_state_PC(f, x, numOfStrategies, num_nodes)

        # 每隔 10*num_nodes 次（即每10代）记录一次状态
        if t % (10 * num_nodes) == 0
            push!(history, copy(x./num_nodes))  # 记录每隔10代的状态
        end
    end

    return history  # 返回记录的数据
end


function main(mode, delta_index, beginning_index, ending_index)
    file_path = "payoffMatrix.txt"
    PM = readdlm(file_path)

    delta_set = [10^(-1), 10^(-0.9), 10^(-0.8), 10^(-0.7), 10^(-0.6), 10^(-0.5), 10^(-0.4), 10^(-0.3), 10^(-0.2), 10^(-0.1), 1, 10^(0.1), 10^(0.2), 10^(0.3), 10^(0.4), 10^(0.5), 10^(0.6), 10^(0.7), 10^(0.8), 10^(0.9), 10] 
    delta = delta_set[delta_index]
    for index in beginning_index:ending_index
        if mode == 1
            PayoffMatrix = PM
            outfile = "WithMTBR_deltaindex$(delta_index)_$(index).txt"
        elseif mode ==2
            PayoffMatrix = PM[2:end,2:end]
            outfile = "WithoutMTBR_deltaindex$(delta_index)_$(index).txt"
        end
        numOfStrategies = size(PayoffMatrix,1)
        num = simulation(PayoffMatrix, numOfStrategies, delta)

        writedlm(outfile, num)
    end

end

if abspath(PROGRAM_FILE) == @__FILE__
    if length(ARGS) < 4
        println("Usage: julia DB_simulation.jl <delta> <mutation_rate> <iteration_number> <index>")
    else
        mode = parse(Int, ARGS[1]) 
        delta_index = parse(Int, ARGS[2])
        beginning_index = parse(Int, ARGS[3])
        ending_index = parse(Int, ARGS[4])
        main(mode, delta_index, beginning_index, ending_index)
    end
end

#=
delta = 0.1

file_path = "payoffMatrix.txt"
PM = readdlm(file_path)
println("PM:$(PM)")
#PayoffMatrix = PM[2:end,2:end]
PayoffMatrix = PM
println("PayoffMatrix:$(PayoffMatrix)")
numOfStrategies = size(PayoffMatrix,1)
println("numOfStrategies:$(numOfStrategies)")
num_nodes = 500 * numOfStrategies
println("num_nodes:$(num_nodes)")

x = matrix = fill(500, numOfStrategies)
println("intial state:$(x)")

for t in 1:(num_nodes*10000)
    u = calculate_payoff(PayoffMatrix, x, num_nodes)
    f = calculate_fitness(u, delta)
    global x = update_state_PC(f, x, numOfStrategies, num_nodes)

    # 每隔 10*num_nodes 次（即每10代）记录一次状态
    if t % ( 10*num_nodes) == 0
        println("state:$(x)")
    end
end
=#