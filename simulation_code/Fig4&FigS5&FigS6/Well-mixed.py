# Configuration section
population_size = 14 # How many AIs in the population
mentor_instances = 14 # How many instances of each defined strategy there are

mentor_strategy = [0,2,3,4,5,6]
episode_length = 20 # How many turns to play
testing_episodes = 10 # How many episodes to play during the testing phase


s = 1 


import sys

import random

import time
start=time.time()
from matplotlib import pyplot as plt
import json
from collections import OrderedDict
import numpy as np
from math import exp


from concurrent.futures import ProcessPoolExecutor  

pool = ProcessPoolExecutor(max_workers=16)
S_list = [2.9,2.7,2.5,2.3,2.1,1.9,1.7,1.5,1.3,1.1,0.9,0.7,0.5,0.3,0.1,-0.1,-0.3,-0.5,-0.7,-0.9]
T_list = [4.9,4.7,4.5,4.3,4.1,3.9,3.7,3.5,3.3,3.1,2.9,2.7,2.5,2.3,2.1,1.9,1.7,1.5,1.3,1.1]


# Q agents learn the best action to perform for every state encountered
class AgentQ:
    def __init__(self, memory):
        self.wins = 0 # Number of times agent has won an episode
        self.losses = 0 # Number of times agent has lost an episode
        self.Q = {} # Stores the quality of each action in relation to each state
        self.memory = memory # The number of previous states the agent can factor into its decision

    def strategy_print(self):
        return "strategy00"
    
    def typeagent(self):
        return 0
        

        

# Defined agents know which action to perform
class AgentDefined:
    def __init__(self, strategy):
        self.wins = 0 # Number of times agent has won an episode
        self.losses = 0 # Number of times agent has lost an episode
        self.strategy = strategy
        
        self.deadlock_threshold = 3
        self.randomness_threshold = 8
        self.randomness_counter = 0
        self.deadlock_counter = 0  
        
        self.calm_count = 0
        self.punish_count = 0 

        
    def strategy_print(self):
        if self.strategy == 0:
            return "TFT"
        elif self.strategy == 1:
            return "GTFT0.3"
        elif self.strategy == 2:
            return "WSLS"
        elif self.strategy == 3:
            return "Holds a grudge"
        elif self.strategy == 4:
            return "Fool me once"
        elif self.strategy == 5:
            return "Omega TFT"
        elif self.strategy == 6:
            return "Gradual TFT"
        
    def typeagent(self):
        if self.strategy == 0:#"TFT"
            return 1 
        elif self.strategy == 2:#"WSLS"
            return 2
        elif self.strategy == 3:#"Holds a grudge"
            return 3
        elif self.strategy == 4:#"Fool me once"
            return 4
        elif self.strategy == 5:#"Omega TFT"
            return 5
        elif self.strategy == 6:#"Gradual TFT"
            return 6
        elif self.strategy == 7:#"ZDExtort2"
            return 7
        elif self.strategy == 8:#"ZDExtort2v2"
            return 8
        elif self.strategy == 9:#"ZDExtort3"
            return 9
        elif self.strategy == 10:#"ZDExtort4"
            return 10
        elif self.strategy == 11:#"ZDGen2"
            return 11
        elif self.strategy == 12:#"ZDGTFT2"
            return 12
        elif self.strategy == 13:#"ZDMischief"
            return 13
        elif self.strategy == 14:#"ZDSet2"
            return 14


# Stores all AIs
population = []

# Stores record of analysis of all AIs
population_analysis = []

# Stores all instances of defined strategies
mentors = []


# TODO: Mentor analysis


import json
filename='D:\code\prisoners-dilemma-q-master\strategy0729.txt'

with open(filename,'r') as file:
    strategy00=json.load(file)
strategy00_dict={item[0]: item[1] for item in strategy00}
# Create a random AI with a random amount of memory
for i in range(population_size):
    population.append(AgentQ(random.randint(2,2)))
    population[i].Q=strategy00_dict

# Create instances of defined strategies
# for i in range(len(mentor_strategy)): # Number of defined strategies
#     for j in range(mentor_instances):
#         mentors.append(AgentDefined(i))
for i in mentor_strategy : # Number of defined strategies
    for j in range(mentor_instances):
        mentors.append(AgentDefined(i))





def main(S,T):

    strategy0729payoffmatrix = []
    strategy0729payoffmatrix = np.load('PAYOFFMATRIX\strategy0907payoffmatrixS=' + str(S) +'T=' + str(T) +'.npy') 
    strategy0729cooprate = np.load('PAYOFFMATRIX\strategy0907cooprateS=' + str(S) +'T=' + str(T) +'.npy')
    #导入策略两两交互的结果
    
    
    # Testing mode
    
    max_episode = 100
    avg_num_strategy= [ [ 0.0 for _ in range(16)  ] for _ in range(max_episode) ] #统计策略的变化
    avg_num_strategy=np.array(avg_num_strategy)
    
    avg_reward_globle = [0.0 for _ in range(max_episode)]
    avg_reward_globle=np.array(avg_reward_globle)
    
    avg_cooprate_globle = [0.0 for _ in range(max_episode)]
    avg_cooprate_globle=np.array(avg_cooprate_globle)
    
    repeattimes = 100
    for repeatnum in range(repeattimes):
        
        bigpopulation=population+mentors
        
        
        # num_strategy = [ [ 0 for _ in range(len(mentor_strategy) + 1)  ] for _ in range(testing_episodes) ] #统计策略的变化
    
        num_strategy = [ [ 0 for _ in range(16)  ] for _ in range(max_episode) ] #统计策略的变化
        num_strategy=np.array(num_strategy)
        # 最多可能有16个策略
        episode_total = 0
        avg_reward_total= [0.0 for _ in range(max_episode)]
        avg_reward_total=np.array(avg_reward_total)
        avg_cooprate_total= [0.0 for _ in range(max_episode)]
        avg_cooprate_total=np.array(avg_cooprate_total)
        
        for iepisode in range(max_episode):
            
            random.shuffle(bigpopulation)
            reward_episode =  [[ 0 for _ in range(len(bigpopulation))] for _ in range(len(bigpopulation))]
            
            cooprate_episode = [[ 0 for _ in range(len(bigpopulation))] for _ in range(len(bigpopulation))]
            avg_reward = [0 for _ in range(len(bigpopulation))]
            avg_cooprate = [0 for _ in range(len(bigpopulation))]
            
            type_player = [ i.typeagent() for i in bigpopulation ]
        
            
            for iplayer1 in range(len(bigpopulation)-1):  #所有人两两博弈
                
                for iplayer2 in range(iplayer1+1,len(bigpopulation)):
                    
                
                    player1 = bigpopulation[iplayer1]
                    player2 = bigpopulation[iplayer2]
                    
                    # type1=2*player1.typeagent()+random.randint(0, 1)
                    # type2=2*player2.typeagent()+random.randint(0, 1)
                    
                    
                    
                    # type1=2*type_player[iplayer1]+random.randint(0, 1)
                    # type2=2*type_player[iplayer2]+random.randint(0, 1)
                    reward_episode[iplayer1][iplayer2] =  strategy0729payoffmatrix[type_player[iplayer1]][type_player[iplayer2]]
                    reward_episode[iplayer2][iplayer1] =  strategy0729payoffmatrix[type_player[iplayer2]][type_player[iplayer1]]
                    
                    cooprate_episode[iplayer1][iplayer2] =  strategy0729cooprate[type_player[iplayer1]][type_player[iplayer2]]
                    cooprate_episode[iplayer2][iplayer1] =  strategy0729cooprate[type_player[iplayer2]][type_player[iplayer1]]
                
               
            for iavg_reward in range(len(bigpopulation)) : #求平均收益 + 平均合作率
                avg_reward[iavg_reward] = sum(reward_episode[iavg_reward]) / (len(bigpopulation)-1)
                avg_cooprate[iavg_reward] = sum(cooprate_episode[iavg_reward]) / (len(bigpopulation)-1)
            #求所有人的平均收益
            avg_reward_total[iepisode] = sum(avg_reward) / (len(bigpopulation)-1)
            avg_cooprate_total[iepisode] = sum(avg_cooprate) / (len(bigpopulation)-1)
               
            for irefresh in range(len(bigpopulation)): #更新
                opponent_refresh = random.randint(0,len(bigpopulation)-1)
                
                Ptrans=1 / (exp(  s *(avg_reward[irefresh] - avg_reward[opponent_refresh]) )  + 1)
                if random.random()<Ptrans:
                    bigpopulation[irefresh] = bigpopulation[opponent_refresh]            
        
            # 绘图来统计策略的演化轨迹
            
            for i_num in range(len(bigpopulation)) :
                if type_player[i_num] == 0 :
                    num_strategy[ iepisode][0] +=1
                elif type_player[i_num] == 1 :
                    num_strategy[ iepisode][1] +=1
                elif type_player[i_num] == -1 :
                    num_strategy[ iepisode][2] +=1
                elif type_player[i_num] == 2 :
                    num_strategy[ iepisode][3] +=1
                elif type_player[i_num] == 3 :
                    num_strategy[ iepisode][4] +=1
                elif type_player[i_num] == 4 :
                    num_strategy[ iepisode][5] +=1
                elif type_player[i_num] == 5 :
                    num_strategy[ iepisode][6] +=1
                elif type_player[i_num] == 6 :
                    num_strategy[ iepisode][7] +=1
                elif type_player[i_num] == 7 :
                    num_strategy[ iepisode][8] +=1
                elif type_player[i_num] == 8 :
                    num_strategy[ iepisode][9] +=1
                elif type_player[i_num] == 9 :
                    num_strategy[ iepisode][10] +=1
                elif type_player[i_num] == 10 :
                    num_strategy[ iepisode][11] +=1
                elif type_player[i_num] == 11 :
                    num_strategy[ iepisode][12] +=1
                elif type_player[i_num] == 12 :
                    num_strategy[ iepisode][13] +=1
                elif type_player[i_num] == 13 :
                    num_strategy[ iepisode][14] +=1
                elif type_player[i_num] == 14 :
                    num_strategy[ iepisode][15] +=1
             
            episode_total += 1
            
            ifbreak = 0 #检查是否已经演化完毕
            for check_break in range(16):  #最多可能有16个策略
                if num_strategy[ iepisode][check_break] >= mentor_instances * (len(mentor_strategy) + 1) :
                    for iall in range(iepisode+1,max_episode):
                        
                        num_strategy[ iall][check_break] = mentor_instances * (len(mentor_strategy) + 1) 
                        avg_reward_total[iall] = avg_reward_total[iepisode]
                        avg_cooprate_total[iall] = avg_cooprate_total[iepisode]
                    # 如果一个700，则全记700，平均收益记最后一步
                    ifbreak = 1024;
                    
            # print (episode_total)
            
            if ifbreak == 1024 or  episode_total == max_episode:
                avg_num_strategy += num_strategy/repeattimes
                avg_reward_globle += avg_reward_total/repeattimes/20
                avg_cooprate_globle += avg_cooprate_total/repeattimes
                break;
    # return (avg_num_strategy)
    return (avg_cooprate_globle[-1]), (avg_reward_globle[-1])


if __name__=="__main__":
    
    final_result=[]
    final_result2=[]
    for s in S_list:
        s = [s] * len(T_list)
        result = list( zip(*pool.map(main,s,T_list) ))
        final_result.append(result[0])
        final_result2.append(result[1])
                
    end = time.time()
    print(end-start) 
    
    final_result=np.flip ( np.array(final_result)    , axis=1)
    final_result2=np.flip ( np.array(final_result2)    , axis=1)
    # file_path = "fig3_复杂网络+收益矩阵/策略演化_WM.npy"
    # np.save("fig3.1_复杂网络+收益矩阵/策略演化_WM_合作率.npy", final_result)
    # np.save("fig3.2_复杂网络+收益矩阵/策略演化_WM_平均收益.npy", final_result2)
    
#%%
    x_labels = [1.1,1.3,1.5,1.7,1.9,2.1,2.3,2.5,2.7,2.9,3.1,3.3,3.5,3.7,3.9,4.1,4.3,4.5,4.7,4.9]
    y_labels = [2.9,2.7,2.5,2.3,2.1,1.9,1.7,1.5,1.3,1.1,0.9,0.7,0.5,0.3,0.1,-0.1,-0.3,-0.5,-0.7,-0.9]
    

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.axis('off')
    plt.tick_params(axis='both', direction='in')
    
    from mpl_toolkits.axes_grid1.inset_locator import inset_axes
    
    main_axes = inset_axes(ax, width="80%", height="80%", loc='center')

    img = main_axes.imshow(final_result, cmap='YlGnBu_r', interpolation='nearest', vmin=1, vmax=0)
    main_axes.set_xticks(np.arange(len(x_labels)))
    main_axes.set_yticks(np.arange(len(y_labels)))
    
    main_axes.set_xticklabels(x_labels, fontsize=14)
    main_axes.set_yticklabels(y_labels, fontsize=14)
    
    main_axes.set_xlabel('T',fontsize=22)
    main_axes.set_ylabel('S',fontsize=22)
    
    cax = inset_axes(ax, width="3%", height="80%", loc='center right')
    fig.colorbar(img, cax=cax, shrink=0.72)
    plt.show()













# y={}

# for j in range(16):#最多可能有16个策略
#     # y[j]=[  i[j] for i in num_strategy ]     
#     y[j]=[  avg_num_strategy[i][j] for i in range(max_episode) ] 
# fig =plt.figure(1)#绘图
# fig.suptitle('evolution of strategy')
# # x=range(testing_episodes)
# x=range(max_episode)
# plt.plot(x,y[0],color='black',linewidth=1.0, linestyle="-",label="strategy729")#
# plt.plot(x,y[1],color='#0000FF',linewidth=1.0, linestyle="-",label="TFT")#
# # plt.plot(x,y[2],color='purple',linewidth=1.0, linestyle="-",label="GTFT0.3")#
# plt.plot(x,y[3],color='#00FFFF',linewidth=1.0, linestyle="-",label="WSLS")#
# plt.plot(x,y[4],color='#FF0000',linewidth=1.0, linestyle="-",label="Holds a grudge")#
# plt.plot(x,y[5],color='#00FF80',linewidth=1.0, linestyle="-",label="Fool me once")#
# plt.plot(x,y[6],color='#FFFF00',linewidth=1.0, linestyle="-",label="Omega TFT")#
# plt.plot(x,y[7],color='#FF00FF',linewidth=1.0, linestyle="-",label="Gradual TFT")#

# linestyle_zd = [ [] for _ in range(8) ]  #循环使用颜色和线
# for i in range(8):
#     # color_zd[i] =  plt.cm.viridis(i / 8)
#     linestyle_zd[i] = [ '--', '-.', ':'][i % 3]
# color_zd = plt.cm.viridis(np.linspace(0, 1, 15))
# plt.plot(x,y[8],color=color_zd[0],linewidth=1.0, linestyle=linestyle_zd[0],label="ZDExtort2")#
# plt.plot(x,y[9],color=color_zd[1],linewidth=1.0, linestyle=linestyle_zd[1],label="ZDExtort2")#
# plt.plot(x,y[10],color=color_zd[2],linewidth=1.0, linestyle=linestyle_zd[2],label="ZDExtort3")#
# plt.plot(x,y[11],color=color_zd[3],linewidth=1.0, linestyle=linestyle_zd[3],label="ZDExtort4")#
# plt.plot(x,y[12],color=color_zd[4],linewidth=1.0, linestyle=linestyle_zd[4],label="ZDGen2")#
# plt.plot(x,y[13],color=color_zd[5],linewidth=1.0, linestyle=linestyle_zd[5],label="ZDGTFT2")#
# plt.plot(x,y[14],color=color_zd[6],linewidth=1.0, linestyle=linestyle_zd[6],label="ZDMischief")#
# plt.plot(x,y[15],color=color_zd[7],linewidth=1.0, linestyle=linestyle_zd[7],label="ZDSet2")#
# plt.legend(loc="best")
# # plt.xlim(1,2)
# # plt.ylim(0,1)
# plt.xlabel('episodes')
# plt.ylabel('num of strategy')
    
# for i_print in range(len(bigpopulation)):
#     print (   bigpopulation[i_print].strategy_print()     )

# fig = plt.figure(2)
# fig.suptitle('globle_avg_payoff during evolutionary process ')
# x=range(max_episode)
# y=avg_reward_globle
# plt.plot(x,y,linewidth=1.0, linestyle="-")
# plt.ylim(1,3)
# plt.legend(loc="best")
# plt.xlabel('episodes')
# plt.ylabel('globle_avg_payoff')
# plt.show
    


