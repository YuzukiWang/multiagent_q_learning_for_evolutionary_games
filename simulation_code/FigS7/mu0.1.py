# Configuration section
population_size = 33 # How many AIs in the population此处改为0即为没有智能体
mentor_instances = 33 # How many instances of each defined strategy there are
# mentor_types = 7 #测试训练集
mentor_strategy = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14]
# mentor_strategy = [0,1,2,3,4,5,6]
episode_length = 20 # How many turns to play
testing_episodes = 10 # How many episodes to play during the testing phase


s = 20 #选择强度

from time import time

start_time = time()


# Prisoner's dillema rewards [Player 1 reward, Player 2 reward]

R = 2
S = 0
T = 3
P = 0.1
reward_matrix = [[[R, R], # Both players cooperate
                [S, T], # Player 1 cooperates, player 2 defects
                [T, S], # Player 1 defects, player 2 cooperates
                [P, P]]] # Both players defect

# Script section
import sys

import random

from time import time
from matplotlib import pyplot as plt
plt.rcParams['font.family'] = 'Times New Roman'
import json
from collections import OrderedDict
import numpy as np
from math import exp

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
        
        
    def get_q(self, state):
        quality1 = self.Q[str(state[-self.memory:])][0]
        quality2 = self.Q[str(state[-self.memory:])][1]

        return quality1, quality2

    def set_q(self, state, quality1, quality2):
        self.Q[str(state[-self.memory:])][0] = quality1
        self.Q[str(state[-self.memory:])][1] = quality2



    def max_q(self, state):
        #quality1, quality2 = self.get_q(state)
        if state==[]:
            quality1,quality2=0,0
        else:
            quality1, quality2 = self.get_q(state)
        if quality1 == quality2:
            return random.randint(0, 1)
        elif quality1 > quality2:
            return 0
        else:
            return 1

    def pick_action(self, state):
        # Decrease learning rate   
        return self.max_q(state)


    def mark_victory(self):
        self.wins += 1

    def mark_defeat(self):
        self.losses += 1

    def analyse(self):
        # What percentage of games resulted in victory/defeat
        percent_won = 0
        if self.wins > 0:
            percent_won = float(self.wins) / (self.wins + self.losses)
        
        '''
        percent_lost = 0
        if self.losses > 0:
            percent_lost = float(self.losses) / (self.wins + self.losses)
        '''

        # How many states will result in cooperation/defection
        times_cooperated = 0
        times_defected = 0

        for state in self.Q:
            action = self.max_q(eval(state))

            if action == 0:
                times_cooperated += 1
            else:
                times_defected += 1

        # What percentage of states will result in cooperation/defection
        percent_cooperated = 0
        if times_cooperated > 0:
            percent_cooperated = float(times_cooperated) / len(self.Q)

        '''
        percent_defected = 0
        if times_defected > 0:
            percent_defected = float(times_defected) / len(self.Q)
        '''

        # Return most relevant analysis
        return self.wins, percent_won, percent_cooperated

    def reset_analysis(self):
        self.wins = 0
        self.losses = 0
        

# Defined agents know which action to perform
class AgentDefined:
    def __init__(self, strategy):
        self.wins = 0 # Number of times agent has won an episode
        self.losses = 0 # Number of times agent has lost an episode
        self.strategy = strategy
        
        self.deadlock_threshold = 3
        self.randomness_threshold = 8
        self.randomness_counter = 0
        self.deadlock_counter = 0  #omegaTFT用
        
        self.calm_count = 0
        self.punish_count = 0 #gradual用

        
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
        elif self.strategy == 1:#"GTFT0.3"
            return 2
        elif self.strategy == 2:#"WSLS"
            return 3
        elif self.strategy == 3:#"Holds a grudge"
            return 4
        elif self.strategy == 4:#"Fool me once"
            return 5
        elif self.strategy == 5:#"Omega TFT"
            return 6
        elif self.strategy == 6:#"Gradual TFT"
            return 7
        elif self.strategy == 7:#"ZDExtort2"
            return 8
        elif self.strategy == 8:#"ZDExtort2v2"
            return 9
        elif self.strategy == 9:#"ZDExtort3"
            return 10
        elif self.strategy == 10:#"ZDExtort4"
            return 11
        elif self.strategy == 11:#"ZDGen2"
            return 12
        elif self.strategy == 12:#"ZDGTFT2"
            return 13
        elif self.strategy == 13:#"ZDMischief"
            return 14
        elif self.strategy == 14:#"ZDSet2"
            return 15
        


    def pick_action(self, state):
        if self.strategy == 0: # Tit for tat
            if len(state) == 0: # On the first tern
                return random.randint(0, 1) # Random
            else: # Otherwise
                # return state[-1] # for only memory the opposite # Pick the last action of the opponent  
                return state[-1][0] # Pick the last action of the opponent
            
        elif self.strategy == 1: # GTFT
            if len(state) == 0: # On the first tern
                return random.randint(0, 1) # Random
            elif state[-1][0] == 0: # Otherwise
                # return state[-1] # for only memory the opposite # Pick the last action of the opponent  
                return state[-1][0] # Pick the last action of the opponent 
            elif state[-1][0] == 1:
                #Pc=min( (1-(T-R)/(R-S)), ((R-P)/(T-P)) )
                Pc=1/3
                if random.random()<Pc:
                    return 0
                else: 
                    return 1
            else:
                print("ERROR")
                return 0
            
        
        elif self.strategy == 2: # WSLS
            if len(state) == 0: # On the first tern
                return random.randint(0, 1) # Random
            elif state[-1][0] == 0:
                return state[-1][1]
            elif state[-1][0] == 1:
                return 1-state[-1][1]
            else:
                print("ERROR")
                return 0
            
        elif self.strategy == 3: # Holds a grudge
            # if 1 in state: # If the enemy has ever defected
            state=np.array(state)
            if len(state) == 0: # On the first tern
                return random.randint(0, 1) # Random
            # elif 1 in state: # If the enemy has ever defected
            elif 1 in state[:,0]: # If the enemy has ever defected
                return 1 # Defect
            else: # Otherwise
                return 0 # Cooperate
            
        elif self.strategy == 4: # Fool me once
            state=np.array(state)
            if len(state) == 0: # On the first tern
                return random.randint(0, 1) #cooperate
            defect_count=0
            for i in range(len(state[:,0])):
                if state[i,0] == 1:
                    defect_count+=1
            if defect_count >= 2:
                return 1
            elif defect_count ==1 and state[-1][0]==1:
                return 1
            else :
                return 0
        
        elif self.strategy == 5: #omegaTFT
            state=np.array(state)
            if len(state) == 0:
                self.deadlock_threshold = 3
                self.randomness_threshold = 8
                self.randomness_counter = 0
                self.deadlock_counter = 0
                return random.randint(0, 1)
            if len(state[:,0]) == 1:
                return state[-1,0]
            if self.deadlock_counter >= self.deadlock_threshold:
                move = 0 #需要解除死锁
                if self.deadlock_counter == self.deadlock_threshold:
                    self.deadlock_counter = self.deadlock_threshold + 1
                else:
                    self.deadlock_counter = 0
            else: 
                if state[-2,0] == [0]  and state[-1,0] == [0]:
                    self.randomness_counter -= 1
            # If the opponent's move changed, increase the counter
                if state[-2,0] != state[-1,0]:
                    self.randomness_counter += 1
            # If the opponent's last move differed from mine,
            # increase the counter
                if state[-1,0] != state[-1,1]:
                    self.randomness_counter += 1
            # Compare counts to thresholds
            # If self.randomness_counter exceeds Y, Defect for the remainder
                if self.randomness_counter >= self.randomness_threshold:
                    move = 1 #超过阈值得allD
                else:
                # TFT
                    move = state[-1][0]
                # Check for deadlock
                    if state[-2,0] != state[-1,0]:
                        self.deadlock_counter += 1
                    else:
                        self.deadlock_counter = 0
            return move
        
        elif self.strategy == 6: #gradual
            state=np.array(state)
            if len(state) == 0:
                self.calm_count = 0
                self.punish_count = 0
                return random.randint(0, 1)

            if self.punish_count > 0:
                self.punish_count -= 1
                return 1

            if self.calm_count > 0:
                self.calm_count -= 1
                return 0

            if state[-1,0] == 1:
                defect_count=0
                for i in range(len(state[:,0])):
                    if state[i,0] == 1:
                        defect_count+=1
                self.punish_count = defect_count - 1
                self.calm_count = 2
                return 1
            return 0
                
            
            
        

    def reward_action(self, state, action, reward):
        pass # Since these agents are defined, no learning occurs

    def mark_victory(self):
        self.wins += 1

    def mark_defeat(self):
        self.losses += 1

    def analyse(self):
        # What percentage of games resulted in victory/defeat
        percent_won = 0
        if self.wins > 0:
            percent_won = float(self.wins) / (self.wins + self.losses)
        
        percent_lost = 0
        if self.losses > 0:
            percent_lost = float(self.losses) / (self.wins + self.losses)

        # Return most relevant analysis
        return self.wins, percent_won

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


strategy0729payoffmatrix = []
strategy0729payoffmatrix = np.load('strategy0729payoffmatrixeta=0.00.npy') #导入策略两两交互的结果


# Testing mode

max_episode = 100
avg_num_strategy= [ [ 0.0 for _ in range(16)  ] for _ in range(max_episode) ] #统计策略的变化
avg_num_strategy=np.array(avg_num_strategy)

avg_reward_globle = [0.0 for _ in range(max_episode)]
avg_reward_globle=np.array(avg_reward_globle)




# population=[] #不加入BRTM
repeattimes = 50

mu = 0.1
strategy_pool = population+mentors

i_num_strategy = [ [ [ 0 for _ in range(16)  ] for _ in range(max_episode) ] for _ in range(repeattimes) ]
i_reward_globle = [ [0.0 for _ in range(max_episode)] for _ in range(repeattimes) ]              
                  
for repeatnum in range(repeattimes):
    
    bigpopulation=population+mentors
    

    
    
    # num_strategy = [ [ 0 for _ in range(len(mentor_strategy) + 1)  ] for _ in range(testing_episodes) ] #统计策略的变化

    num_strategy = [ [ 0 for _ in range(16)  ] for _ in range(max_episode) ] #统计策略的变化
    num_strategy=np.array(num_strategy)
    # 最多可能有16个策略
    episode_total = 0
    avg_reward_total= [0.0 for _ in range(max_episode)]
    avg_reward_total=np.array(avg_reward_total)
    
    for iepisode in range(max_episode):
        
        random.shuffle(bigpopulation)
        reward_episode =  [[ 0 for _ in range(len(bigpopulation))] for _ in range(len(bigpopulation))]
        avg_reward = [0 for _ in range(len(bigpopulation))]
        
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
           
        for iavg_reward in range(len(bigpopulation)) : #求平均收益
            avg_reward[iavg_reward] = sum(reward_episode[iavg_reward]) / (len(bigpopulation)-1)
            
        #求所有人的平均收益
        avg_reward_total[iepisode] = sum(avg_reward) / (len(bigpopulation)-1)
           
        for irefresh in range(len(bigpopulation)): #更新
            if random.random() < mu:
        # 发生突变：从策略池中随机选一个新策略
                bigpopulation[irefresh] = random.choice(strategy_pool)
            else:
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
            elif type_player[i_num] == 2 :
                num_strategy[ iepisode][2] +=1
            elif type_player[i_num] == 3 :
                num_strategy[ iepisode][3] +=1
            elif type_player[i_num] == 4 :
                num_strategy[ iepisode][4] +=1
            elif type_player[i_num] == 5 :
                num_strategy[ iepisode][5] +=1
            elif type_player[i_num] == 6 :
                num_strategy[ iepisode][6] +=1
            elif type_player[i_num] == 7 :
                num_strategy[ iepisode][7] +=1
            elif type_player[i_num] == 8 :
                num_strategy[ iepisode][8] +=1
            elif type_player[i_num] == 9 :
                num_strategy[ iepisode][9] +=1
            elif type_player[i_num] == 10 :
                num_strategy[ iepisode][10] +=1
            elif type_player[i_num] == 11 :
                num_strategy[ iepisode][11] +=1
            elif type_player[i_num] == 12 :
                num_strategy[ iepisode][12] +=1
            elif type_player[i_num] == 13 :
                num_strategy[ iepisode][13] +=1
            elif type_player[i_num] == 14 :
                num_strategy[ iepisode][14] +=1
            elif type_player[i_num] == 15 :
                num_strategy[ iepisode][15] +=1
         
        episode_total += 1
        
        ifbreak = 0 #检查是否已经演化完毕
        for check_break in range(16):  #最多可能有16个策略
            if num_strategy[ iepisode][check_break] >= mentor_instances * (len(mentor_strategy) + 1) :
                for iall in range(iepisode+1,max_episode):
                    
                    num_strategy[ iall][check_break] = mentor_instances * (len(mentor_strategy) + 1) 
                    avg_reward_total[iall] = avg_reward_total[iepisode]
                # 如果一个700，则全记700，平均收益记最后一步
                ifbreak = 1024;
                
        # print (episode_total)
        
        if ifbreak == 1024 or  episode_total == max_episode:
            avg_num_strategy += num_strategy/repeattimes
            avg_reward_globle += avg_reward_total/repeattimes/20
            i_num_strategy[repeatnum] = num_strategy
            i_reward_globle[repeatnum] += avg_reward_total/20
            break;
            
end_time = time()
print(f"运行时间: {end_time - start_time:.4f} 秒")
#%%



plt.rcParams['font.family'] = 'Times New Roman'
y={}

for j in range(16):#最多可能有16个策略
    # y[j]=[  i[j] for i in num_strategy ]     
    y[j]=[  ( avg_num_strategy[i][j] /  ( mentor_instances * (len(mentor_strategy) + population_size/33 ) ) )  for i in range(max_episode) ] 
# fig =plt.figure(1)#绘图

file_path = "突变/mu01.npy"
np.save(file_path, y)

fig,ax_main = plt.subplots(figsize=(6, 6))
ax_main.tick_params(axis='both', direction='in')
    
fig.suptitle('Evolution of Strategies',fontsize=18)
# x=range(testing_episodes)
x=[value * 224 for value in range(max_episode)]
ax_main.plot(x,y[0],color='black',linewidth=1.0, linestyle="-",label="BRTM")#
ax_main.plot(x,y[1],color='#0000FF',linewidth=1.0, linestyle="-",label="TFT")#
ax_main.plot(x,y[2],color='purple',linewidth=1.0, linestyle="-",label="GTFT0.3")#
ax_main.plot(x,y[3],color='#00FFFF',linewidth=1.0, linestyle="-",label="WSLS")#
ax_main.plot(x,y[4],color='#FF0000',linewidth=1.0, linestyle="-",label="Holds a grudge")#
ax_main.plot(x,y[5],color='#00FF80',linewidth=1.0, linestyle="-",label="Fool me once")#
ax_main.plot(x,y[6],color='#FFFF00',linewidth=1.0, linestyle="-",label="Omega TFT")#
ax_main.plot(x,y[7],color='#FF00FF',linewidth=1.0, linestyle="-",label="Gradual TFT")#

linestyle_zd = [ [] for _ in range(8) ]  #循环使用颜色和线
for i in range(8):
    # color_zd[i] =  ax_main.cm.viridis(i / 8)
    linestyle_zd[i] = [ '--', '-.', ':'][i % 3]
color_zd = plt.cm.viridis(np.linspace(0, 1, 15))
ax_main.plot(x,y[8],color=color_zd[0],linewidth=1.0, linestyle=linestyle_zd[0],label="ZDExtort2")#
ax_main.plot(x,y[9],color=color_zd[1],linewidth=1.0, linestyle=linestyle_zd[1],label="ZDExtort2v2")#
ax_main.plot(x,y[10],color=color_zd[2],linewidth=1.0, linestyle=linestyle_zd[2],label="ZDExtort3")#
ax_main.plot(x,y[11],color=color_zd[3],linewidth=1.0, linestyle=linestyle_zd[3],label="ZDExtort4")#
ax_main.plot(x,y[12],color=color_zd[4],linewidth=1.0, linestyle=linestyle_zd[4],label="ZDGen2")#
ax_main.plot(x,y[13],color=color_zd[5],linewidth=1.0, linestyle=linestyle_zd[5],label="ZDGTFT2")#
ax_main.plot(x,y[14],color=color_zd[6],linewidth=1.0, linestyle=linestyle_zd[6],label="ZDMischief")#
ax_main.plot(x,y[15],color=color_zd[7],linewidth=1.0, linestyle=linestyle_zd[7],label="ZDSet2")#
# plt.legend(loc="best")

yi=[ y.copy() for _ in range(repeattimes)]
alpha = 0.1#透明度
for irepeat in range(repeattimes):
    for jrepeat in range(16):
        yi[irepeat][jrepeat] = [  ( i_num_strategy[irepeat][i][jrepeat] /  ( mentor_instances * (len(mentor_strategy) + population_size/33) ) )  for i in range(max_episode) ]
    x=[value * 224 for value in range(max_episode)]
    ax_main.plot(x,yi[irepeat][0],color='black',linewidth=1.0, linestyle="-",label="BRTM",alpha=alpha)#
    ax_main.plot(x,yi[irepeat][1],color='#0000FF',linewidth=1.0, linestyle="-",label="TFT",alpha=alpha)#
    ax_main.plot(x,yi[irepeat][2],color='purple',linewidth=1.0, linestyle="-",label="GTFT0.3",alpha=alpha)#
    ax_main.plot(x,yi[irepeat][3],color='#00FFFF',linewidth=1.0, linestyle="-",label="WSLS",alpha=alpha)#
    ax_main.plot(x,yi[irepeat][4],color='#FF0000',linewidth=1.0, linestyle="-",label="Holds a grudge",alpha=alpha)#
    ax_main.plot(x,yi[irepeat][5],color='#00FF80',linewidth=1.0, linestyle="-",label="Fool me once",alpha=alpha)#
    ax_main.plot(x,yi[irepeat][6],color='#FFFF00',linewidth=1.0, linestyle="-",label="Omega TFT",alpha=alpha)#
    ax_main.plot(x,yi[irepeat][7],color='#FF00FF',linewidth=1.0, linestyle="-",label="Gradual TFT",alpha=alpha)#
    
    linestyle_zd = [ [] for _ in range(8) ]  #循环使用颜色和线
    for i in range(8):
        # color_zd[i] =  ax_main.cm.viridis(i / 8)
        linestyle_zd[i] = [ '--', '-.', ':'][i % 3]
    color_zd = plt.cm.viridis(np.linspace(0, 1, 15))
    ax_main.plot(x,yi[irepeat][8],color=color_zd[0],linewidth=1.0, linestyle=linestyle_zd[0],label="ZDExtort2",alpha=alpha)#
    ax_main.plot(x,yi[irepeat][9],color=color_zd[1],linewidth=1.0, linestyle=linestyle_zd[1],label="ZDExtort2v2",alpha=alpha)#
    ax_main.plot(x,yi[irepeat][10],color=color_zd[2],linewidth=1.0, linestyle=linestyle_zd[2],label="ZDExtort3",alpha=alpha)#
    ax_main.plot(x,yi[irepeat][11],color=color_zd[3],linewidth=1.0, linestyle=linestyle_zd[3],label="ZDExtort4",alpha=alpha)#
    ax_main.plot(x,yi[irepeat][12],color=color_zd[4],linewidth=1.0, linestyle=linestyle_zd[4],label="ZDGen2",alpha=alpha)#
    ax_main.plot(x,yi[irepeat][13],color=color_zd[5],linewidth=1.0, linestyle=linestyle_zd[5],label="ZDGTFT2",alpha=alpha)#
    ax_main.plot(x,yi[irepeat][14],color=color_zd[6],linewidth=1.0, linestyle=linestyle_zd[6],label="ZDMischief",alpha=alpha)#
    ax_main.plot(x,yi[irepeat][15],color=color_zd[7],linewidth=1.0, linestyle=linestyle_zd[7],label="ZDSet2",alpha=alpha)#
    # plt.legend(loc="best")
    
file_path = "突变/mu01_多次.npy"
np.save(file_path, yi)
    
legend_main = ax_main.legend(loc='upper left')
legend_main.remove()
# plt.xlim(1,2)
# plt.ylim(0,1)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.xlabel('Generations',fontsize=18)
plt.ylabel('Fraction of Strategies',fontsize=18)

fig_legend, ax_legend = plt.subplots()
ax_legend.axis('off')
legend_legend1 = ax_legend.legend(handles=legend_main.legendHandles[:8], labels=[text.get_text() for text in legend_main.get_texts()][:8], loc='center', fontsize='x-large')
#legend_legend2 = ax_legend.legend(handles=legend_main.legendHandles[7:], labels=[text.get_text() for text in legend_main.get_texts()][7:], loc='center', fontsize='x-large')

fig_legend.subplots_adjust(top=0.8, bottom=0.2, left=0.2, right=0.8)
plt.show()



    


