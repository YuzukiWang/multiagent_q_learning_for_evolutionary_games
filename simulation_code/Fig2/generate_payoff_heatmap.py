# Configuration section
population_size = 1# How many AIs in the population
mentor_instances = 1 # How many instances of each defined strategy there are
mentor_strategy = 7 
mentor_strategy = [0,2,3,4,5,6,7,8,9,10,11,12,13,14]
episode_length = 20 # How many turns to play
testing_episodes = 10 # How many episodes to play during the testing phase


s = 0.05 #selection intensity

# Prisoner's dillema rewards [Player 1 reward, Player 2 reward]

R = 3
S = 0
T = 5
P = 1
reward_matrix = [[[R, R], # Both players cooperate
                [S, T], # Player 1 cooperates, player 2 defects
                [T, S], # Player 1 defects, player 2 cooperates
                [P, P]]] # Both players defect

# Script section
import sys

import random
random.seed()

from time import time
from matplotlib import pyplot as plt
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
        
        
    def get_q(self, state):
        quality1 = self.Q[str(state[-self.memory:])][0]
        quality2 = self.Q[str(state[-self.memory:])][1]

        return quality1, quality2

    def set_q(self, state, quality1, quality2):
        self.Q[str(state[-self.memory:])][0] = quality1
        self.Q[str(state[-self.memory:])][1] = quality2



    def max_q(self, state, iterp):
        #quality1, quality2 = self.get_q(state)
        if state==[]:
            # quality1,quality2=0,0
            if iterp == True:
                return 0
            else:
                return 1
        else:
            quality1, quality2 = self.get_q(state)
        if quality1 == quality2:
            return random.randint(0, 1)
        elif quality1 > quality2:
            return 0
        else:
            return 1

    def pick_action(self, state, iterp):
        # Decrease learning rate   
        return self.max_q(state, iterp)


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
        self.deadlock_counter = 0  #omegaTFT
        
        self.calm_count = 0
        self.punish_count = 0 #gradual

        
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
        elif self.strategy == 7:
            return "ZDExtort2"
        elif self.strategy == 8:
            return "ZDExtort2v2"
        elif self.strategy == 9:
            return "ZDExtort3"
        elif self.strategy == 10:
            return "ZDExtort4"
        elif self.strategy == 11:
            return "ZDGen2"
        elif self.strategy == 12:
            return "ZDGTFT2"
        elif self.strategy == 13:
            return "ZDMischief"
        elif self.strategy == 14:
            return "ZDSet2"
        elif self.strategy == 15:
            return "AllD"
        
    def receive_match_attributes(self,l,phi,s):
        """
        Parameters

        phi, s, l: floats
            Parameter used to compute the four-vector according to the
            parameterization of the strategies below.
        """


        # Check parameters
        s_min = -min((T - l) / (l - S), (l - S) / (T - l))
        if (l < P) or (l > R) or (s > 1) or (s < s_min):
            raise ValueError

        p1 = 1 - phi * (1 - s) * (R - l)
        p2 = 1 - phi * (s * (l - S) + (T - l))
        p3 = phi * ((l - S) + s * (T - l))
        p4 = phi * (1 - s) * (l - P)

        four_vector = [p1, p2, p3, p4]
        return four_vector  

    def pick_action(self, state,iterp):
        if self.strategy == 0: # Tit for tat
            if len(state) == 0: # On the first tern
                if iterp == True:
                    return 0
                else:
                    return 1
            
                # return random.randint(0, 1) # Random
            
            
            else: # Otherwise
                # return state[-1] # for only memory the opposite # Pick the last action of the opponent  
                return state[-1][0] # Pick the last action of the opponent
            
        elif self.strategy == 1: # GTFT
            if len(state) == 0: # On the first tern
            
                if iterp == True:
                    return 0
                else:
                    return 1
                # return random.randint(0, 1) # Random
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
                if iterp == True:
                    return 0
                else:
                    return 1
                # return random.randint(0, 1) # Random
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
                if iterp == True:
                    return 0
                else:
                    return 1
                # return random.randint(0, 1) # Random
            # elif 1 in state: # If the enemy has ever defected
            elif 1 in state[:,0]: # If the enemy has ever defected
                return 1 # Defect
            else: # Otherwise
                return 0 # Cooperate
            
        elif self.strategy == 4: # Fool me once
            state=np.array(state)
            if len(state) == 0: # On the first tern
                if iterp == True:
                    return 0
                else:
                    return 1
                # return random.randint(0, 1) #cooperate
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
                if iterp == True:
                    return 0
                else:
                    return 1
                # return random.randint(0, 1)
            if len(state[:,0]) == 1:
                return state[-1,0]
            if self.deadlock_counter >= self.deadlock_threshold:
                move = 0 
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
                    move = 1 
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
                if iterp == True:
                    return 0
                else:
                    return 1
                # return random.randint(0, 1)

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
        
        elif self.strategy == 7: #"ZDExtort2"
            state=np.array(state)
            
            if len(state) == 0: # On the first tern
                if iterp == True:
                    return 0
                else:
                    return 1
                
            four_vector=self.receive_match_attributes(phi = 1/9, s = 0.5, l = P) 
            # print( "ZDExtort2",four_vector)
            
            if state[-1][1] == 0 and state[-1][0] ==0: #CC
                if random.random()<four_vector[0]:
                    return 0
                else:
                    return 1
            if state[-1][1] == 0 and state[-1][0] ==1: #CD
                if random.random()<four_vector[1]:
                    return 0
                else:
                    return 1
            if state[-1][1] == 1 and state[-1][0] ==0: #DC
                if random.random()<four_vector[2]:
                    return 0
                else:
                    return 1
            if state[-1][1] == 1 and state[-1][0] ==1: #DD
                if random.random()<four_vector[3]:
                    return 0
                else:
                    return 1
                
        elif self.strategy == 8: #"ZDExtort2v2"
            state=np.array(state)
            
            if len(state) == 0: # On the first tern
                if iterp == True:
                    return 0
                else:
                    return 1
                
            four_vector=self.receive_match_attributes(phi = 1/8, s = 0.5, l = 1) 
            # print( "ZDExtort2v2",four_vector)
            
            if state[-1][1] == 0 and state[-1][0] ==0: #CC
                if random.random()<four_vector[0]:
                    return 0
                else:
                    return 1
            if state[-1][1] == 0 and state[-1][0] ==1: #CD
                if random.random()<four_vector[1]:
                    return 0
                else:
                    return 1
            if state[-1][1] == 1 and state[-1][0] ==0: #DC
                if random.random()<four_vector[2]:
                    return 0
                else:
                    return 1
            if state[-1][1] == 1 and state[-1][0] ==1: #DD
                if random.random()<four_vector[3]:
                    return 0
                else:
                    return 1
        
        elif self.strategy == 9: #"ZDExtort3"
            state=np.array(state)
            
            if len(state) == 0: # On the first tern
                if iterp == True:
                    return 0
                else:
                    return 1
                
            four_vector=self.receive_match_attributes(phi = 3 / 26, s = 1 / 3, l = 1) 
            # print( "ZDExtort3",four_vector)
            
            if state[-1][1] == 0 and state[-1][0] ==0: #CC
                if random.random()<four_vector[0]:
                    return 0
                else:
                    return 1
            if state[-1][1] == 0 and state[-1][0] ==1: #CD
                if random.random()<four_vector[1]:
                    return 0
                else:
                    return 1
            if state[-1][1] == 1 and state[-1][0] ==0: #DC
                if random.random()<four_vector[2]:
                    return 0
                else:
                    return 1
            if state[-1][1] == 1 and state[-1][0] ==1: #DD
                if random.random()<four_vector[3]:
                    return 0
                else:
                    return 1
                
        elif self.strategy == 10: #"ZDExtort4"
            state=np.array(state)
            
            if len(state) == 0: # On the first tern
                if iterp == True:
                    return 0
                else:
                    return 1
                
            four_vector=self.receive_match_attributes(phi = 4 / 17, s = 0.25, l = 1) 
            # print( "ZDExtort4", four_vector)
            
            if state[-1][1] == 0 and state[-1][0] ==0: #CC
                if random.random()<four_vector[0]:
                    return 0
                else:
                    return 1
            if state[-1][1] == 0 and state[-1][0] ==1: #CD
                if random.random()<four_vector[1]:
                    return 0
                else:
                    return 1
            if state[-1][1] == 1 and state[-1][0] ==0: #DC
                if random.random()<four_vector[2]:
                    return 0
                else:
                    return 1
            if state[-1][1] == 1 and state[-1][0] ==1: #DD
                if random.random()<four_vector[3]:
                    return 0
                else:
                    return 1
        
        elif self.strategy == 11: #"ZDGen2"
            state=np.array(state)
            
            if len(state) == 0: # On the first tern
                if iterp == True:
                    return 0
                else:
                    return 1
                
            four_vector=self.receive_match_attributes(phi = 1/8, s = 0.5, l = 3) 
            # print( "ZDGen2", four_vector)
            
            if state[-1][1] == 0 and state[-1][0] ==0: #CC
                if random.random()<four_vector[0]:
                    return 0
                else:
                    return 1
            if state[-1][1] == 0 and state[-1][0] ==1: #CD
                if random.random()<four_vector[1]:
                    return 0
                else:
                    return 1
            if state[-1][1] == 1 and state[-1][0] ==0: #DC
                if random.random()<four_vector[2]:
                    return 0
                else:
                    return 1
            if state[-1][1] == 1 and state[-1][0] ==1: #DD
                if random.random()<four_vector[3]:
                    return 0
                else:
                    return 1
                
        elif self.strategy == 12: #"ZDGTFT2"
            state=np.array(state)
            
            if len(state) == 0: # On the first tern
                if iterp == True:
                    return 0
                else:
                    return 1
                
            four_vector=self.receive_match_attributes(phi = 0.25, s = 0.5, l = R) 
            # print("ZDGTFT2", four_vector)
            
            if state[-1][1] == 0 and state[-1][0] ==0: #CC
                if random.random()<four_vector[0]:
                    return 0
                else:
                    return 1
            if state[-1][1] == 0 and state[-1][0] ==1: #CD
                if random.random()<four_vector[1]:
                    return 0
                else:
                    return 1
            if state[-1][1] == 1 and state[-1][0] ==0: #DC
                if random.random()<four_vector[2]:
                    return 0
                else:
                    return 1
            if state[-1][1] == 1 and state[-1][0] ==1: #DD
                if random.random()<four_vector[3]:
                    return 0
                else:
                    return 1
                
        elif self.strategy == 13: #"ZDMischief"
            state=np.array(state)
            
            if len(state) == 0: # On the first tern
                if iterp == True:
                    return 0
                else:
                    return 1
                
            four_vector=self.receive_match_attributes(phi = 0.1, s = 0.0, l = 1) 
            # print("ZDMischief", four_vector)
            
            if state[-1][1] == 0 and state[-1][0] ==0: #CC
                if random.random()<four_vector[0]:
                    return 0
                else:
                    return 1
            if state[-1][1] == 0 and state[-1][0] ==1: #CD
                if random.random()<four_vector[1]:
                    return 0
                else:
                    return 1
            if state[-1][1] == 1 and state[-1][0] ==0: #DC
                if random.random()<four_vector[2]:
                    return 0
                else:
                    return 1
            if state[-1][1] == 1 and state[-1][0] ==1: #DD
                if random.random()<four_vector[3]:
                    return 0
                else:
                    return 1
        
        elif self.strategy == 14: #"ZDSet2"
            state=np.array(state)
            
            if len(state) == 0: # On the first tern
                if iterp == True:
                    return 0
                else:
                    return 1
                
            four_vector=self.receive_match_attributes(phi = 1/4, s = 0.0, l = 2) 
            # print("ZDSet2", four_vector)
            
            if state[-1][1] == 0 and state[-1][0] ==0: #CC
                if random.random()<four_vector[0]:
                    return 0
                else:
                    return 1
            if state[-1][1] == 0 and state[-1][0] ==1: #CD
                if random.random()<four_vector[1]:
                    return 0
                else:
                    return 1
            if state[-1][1] == 1 and state[-1][0] ==0: #DC
                if random.random()<four_vector[2]:
                    return 0
                else:
                    return 1
            if state[-1][1] == 1 and state[-1][0] ==1: #DD
                if random.random()<four_vector[3]:
                    return 0
                else:
                    return 1
        
        elif self.strategy == 15: #ALLD
            return 1
            
        

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

population2 = [] 

# Stores record of analysis of all AIs
population_analysis = []

# Stores all instances of defined strategies
mentors = []
mentors2 = []

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
    
    population2.append(AgentQ(random.randint(2,2)))
    population2[i].Q=strategy00_dict
    # population.append(AgentDefined(6))
    # population2.append(AgentDefined(6))

# Create instances of defined strategies
for i in mentor_strategy: # Number of defined strategies
    for j in range(mentor_instances):
        mentors.append(AgentDefined(i))
        mentors2.append(AgentDefined(i))




# Testing mode



bigpopulation=population+mentors
bigpopulation2=population2+mentors2

# num_strategy = [ [ 0 for _ in range(mentor_strategy + 1)  ] for _ in range(testing_episodes) ] 
max_episode = 1
num_strategy = [ [ 0 for _ in range(11)  ] for _ in range(max_episode) ] 

episode_total = 0

iterp1=False
iterp2=False


test_times=10000

for iepisode in range(max_episode):
    
    # random.shuffle(bigpopulation)
    reward_episode =  [[ 0 for _ in range(len(bigpopulation))] for _ in range(len(bigpopulation))]
    avg_reward = [0 for _ in range(len(bigpopulation))]
    

    

    
    # for iplayer1 in range(len(bigpopulation)):  
    for iplayer1 in range(len(bigpopulation)): 
        # if iterp1==True:
        #     iterp1 = False
        # else:
        #     iterp1 = True  
    
    
        # for iplayer2 in range(iplayer1+1,len(bigpopulation)):
        for iplayer2 in range(len(bigpopulation2)):   
            
            # if iterp2==True:
            #     iterp2 = False
            # else:
            #     iterp2 = True
            for itertimes in range(test_times):
                
                if iterp1 == False and iterp2 == False:
                    iterp1 = True 
                    iterp2 = True
                elif iterp1 == True and iterp2 == True:
                    iterp1 = True 
                    iterp2 = False
                elif iterp1 == True and iterp2 == False:
                    iterp1 = False 
                    iterp2 = True
                elif iterp1 == False and iterp2 == True:
                    iterp1 = False 
                    iterp2 = False
            
            
                player1 = bigpopulation[iplayer1]
                player2 = bigpopulation2[iplayer2]
                state1 = [] # State visible to player 1 (actions of player 2)
                state2 = [] # State visible to player 2 (actions of player 1)
            
                # Use a human to serve as player 1
                #player1 = random.choice(population)
            
                # Use a random AI to serve as player 2
                #player2 = random.choice(mentors)
                
            
                for i in range(episode_length):
                    action1 = player1.pick_action(state1,iterp1) # Allow player 1 to pick action
                    action2 = player2.pick_action(state2,iterp2) # Select action for player 2
            
                    # state1.append(action2) # Log action of player 2 for player 1
                    # state2.append(action1) # Log action of player 1 for player 2
                    state1.append([action2,action1]) # Log action of player 2 for player 1
                    state2.append([action1,action2]) # Log action of player 1 for player 2
            
                total_reward1 = 0
                total_reward2 = 0
            
                for i in range(episode_length):
                    action1 = state2[i][0]
                    action2 = state1[i][0]
            
                    reward1 = 0 # Total reward due to the actions of player 1 in the entire episode
                    reward2 = 0 # Total reward due to the actions of player 2 in the entire episode
            
                    # Calculate rewards for each player
                    if action1 == 0 and action2 == 0: # Both players cooperate
                        reward1 = reward_matrix[0][0][0]
                        reward2 = reward_matrix[0][0][1]
            
                    elif action1 == 0 and action2 == 1: # Only player 2 defects
                        reward1 = reward_matrix[0][1][0]
                        reward2 = reward_matrix[0][1][1]
            
                    elif action1 == 1 and action2 == 0: # Only player 1 defects
                        reward1 = reward_matrix[0][2][0]
                        reward2 = reward_matrix[0][2][1]
            
                    elif action1 == 1 and action2 == 1: # Both players defect
                        reward1 = reward_matrix[0][3][0]
                        reward2 = reward_matrix[0][3][1]
        
                    total_reward1 += reward1
                    total_reward2 += reward2
                reward_episode[iplayer1][iplayer2] += total_reward1/test_times/2  
                reward_episode[iplayer2][iplayer1] += total_reward2/test_times/2  
                
                # print("Score: " + player1.strategy_print()  + "      " + str(round(total_reward1,1)) + " to " + str(round(total_reward2,1)) + "      " +player2.strategy_print())
    
                # print(state2,"\n",state1)
       
    for iavg_reward in range(len(bigpopulation)) : 
        avg_reward[iavg_reward] = sum(reward_episode[iavg_reward]) / (len(bigpopulation)) / episode_length 
        
    reward_episode_np = np.array(reward_episode)
    
    reward_episode = reward_episode_np / 20
       
    # for irefresh in range(len(bigpopulation)): 
    #     opponent_refresh = random.randint(0,len(bigpopulation)-1)
        
    #     Ptrans=1 / (exp(  s *(avg_reward[len(bigpopulation) - 1 - irefresh] - avg_reward[len(bigpopulation) - 1 - opponent_refresh]) )  + 1)
    #     if random.random()<Ptrans:
    #         bigpopulation[len(bigpopulation) - 1 - irefresh] = bigpopulation[len(bigpopulation) - 1 - opponent_refresh]            

    for i_num in range(len(bigpopulation)) :
        if bigpopulation[i_num].strategy_print() == "strategy00" :
            num_strategy[ iepisode][0] +=1
        elif bigpopulation[i_num].strategy_print() == "TFT" :
            num_strategy[ iepisode][1] +=1
        elif bigpopulation[i_num].strategy_print() == "GTFT0.3" :
            num_strategy[ iepisode][2] +=1
        elif bigpopulation[i_num].strategy_print() == "WSLS" :
            num_strategy[ iepisode][3] +=1
        elif bigpopulation[i_num].strategy_print() == "Holds a grudge" :
            num_strategy[ iepisode][4] +=1
        elif bigpopulation[i_num].strategy_print() == "Fool me once" :
            num_strategy[ iepisode][5] +=1
        elif bigpopulation[i_num].strategy_print() == "Omega TFT" :
            num_strategy[ iepisode][6] +=1
        elif bigpopulation[i_num].strategy_print() == "Gradual TFT" :
            num_strategy[ iepisode][7] +=1
        elif bigpopulation[i_num].strategy_print() == "ZDExtort3" :
            num_strategy[ iepisode][10] +=1
     
    episode_total += 1
    
    ifbreak = 0 
    for check_break in range(11):
        if num_strategy[ iepisode][check_break] >= mentor_instances * (len(mentor_strategy) + 1) -2:
            ifbreak = 1024;
            
    # print (episode_total)
    
    if ifbreak == 1024:
        break;
        

# y={}

# for j in range(11):
#     # y[j]=[  i[j] for i in num_strategy ]     
#     y[j]=[  num_strategy[i][j] for i in range(episode_total) ] 
# fig =plt.figure()
# fig.suptitle('evolution of strategy')
# # x=range(testing_episodes)
# x=range(episode_total)
# plt.plot(x,y[0],color='black',linewidth=1.0, linestyle="-",label="strategy00")#
# plt.plot(x,y[1],color='#0000FF',linewidth=1.0, linestyle="-",label="TFT")#
# plt.plot(x,y[2],color='purple',linewidth=1.0, linestyle="-",label="GTFT0.3")#
# plt.plot(x,y[3],color='#00FFFF',linewidth=1.0, linestyle="-",label="WSLS")#
# plt.plot(x,y[4],color='#FF0000',linewidth=1.0, linestyle="-",label="Holds a grudge")#
# plt.plot(x,y[5],color='#00FF80',linewidth=1.0, linestyle="-",label="Fool me once")#
# plt.plot(x,y[6],color='#FFFF00',linewidth=1.0, linestyle="-",label="Omega TFT")#
# plt.plot(x,y[7],color='#FF00FF',linewidth=1.0, linestyle="-",label="Gradual TFT")#
# plt.legend(loc="best")
# # plt.xlim(1,2)
# # plt.ylim(0,1)
# plt.xlabel('episodes')
# plt.ylabel('num of strategy')
# plt.show
    
# for i_print in range(len(bigpopulation)):
#     print (   bigpopulation[i_print].strategy_print()     )
    
    
import seaborn as sns
strategies = [
    "MTBR", "TFT", "WSLS", "Holds a grudge", "Fool me once", "Omega TFT",
    "Gradual TFT", "ZDExtort2", "ZDExtort2v2", "ZDExtort3", "ZDExtort4",
    "ZDGen2", "ZDGTFT2", "ZDMischief", "ZDSet2"
]
plt.rcParams["font.family"] = "Arial"
plt.rcParams["font.size"] = 11
plt.rcParams["axes.linewidth"] = 0.5
plt.rcParams["xtick.major.width"] = 0.5
plt.rcParams["ytick.major.width"] = 0.5


plt.figure(figsize=(12, 10))  
sns.heatmap(reward_episode, xticklabels=strategies, yticklabels=strategies, cmap="YlGnBu", annot=True, fmt=".2f", annot_kws={"size": 9}, linewidths=0, linecolor='black')
plt.title("Payoff Matrix Heatmap", fontsize=12)
plt.xticks(fontsize=11, rotation=45, ha="right")  
plt.yticks(fontsize=11)
plt.tight_layout(pad=2.0)  


# plt.savefig("reward_episode_heatmap.pdf", format='pdf')


plt.show()

    
    


