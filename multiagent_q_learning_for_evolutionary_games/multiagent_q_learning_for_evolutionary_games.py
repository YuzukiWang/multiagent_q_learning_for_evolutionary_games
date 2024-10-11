# Configuration section
POPULATION_SIZE = 49 # How many AIs in the population
MENTOR_INSTANCES = 7 # How many instances of each defined strategy there are
MENTOR_TYPES = 7 # How many types of mentors in the population
EPISODE_LENGTH = 20 # How many turns to play
reward_weighting_factor = 0.8 # During vs. ending reward (theta)
training_time =  5 # How long to train in seconds per agent
TESTING_EPISODES = 1000 # How many episodes to play during the testing phase
alpha = 0.2 # Learning rate
gamma = 0.5 # Discount factor

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
import json
from collections import OrderedDict
import numpy as np
import pickle

# Human agents pick which action to perform
class AgentHuman:
    def pick_action(self, state):
        action = -1

        # Print the given state
        print("State: " + str(state) + " (" + str(len(state)) + "/" + str(EPISODE_LENGTH) + ")")

        # Repeat until valid input provided
        while action not in [0, 1]:
            try:
                # Parse human's chosen action
                action = int(input("Choose Cooperate/Defect (0/1): "))
            except ValueError:
                # Prompt human for valid input
                print("Please input a number.")
        
        return action

    def reward_action(self, state, action, reward):
        pass

# Q agents learn the best action to perform for every state encountered
class AgentQ:
    def __init__(self, memory):
        self.wins = 0 # Number of times agent has won an episode
        self.losses = 0 # Number of times agent has lost an episode
        self.Q = {} # Stores the quality of each action in relation to each state
        self.memory = memory # The number of previous states the agent can factor into its decision
        self.epsilon_counter = 1 # Inversely related to learning rate
        self.delta=[] #
        self.avgdelta_list=[] #

    def get_q(self, state):
        quality1 = self.Q[str(state[-self.memory:])][0]
        quality2 = self.Q[str(state[-self.memory:])][1]

        return quality1, quality2

    def set_q(self, state, quality1, quality2):
        self.Q[str(state[-self.memory:])][0] = quality1
        self.Q[str(state[-self.memory:])][1] = quality2

    def normalize_q(self, state):
        quality1, quality2 = self.get_q(state)

        normalization = min(quality1, quality2)

        self.set_q(state, (quality1 - normalization) * 0.95, (quality2 - normalization) * 0.95)

    def max_q(self, state):
        #quality1, quality2 = self.get_q(state)
        if state==[]:
            quality1,quality2=0,0
        else:
            quality1, quality2 = self.get_q(state)
        if quality1 == quality2 or random.random() < (500 / self.epsilon_counter):
            return random.randint(0, 1)
        elif quality1 > quality2:
            return 0
        else:
            return 1

    def pick_action(self, state):
        # Decrease learning rate
        self.epsilon_counter += 1
        
        if self.epsilon_counter % 1000 == 1:
            self.avgdelta_list.append(self.delta_average()) 

        # If the given state was never previously encountered
        if str(state[-self.memory:]) not in self.Q:
            # Initialize it with zeros
            self.Q[str(state[-self.memory:])] = [0, 0]
    
        return self.max_q(state)

    def reward_action(self, state, action, reward):
        # Increase the quality of the given action at the given state
        # self.Q[str(state[-self.memory:])][action] += reward
        # oldq=self.getQ(self.oldstate,self.oldaction)
        # maxqnew=max([self.getQ(newstate,a) for a in self.actions])
        # self.setQ(self.oldstate,self.oldaction,oldq+self.alpha*(reward+self.lambd*maxqnew-oldq))
        
        oldstate = state[-self.memory -1 : -1]  
        newstate = state[-self.memory : len(state)] 
        oldq = self.Q[str(oldstate)][action]
        maxqnew = max( self.Q[str(newstate)]   )
        self.Q[str(oldstate)][action]=oldq +  alpha*(reward + gamma*maxqnew - oldq) 
        
        self.delta.append(reward + gamma*maxqnew - oldq)
        if len(self.delta)>1000:
            del self.delta[0]
        



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
        
    def delta_average(self):
        return sum(self.delta) / len(self.delta) 
    
    def get_avgdelta_list(self):
        return self.avgdelta_list

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
                return random.randint(0, 1) 
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

# Create a random AI with a random amount of memory
for i in range(POPULATION_SIZE):
    population.append(AgentQ(random.randint(2,2)))

# Create instances of defined strategies
for i in range(MENTOR_TYPES): # Number of defined strategies
    for j in range(MENTOR_INSTANCES):
        mentors.append(AgentDefined(i))

# Training time initialization
start_time = time()
remaining_time = training_time * POPULATION_SIZE
last_remaining_time = int(remaining_time)
total_training_time = training_time * POPULATION_SIZE

# Training mode with AIs
while remaining_time > 0:
    # Calculate remaining training time
    remaining_time = start_time + total_training_time - time()

    # Things to be done every second
    if 0 <= remaining_time < last_remaining_time:
        # Alert user to remaining time
        progress = 100 * (total_training_time - remaining_time) / total_training_time
        sys.stdout.write('\rTraining [{0}] {1}%'.format(('#' * int(progress / 5)).ljust(19), int(min(100, progress + 5))))
        sys.stdout.flush()
        last_remaining_time = int(remaining_time * 2) / float(2)

        # Analyse population
        if time() > start_time + 0.5:
            time_step = []
            for agent in population:
                time_step.append(agent.analyse())
                agent.reset_analysis()
            population_analysis.append(time_step)

        # TODO: Analyse mentors

    state1 = [] # State visible to player 1 (actions of player 2)
    state2 = [] # State visible to player 2 (actions of player 1)

    # Pick a random member of the population to serve as player 1
    player1 = random.choice(population)

    # Pick a random member of the population or a defined strategy to serve as player 2
    player2 = random.choice(population + mentors)

    for i in range(EPISODE_LENGTH):
        action = None

        action1 = player1.pick_action(state1) # Select action for player 1
        action2 = player2.pick_action(state2) # Select action for player 2

        state1.append([action2,action1]) # Log action of player 2 for player 1
        state2.append([action1,action2]) # Log action of player 1 for player 2
        
        # state1.append(action2) # Log action of player 2 for player 1  for only memory opposite
        # state2.append(action1) # Log action of player 1 for player 2

    # Stores the total reward over all games in an episode
    total_reward1 = 0
    total_reward2 = 0
    reward1 = [0]*EPISODE_LENGTH # Total reward due to the actions of player 1 in the entire episode
    reward2 = [0]*EPISODE_LENGTH # Total reward due to the actions of player 2 in the entire episode

    for i in range(EPISODE_LENGTH):
        action1 = state2[i][0]
        action2 = state1[i][0]
        # action1 = state2[i]  #for only memory opposite
        # action2 = state1[i]



        # Calculate rewards for each player
        if action1 == 0 and action2 == 0: # Both players cooperate
            reward1[i] = reward_matrix[0][0][0]
            reward2[i] = reward_matrix[0][0][1]
        elif action1 == 0 and action2 == 1: # Only player 2 defects
            reward1[i] = reward_matrix[0][1][0]
            reward2[i] = reward_matrix[0][1][1]
        elif action1 == 1 and action2 == 0: # Only player 1 defects
            reward1[i] = reward_matrix[0][2][0]
            reward2[i] = reward_matrix[0][2][1]
        elif action1 == 1 and action2 == 1: # Both players defect
            reward1[i] = reward_matrix[0][3][0]
            reward2[i] = reward_matrix[0][3][1]

        total_reward1 += reward1[i]
        total_reward2 += reward2[i]

        # player1.reward_action(state1[:i], action1, reward1 * reward_weighting_factor) # Assign reward to action of player 1
        # player2.reward_action(state2[:i], action2, reward2 * reward_weighting_factor) # Assign reward to action of player 2

    # Assign reward for winning player
    if total_reward1 > total_reward2:
        reward_chunk = total_reward1 / EPISODE_LENGTH * (1 - reward_weighting_factor)

        for i in range(EPISODE_LENGTH-1):  
            # action1 = state2[i]
            action1 = state2[i][0]

            player1.reward_action(state1[:i+1], action1, reward_chunk + reward1[i] * reward_weighting_factor) 
            
            action2 = state1[i][0]
            
            player2.reward_action(state2[:i+1], action2, reward2[i] * reward_weighting_factor)

            player1.mark_victory()
            player2.mark_defeat()
            
    elif total_reward2 > total_reward1:
        reward_chunk = total_reward2 / EPISODE_LENGTH * (1 - reward_weighting_factor)

        for i in range(EPISODE_LENGTH-1):
            # action2 = state1[i]
            action2 = state1[i][0]

            player2.reward_action(state2[:i+1], action2, reward_chunk + reward2[i] * reward_weighting_factor)
            
            action1 = state2[i][0]

            player1.reward_action(state1[:i+1], action1, reward1[i] * reward_weighting_factor) 

            player1.mark_victory()
            player2.mark_defeat()
    
    elif total_reward1 == total_reward2:
        reward_chunk = total_reward1 / EPISODE_LENGTH * (1 - reward_weighting_factor)

        for i in range(EPISODE_LENGTH-1):
            # action1 = state2[i]
            action1 = state2[i][0]

            player1.reward_action(state1[:i+1], action1, reward_chunk + reward1[i] * reward_weighting_factor) 
            
            action2 = state1[i][0]
            
            player2.reward_action(state2[:i+1], action2, reward_chunk + reward2[i] * reward_weighting_factor)
            


# Start new line
print("")
Qtable_all=[]
for Agent in population: 
    Qtable_all.append(sorted(Agent.Q.items()))  # output as sorted
# with open('Q_table_0714_01.txt', 'w') as f:
#    for d in Qtable_all:
#        f.write(json.dumps(d) + '\n')
from datetime import datetime
current_time = datetime.now().strftime("%Y%m%d_%H%M")
with open(f'Q_table_{current_time}.txt', 'w') as f:
    for d in Qtable_all:
        f.write(json.dumps(d) + '\n')





#%%
# Testing mode
wins1 = 0
wins2 = 0
tie = 0
Nc_agent=0
Nc_mentor=0
Nd_agent=0
Nd_mentor=0

for i in range(TESTING_EPISODES):
    state1 = [] # State visible to player 1 (actions of player 2)
    state2 = [] # State visible to player 2 (actions of player 1)

    # Use a human to serve as player 1
    player1 = random.choice(population)

    # Use a random AI to serve as player 2
    player2 = random.choice(mentors)

    for i in range(EPISODE_LENGTH):
        action1 = player1.pick_action(state1) # Allow player 1 to pick action
        action2 = player2.pick_action(state2) # Select action for player 2

        # state1.append(action2) # Log action of player 2 for player 1
        # state2.append(action1) # Log action of player 1 for player 2
        state1.append([action2,action1]) # Log action of player 2 for player 1
        state2.append([action1,action2]) # Log action of player 1 for player 2

    total_reward1 = 0
    total_reward2 = 0

    for i in range(EPISODE_LENGTH):
        action1 = state2[i][0]
        action2 = state1[i][0]

        reward1 = 0 # Total reward due to the actions of player 1 in the entire episode
        reward2 = 0 # Total reward due to the actions of player 2 in the entire episode

        # Calculate rewards for each player
        if action1 == 0 and action2 == 0: # Both players cooperate
            reward1 = reward_matrix[0][0][0]
            reward2 = reward_matrix[0][0][1]
            Nc_agent+=1
            Nc_mentor+=1
        elif action1 == 0 and action2 == 1: # Only player 2 defects
            reward1 = reward_matrix[0][1][0]
            reward2 = reward_matrix[0][1][1]
            Nc_agent+=1
            Nd_mentor+=1
        elif action1 == 1 and action2 == 0: # Only player 1 defects
            reward1 = reward_matrix[0][2][0]
            reward2 = reward_matrix[0][2][1]
            Nd_agent+=1
            Nc_mentor+=1
        elif action1 == 1 and action2 == 1: # Both players defect
            reward1 = reward_matrix[0][3][0]
            reward2 = reward_matrix[0][3][1]
            Nd_agent+=1
            Nd_mentor+=1

        total_reward1 += reward1
        total_reward2 += reward2

    # Print the winning player and score
    print("Score: " + str(round(total_reward1,1)) + " to " + str(round(total_reward2,1)) + "      " +player2.strategy_print())
    if total_reward1 > total_reward2:
        #print("Player 1 wins!")
        wins1 += 1
    elif total_reward2 > total_reward1:
        #print("Player 2 wins!")
        wins2 += 1
    else:
        tie += 1
        #print("Tie!")
    print(state2,"\n",state1)
    
    
print("Player 1 won " + str(wins1) + " times")
print("Player 2 won " + str(wins2) + " times")
print("tie          " + str(tie) + " times")
for agent in population:
    print(agent.epsilon_counter)
    break
print("phoC_agent: ", Nc_agent/TESTING_EPISODES/EPISODE_LENGTH, "  phoC_mentor: ", Nc_mentor/TESTING_EPISODES/EPISODE_LENGTH)


during=population[0].get_avgdelta_list()
x=range(len(during))
y=during
fig=plt.plot(x,y)
plt.show()

