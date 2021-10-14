import numpy as np
import pandas as pd

##--- support functions----##

### 1. function to genereate next step based on probability 


# L--> left ; R-->right, U-->up, T--> terminal

policy = np.array([["T","L","L","L"],["U","U","D","D"],["U","U","D","D"],["U","R","R","T"]])
grid_shape = policy.shape
states = np.array([[0,1,2,3],[4,5,6,7],[8,9,10,11],[12,13,14,15]])
val_coord_dict = {}
for idx,row in enumerate(states):
    for idx1,val in enumerate(row):
        val_coord_dict[val] = (idx,idx1)
coord_val_dict = dict(zip(val_coord_dict.values(),val_coord_dict.keys()))

def get_neigbouring_state(query_state):
    """
    return clockwise starting from the optimal policy
    """
    # left up right bottom
    x,y = coords = val_coord_dict[query_state]
    neigbouring_state = [(x-1,y),(x,y+1),(x+1,y),(x,y-1)]
    if x-1<0:
        neigbouring_state[0] = (x,y)
    if y-1<0:
        neigbouring_state[1] = (x,y)
    if x+1>grid_shape[1]:
        neigbouring_state[2] = (x,y)
    if y+1>grid_shape[0]:
        neigbouring_state[3] = (x,y)
    # neigbouring_state = [ query_state if i<0 else i for i in neigbouring_state]
    neigbouring_state = [coord_val_dict[i] for i in neigbouring_state]
    print("neighbourin states  are {}".format(neigbouring_state))
    optimal_action = policy[coords]
    print("optimal action is {}".format(optimal_action))
    if optimal_action=="L":
        return neigbouring_state
    elif optimal_action=="U":
        return neigbouring_state[1:]+neigbouring_state[0]
    elif optimal_action=="R":
        return neigbouring_state[2:]+neigbouring_state[:2]
    else:
        return neigbouring_state[3:]+neigbouring_state[:3]

    

def get_next_state(current_state):
    if current_state in [5,6,9,10]:
        prob_bins = [.7,.8,.9,1]
        idx = np.digitize(np.random.rand(),prob_bins)
        if current_state==5:#up
            neigbouring_state = [6,1,4,9]
            return neigbouring_state[idx]
        elif current_state==9:#up
            neigbouring_state = [10,5,8,13]
            return neigbouring_state[idx]
        elif current_state==6:#down
            neigbouring_state = [10,5,2,7]
            return neigbouring_state[idx]
        else: #current_state== 10:#down
            neigbouring_state = [14,9,6,11]
            return neigbouring_state[idx] 
    elif current_state in [1,2,4,7,8,11,13,14]:
        prob_bins = [.7,.8,.9,1]
        idx = np.digitize(np.random.rand(),prob_bins)
        if current_state==1:
            neigbouring_state = [0,1,2,5]
            return neigbouring_state[idx] 
        elif current_state==2:
            neigbouring_state=[1,2,3,6]
            return neigbouring_state[idx]






        


class probability_table():
    def __init__(self):
        self.table_dict = {}
    def create_entry(self,next_state,current_state,direction,prob):
        self.table_dict[(next_state,current_state,direction)] = prob



def get_value_next_state(current_state,action):
    if action=="U":
        next_state = current_state-grid_shape[0]
    elif action=="D":
        next_state = current_state+grid_shape[0]
    elif action=="R":
        next_state = current_state+1
    else:
        next_state = current_state-1
    
    if next_state<0:
        next_state = current_state
    return next_state
    
## for interior state [5,6,9,10]  
# the next state is remain interna 
def get_next_state_1(current_state,action):
    rand_prob = np.random.rand()
    
    if rand_prob<.7:


    

def get_next_state_action(current_state):
    """
    current_state : it is the value between 0 to 15
    """
    (x,y) = current_coord = val_coord_dict[current_state]
    action = policy[current_coord]

    if current_state in [0,15]:
        return None
    elif current_state in [5,6,9,10]:
        gen_prob = np.random.rand()
            if gen_prob<.7:
                next_state = 
            








def generate_episode():
    #exploratory start on non terminal state
    S = []
    state_0 = np.random.choice(list(range(1,15)))
    S.append(state_0) 
    while True:


