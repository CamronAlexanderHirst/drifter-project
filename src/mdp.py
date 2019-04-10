"""
MDP - Markov Decision Process class for the Soarer-Drifter system
Author: John Jackson and Alex Hirst
"""

import numpy as np
import math
import matplotlib

matplotlib.use('TkAgg')

import matplotlib.pyplot as plt


class SoarerDrifterMDP:

    def __init__(self, N):

        # TODO: Match discrete size with balloon grid
        # TODO: Currently not used for anything
        self.size = (20, N)   # Size of the discrete world NxM

        # Discretized control space - (sUAS control, balloon control)
        # sUAS: 0->"turn" left, 1->forward, 2->"turn" right
        # balloon: 0->no release, 1->release
        self.control_space = [[0, 0], [1, 0], [2, 0],
                              [0, 1], [1, 1], [2, 1]]

        # TODO: Chosen probabilities are arbitrary.
        self.transition_probability = np.array([[0.75, 0.125, 0.125],
                                                [0.125, 0.75, 0.125],
                                                [0.125, 0.125, 0.75]])

        # sUAS Parameters
        self.x0 = np.array([5, 0])  # Starting cell of the sUAS
        self.x_old = self.x0
        self.state_history = [np.copy(self.x0)]


        self.current_state = [self.x0, 1] #initialize location, num. balloons
        self.planning_horizon = None

        self.xgoal = None
        self.ygoal = None
        self.balloon_dt = None
        self.initial_state = None
        self.state_history = []

        self.balloon_reward_dict = {}

    def take_action(self, state, action):

        '''
        suas_action = action[0]

        # Get the transition row based on the suas action taken
        transition_row = np.copy(self.transition_probability[suas_action, :])

        # Adjust row based on whether or not we are at the edge
        if self.x_old[0] == 0:
            transition_row[0] = 0.
            transition_row = transition_row / np.sum(transition_row)
        elif self.x_old[0] == (self.size[0] - 1):
            transition_row[2] = 0.
            transition_row = transition_row / np.sum(transition_row)

        # Take action here!
        x_new = self.suas_control(transition_row)
        # Calculate the reward!
        reward = self.calculate_reward(x_new, action)
        self.x_old = np.copy(x_new)
        self.state_history.append(np.copy(x_new))

        return reward '''

        s = state
        A = action

        if A:
            if A[1] == 1: #if we release a balloon at the next step
                B = self.field

                self.x_release = s[0] + 1
                self.y_release = s[1] + (-1*(A[0]-1))

                B.prop_balloon(s[0] + 1 , s[1] + (-1*(A[0]-1)))

                self.release_traj = [B.position_history_x_samps, B.position_history_y_samps,
                B.position_history_x_orig, B.position_history_y_orig]


        return ([s[0] + 1 , s[1] + (-1*(A[0]-1)) , s[2] - A[1]])


    def suas_control(self, transition_row):

        result = np.random.multinomial(1, transition_row)
        control_result = np.argmax(result)

        x_new = self.x_old
        x_new[1] += 1

        if control_result == 0:
            x_new[0] -= 1
        elif control_result == 2:
            x_new[0] += 1
        else:
            # Continue level!
            pass

        return x_new

    def calculate_reward(self, s, action):

        ''' Calculates rewards, really should work on tuning this in the future

        OVERALL RUNTIME:
        before implementing dict: ~40 secs
        after implementing dict: ~9 secs
        of course, these results will vary with input
        '''
        balloon_action = action[1]
        suas_action = action[0]

        if balloon_action > 0:

            x = s[0] + 1
            y = s[1] + (-1*(action[0]-1))
            if (x,y) in self.balloon_reward_dict:
                balloon_reward = self.balloon_reward_dict[(x,y)]

            else:
                self.field.prop_balloon(x, y)
                [mu,std] = self.field.calc_util()
                balloon_reward = 100./abs(mu - self.xgoal) - 5.*std
                #TUNE HERE!!!
                self.balloon_reward_dict[(x,y)] = balloon_reward

        else:
            balloon_reward = 0.




        # Control action costs
        suas_control_cost = 0
        if suas_action == 0:
            suas_control_cost = -.1
        if suas_action == 1:
            suas_control_cost = 0.
        if suas_action == 2:
            suas_control_cost = .1

        suas_position_cost = -0.1*s[1]

        total_reward = balloon_reward + suas_control_cost + suas_position_cost

        return total_reward


    def reset_mdp(self):
        self.x_old = self.x0
        self.state_history = [np.copy(self.x0)]

    def import_windfield(self, field):
        self.field = field
        self.field.dt = self.balloon_dt #set balloon propagation timestep length
        self.field.y_goal = self.ygoal #set balloon propagation ygoal

    def import_actionspace(self,x,y):
        #self.actionspace = actionspace future use
        self.xmin = x[0]
        self.xmax = x[1]

        self.ymin = y[0]
        self.ymax = y[1]


    def getactionspace(self, s):
        A_s = [] #empty list of actions

        brange = [0,1]
        yrange = [0,1,2]

        #How to define action set?
        if s[2] <= 0 :
            brange = [0]

        if s[1] == self.ymax:
            yrange = [1,2]
        if s[1] > self.ymax:
            yrange = [2]

        if s[1] == self.ymin:
            yrange = [0,1]
        if s[1] < self.ymin:
            yrange = [0]

        for y in yrange:
            for b in brange:
                A_s.append([y,b])


        if s[0] >= self.xmax:
            A_s = []

        return A_s

    def getstatespace(self, s, A):

        S_s = []
        S_s.append([s[0] + 1 , s[1] + (-1*(A[0]-1)) , s[2] - A[1]])

        return S_s

    def selectaction(self, s, d):
        gamma = 1 #tuning parameter

        if d == 0:
            return (None,0)

        a_opt = None
        v_opt = -math.inf #initialize optimal a and v


        A_s = self.getactionspace(s) #get action space for state s
        #print(A_s)

        for a in A_s: #for every action in action space
            v = self.calculate_reward(s,a) #get reward of doing action a at state s
            #forget about transition prob. for now
            S_s = self.getstatespace(s, a) #get the space of states from doing action a at state s
            #print(S_s)
            #print(a)
            for sp in S_s: #for every potential resulting state in state s
                #print(a)
                #print(sp)
                [ap, vp] = self.selectaction(sp, d-1)
                v = v + gamma*vp

            if v > v_opt:
                a_opt = a
                v_opt = v

        return [a_opt,v_opt]
