"""
MDP - Markov Decision Process class for the Soarer-Drifter system
Author: John Jackson and Alex Hirst
"""

import numpy as np
import math
import random


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
        self.transition_probability = np.array([[.9, .05, .05],
                                                [.05, .9, .05],
                                                [.05, .05, .9]])

        # sUAS Parameters
        self.x0 = np.array([5, 0])  # Starting cell of the sUAS
        self.x_old = self.x0
        self.state_history = [np.copy(self.x0)]

        self.current_state = [self.x0, 1]  # initialize location, num. balloons
        self.planning_horizon = None

        self.xgoals = None
        self.ygoals = None
        self.num_goals = None  # number of balloons onboard
        self.num_releases = 0  # number of balloons released by suas - initialized at 0

        self.balloon_dt = None
        self.initial_state = None
        self.state_history = []
        self.release_traj = {}

        self.balloon_reward_dict = {}  # stores balloon rewards for already explored release points.
        self.balloon_stats_dict = {}  # stores balloon statistics
        self.num_unique_queries = 0

        #N,Q,T for MCTS
        self.N = {}  # can carry these over from last search if we want.
        self.Q = {}
        self.T = []

    def set_goals(self, xgoal_list, ygoal_list):
        # sets the goals for the balloon
        self.xgoals = xgoal_list
        self.ygoals = ygoal_list
        self.num_goals = len(xgoal_list)  # assume num balloons = num goals

    def take_action(self, state, action):

        # NON- DETERMINISTIC
        s = state
        A = action
        transition_row = self.transition_probability[A[0]]
        result = np.random.multinomial(1, transition_row)
        control_result = np.argmax(result)

        if A:
            if A[1] == 1:  # if we release a balloon at the next step

                B = self.field
                self.x_release = s[0] + 1
                self.y_release = s[1] + (-1*(control_result-1))

                ygoal = self.ygoals[self.num_goals - s[2]]
                B.prop_balloon(s[0] + 1, s[1] + (-1*(control_result-1)), ygoal)

                self.release_traj[self.num_releases] = [B.position_history_x_samps,
                                                        B.position_history_y_samps,
                                                        B.position_history_x_orig,
                                                        B.position_history_y_orig]
                self.num_releases += 1  # add count for number of releases
        return ([s[0] + 1, s[1] + (-1*(control_result-1)), s[2] - A[1]])



    def calculate_reward(self, s, action):
        ''' Calculates and returns EXPECTED reward for FORWARD Search
        OVERALL RUNTIME (results may vary):
        before implementing dict: ~40 secs
        after implementing dict: ~9 secs
        '''
        total_reward = 0
        balloon_action = action[1]
        suas_action = action[0]

        T = self.transition_probability[suas_action]

        for i in range(len(T)):
            t = T[i]
            suas_action = i
            x = s[0] + 1
            y = s[1] + (-1*(suas_action-1))
            bal = s[2]

            if balloon_action > 0:
                balloon_reward = self.calc_balloon_reward(x, y, bal)
            else:
                balloon_reward = 0.

            uas_reward = self.calc_uas_reward(suas_action, y)

            local_reward = t*(balloon_reward + uas_reward)
            total_reward = total_reward + local_reward
        return total_reward # return average reward


    def calc_balloon_reward(self, x, y, bal):
        # function to calculate the balloon reward
        # x,y position OF RELEASE
        # bal = number of balloons on uas at time of release
        goal_index = self.num_goals - bal
        if (x, y, bal) in self.balloon_reward_dict:
            balloon_reward = self.balloon_reward_dict[(x, y, bal)]
        else:
            self.num_unique_queries = self.num_unique_queries + 1
            self.field.prop_balloon(x, y, self.ygoals[goal_index])
            [mu, std] = self.field.calc_util()
            self.balloon_stats_dict[(x, y)] = [mu, std]

            if abs(mu - self.xgoals[goal_index]) > 6:
                balloon_reward = - 100 * abs(mu - self.xgoals[goal_index])
                self.balloon_reward_dict[(x, y, bal)] = balloon_reward
            else:
                balloon_reward = 500./abs(mu - self.xgoals[goal_index]) - 2.5*std
                self.balloon_reward_dict[(x, y, bal)] = balloon_reward
        return balloon_reward

    def balloon_statistics(self, x, y):
        # function to get the statistics of a balloon number bal released at state
        # x, y. returns [mu, std]
        if (x, y) in self.balloon_stats_dict:
            return self.balloon_stats_dict[(x, y)]

    def calc_uas_reward(self, suas_action, y):
        # Control action costs
        suas_control_cost = 0
        # if suas_action == 0:
        #     suas_control_cost = -.1
        # if suas_action == 1:
        #     suas_control_cost = 0.
        # if suas_action == 2:
        #     suas_control_cost = .1

        if (y <= self.ymax) and (y >= self.ymin):
            suas_position_cost = -0.1*abs(y - (self.ymax - self.ymin)/2)
        else:
            suas_position_cost = -10000 # UUGE cost for going outside of bounds

        return suas_control_cost + suas_position_cost


    def reset_mdp(self):
        self.x_old = self.x0
        self.state_history = [np.copy(self.x0)]


    def import_windfield(self, field):
        self.field = field
        self.field.dt = self.balloon_dt  # set balloon propagation timestep length
        #self.field.y_goal = self.ygoal  # set balloon propagation ygoal


    def import_actionspace(self, x, y):
        # self.actionspace = actionspace future use
        self.xmin = x[0]
        self.xmax = x[1]
        self.ymin = y[0]
        self.ymax = y[1]


    def getactionspace(self, s):
        A_s = []  # empty list of actions

        brange = [0, 1] # balloon actions
        yrange = [0, 1, 2] # control actions

        # How to define action set?
        if s[2] <= 0:
            brange = [0]
        # if s[1] == self.ymax:
        #     yrange = [1, 2]
        # if s[1] > self.ymax:
        #     yrange = [2]
        # if s[1] == self.ymin:
        #     yrange = [0, 1]
        # if s[1] < self.ymin:
        #     yrange = [0]

        for y in yrange:
            for b in brange:
                A_s.append([y, b])

        if s[0] >= self.xmax:
            A_s = []

        return A_s


    def getstatespace(self, s, A):
        S_s = []
        S_s.append([s[0] + 1, s[1] + (-1*(A[0]-1)), s[2] - A[1]])

        return S_s


    def selectaction(self, s, d):
        # Forward search algorithm... over finite horizon d
        gamma = 1  # tuning parameter
        if d == 0:
            return (None, 0)

        a_opt = None
        v_opt = -math.inf  # initialize optimal a and v

        A_s = self.getactionspace(s)  # get action space for state s
        for a in A_s:  # for every action in action space
            v = self.calculate_reward(s, a)  # get reward of doing action a at state s
            # forget about transition prob. for now
            S_s = self.getstatespace(s, a)  # get the space of states from doing action a at state s
            for sp in S_s:  # for every potential resulting state in state s
                [ap, vp] = self.selectaction(sp, d-1)
                v = v + gamma*vp # DETERMINISTIC!!!

            if v > v_opt:
                a_opt = a
                v_opt = v
        return [a_opt, v_opt]


    def selectaction_SPARCE(self, s, d, n, gamma):
        # sparce sampling method. Does n random searches for each tree.
        # n = number of searches
        # gamma is a tuning parameter

        if d == 0:
            return (None, 0)
        a_s, v_s = None, -math.inf
        A = self.getactionspace(s)
        for a in A:
            v = 0
            for i in range(n):
                [s_p, r] = self.generative_model(s, a)
                [a_p, v_p] = self.selectaction_SPARCE(s_p, d-1, n, gamma)
                v = v + (r + gamma*v_p)/n
            if v > v_s:
                a_s, v_s = a, v
        return [a_s, v_s]


# MONTE CARLO TREE SEARCH (MCTS)

    def selectaction_MCTS(self, s, d, c, gamma, n):
        # re-initialize N and Q Dictionaries
        self.N = {}  # can carry these over from last search if we want.
        self.Q = {}
        self.T = []

        self.c = c
        self.gamma = gamma

        # d = planning Horizon
        # s = state
        #n = 50, c = 100, gamma = 1 works
        for i in range(n):  # loop iterations? unsure how this works
            self.SIMULATE(s, d)

        # pick a with max Q value at s
        a = max(self.Q[tuple(s)], key=self.Q[tuple(s)].get)

        if a[1] == 1:
            print(a)
            print(self.Q[tuple(s)])
            print(self.Q)
        return list(a)


    def SIMULATE(self, s, d):
        c = self.c  # TUNING parameter for exploration!!!
        N = self.N
        Q = self.Q
        T = self.T
        gamma = self.gamma

        if d == 0:
            return 0
        s = tuple(s) # does this have to be changed back to list?
        if s not in N: # if state dict doesn't exist yet, then add it.
            N[s] = {}
            Q[s] = {}

        if s not in T:
            A_s = self.getactionspace_MCTS(s)  # get action space for state s
            for a in A_s:
                    a = tuple(a)
                    N[s][a] = 0 # update number of times visited this state
                    Q[s][a] = 0 # update state - value function
            self.T.append(s)
            return self.ROLLOUT(s,d)

        N = self.N
        Q = self.Q
        N_s = sum(N[s].values())
        Q_sa = Q[s]
        N_sa = N[s]
        value_dict = {}
        for action, q in Q_sa.items():
            n = N_sa[action]
            if n == 0:
                value_dict[action] = math.inf
            else:
                # is q/n?
                value_dict[action] = q + c * math.sqrt(np.log(N_s)/n)

        a = max(value_dict, key=value_dict.get)
        (s_p, r) = self.generative_model(s,a)
        q = r + gamma*self.SIMULATE(s_p, d-1)
        self.N[s][a] = N[s][a] + 1
        self.Q[s][a] = Q[s][a] + (q-Q[s][a])/N[s][a]
        return q


    def ROLLOUT(self, s, d):
        if d == 0:
            return 0
        a = self.ROLLOUT_policy(s)
        sp, r = self.generative_model(s, a)
        return r + self.gamma*self.ROLLOUT(sp, d-1)


    def ROLLOUT_policy(self, s):
        # current rollout policy: pick a completely random action from action space.
        action_space = self.getactionspace_MCTS(s)
        as_length = len(action_space)
        i = np.random.randint(as_length) # pick a totally uniformly random action.
        return action_space[i]

    def getactionspace_MCTS(self, s):
        # check if releasing at the position to right is even feasible, if not, then don't
        A_s = []  # empty list of actions

        x, y, bal = s[0], s[1], s[2]
        #print(x, y, bal)
        if bal > 0:
            _ = self.calc_balloon_reward(x+1, y, bal)
            [mu, std] = self.balloon_statistics(x+1, y)
            goal = self.xgoals[self.num_goals - (bal)]

        brange = [0, 1] # balloon actions
        yrange = [0, 1, 2] # control actions

        # How to define action set?
        if (s[2] <= 0) or (abs(mu - goal)>6):
            brange = [0]
        if s[1] == self.ymax:
            yrange = [2]
        if s[1] > self.ymax:
            yrange = [2]
        if s[1] == self.ymin:
            yrange = [0]
        if s[1] < self.ymin:
            yrange = [0]

        for y in yrange:
            for b in brange:
                A_s.append([y, b])

        if s[0] >= self.xmax:
            A_s = []

        return A_s


    def generative_model(self, s, action):
        # input: state s, action
        # output: state s_p and reward r
        # state: [x, y, # of balloons]

        # All of the information about the state transitions and rewards is
        # represented by G. The state transition probabilities and expected
        # reward functions are not used directly.
        suas_action = action[0]
        balloon_action = action[1]

        k = np.random.uniform(0,1)
        T = self.transition_probability[suas_action]

        if k <= T[0]:
            a = 0
        elif k <= T[1]:
            a = 1
        else:
            a = 2

        x = s[0] + 1
        y = s[1] + (-1*(a-1))
        bal = s[2]

        if balloon_action > 0:
            balloon_reward = self.calc_balloon_reward(x, y, bal)
        else:
            balloon_reward = 0.

        suas_reward = self.calc_uas_reward(suas_action, y)

        state = [x, y, s[2] - action[1]]
        reward = balloon_reward + suas_reward

        return (state, reward)
