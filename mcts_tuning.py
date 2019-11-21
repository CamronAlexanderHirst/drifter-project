#!/usr/bin/python3
"""
Test Script for the MDP
Alex Hirst
"""
import logging
import datetime
from src import mdp
from src import wind_field
from src import gen_random_field
from src import visualizer
import matplotlib.pyplot as plt
import clear_folder
import time as t


horizons = [3,5,7,9,11] # planning horizon
cs = [100] # exploration factors
ns = [100] # number of runs

configs = []
for h in horizons:
    for c in cs:
        for n in ns:
            configs.append([h, c, n])

config_stats = {}
for config in configs:
    config_stats[tuple(config)] = []

print("total number of tests: ", len(configs)*10)

for i in range(1):
    test_steps = 100 # currently unused, number of actions to take?
    n = 10  # # of cells height of field (y)
    m = 25  # # of cells width of field (x)
    length = 5  # cell unit width for field estimate
    n_samples = 75  # number of balloons propagated at each space
    field = gen_random_field.field_generator(n, m, length, 0, 0.25, n_samples, 'Normal')

    # speed statistics
    field.nrm_mean_s = 0
    field.nrm_sig_s = 0.25
    # heading statistics
    field.nrm_mean_h = 0
    field.nrm_sig_h = 0.25
    field.sample_heading_speed() # sample n_samples fields with heading and speed stats
    A = wind_field.wind_field(field.vel, field.loc, length, field.nsamps, field.samples)

    print("Test number: ", i)
    for config in configs:

        horizon = config[0]
        c = config[1]
        n = config[2]

        time_start = t.process_time()
        reward = 0


        # Create MDP object
        mdp1 = mdp.SoarerDrifterMDP(test_steps)
        mdp1.set_xgoals([20, 30, 40, 50])
        mdp1.ygoal = 20
        mdp1.balloon_dt = 0.25
        mdp1.import_windfield(A)  # import windfield
        mdp1.import_actionspace([0, 120], [0, 10])  # import action space [xlimits], [ylimits]
        mdp1.initial_state = [10, 0, mdp1.num_balloons]  # initialize state
        mdp1.state_history.append(mdp1.initial_state)
        state = mdp1.initial_state

        # planner settings
        num_actions = 90  # number of actions for MDP to take
        for i in range(num_actions):
            #print(i)

            a_opt = mdp1.selectaction_MCTS(state, horizon, c, .95, n)
            #print("action: ", a_opt)
            #print("state: ", state)

            state = mdp1.take_action(state, a_opt)
            mdp1.state_history.append(state)

            # store rewards
            if a_opt[1] == 1:
                br = mdp1.calc_balloon_reward(state[0], state[1], state[2]+1) #balloon reward
            else:
                br = 0
            ur = mdp1.calc_uas_reward(a_opt[0], state[1]) #uas reward
            ar = ur + br # action reward
            reward = reward + ar


        time_end = t.process_time()
        duration = time_end - time_start
        statistics = [duration, reward]
        config_stats[tuple(config)].append(statistics)

print("config_stats:", config_stats)


# plt.figure()
# for config_stat in config_stats:
#
# plt.
