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
import matplotlib
import numpy as np
import clear_folder
import time as t

np.random.seed(12)


logFormatter = logging.Formatter('%(asctime)s [%(threadName)-12.12s] [%(levelname)-5.5s]  %(message)s')
logger = logging.Logger(logFormatter)
hdlr = logging.FileHandler(datetime.datetime.now().strftime('logs/drifter_mdp_%H_%M_%d_%m_%Y.log'))
logger.addHandler(hdlr)
consoleHandler = logging.StreamHandler()
consoleHandler.setFormatter(logFormatter)
logger.addHandler(consoleHandler)
logger.setLevel(logging.INFO)


test_steps = 100 # currently unused, number of actions to take?
n = 10  # # of cells height of field (y)
m = 25  # # of cells width of field (x)
length = 5  # cell unit width for field estimate
n_samples = 75  # number of balloons propagated at each space


time_start = t.process_time()

# below: 0.25 for most results..
field = gen_random_field.field_generator(n, m, length, 0, 0.25, n_samples, 'Normal')
field.nrm_mean = 0  # Can use matrix here to specify distributions for each measurement
field.nrm_sig = 1

# speed statistics
field.nrm_mean_s = 0
field.nrm_sig_s = 0.25
# heading statistics
field.nrm_mean_h = 0
field.nrm_sig_h = 0.25
field.sample_heading_speed() # sample n_samples fields with heading and speed stats
logger.info('generated field')

A = wind_field.wind_field(field.vel, field.loc, length, field.nsamps, field.samples)
# A.xgoal = 18 # Not currently used?
# A.y_goal = 18

# Create MDP object
mdp = mdp.SoarerDrifterMDP(test_steps)
x_goals = [20, 40, 60]  # x position of goals
y_goals = [30, 40, 30]  # y position of goals
mdp.set_goals(x_goals, y_goals)
mdp.balloon_dt = 0.25
logger.info('X Goals: {}'.format(mdp.xgoals))
logger.info('Y Goals: {}'.format(mdp.ygoals))
logger.info('Balloon dt: {}'.format(mdp.balloon_dt))
logger.info('generated mdp')

mdp.import_windfield(A)  # import windfield
xlimits = [0, 115]
ylimits = [0, 10]
mdp.import_actionspace(xlimits, ylimits)  # import action space [xlimits], [ylimits]
logger.info('imported windfield and actionspace')

logger.info('running mdp simulation...')
mdp.initial_state = [10, 0, mdp.num_goals]  # initialize state
mdp.state_history.append(mdp.initial_state)
state = mdp.initial_state

# planner settings
horizon = 5  # planning horizon
num_actions = 100  # number of actions for MDP to take

logger.info('Horizon: {}'.format(horizon))
logger.info('Number of actions: {}'.format(num_actions))



# setup variables to store the metrics:
reward = 0  # store reward
b = 0  # balloon index
comp_time_b = []  # balloon computation time list
balloon_stats = []  # balloon final location statistics list
bd_avg = 0 # mean distance to goal
num_leave_ws = 0  # number of times A/C leaves ws

solver = 'MCTS'  # either 'Forward', 'Sparce', or 'MCTS'
logger.info('Solver: {}'.format(solver))

for i in range(num_actions):
    print(i)
    if solver == 'Forward':
        [a_opt, v_opt] = mdp.selectaction(state, horizon)
    if solver == 'Sparce':
        [a_opt, v_opt] = mdp.selectaction_SPARCE(state, horizon, 1, .95)
    if solver == 'MCTS':
        a_opt = mdp.selectaction_MCTS(state, horizon, 500, .9, 200) # c, gamma, n
    print("action: ", a_opt)
    print("state: ", state)

    if (abs(state[0] - 20) < 5)  or (abs(state[0] - 40) < 5):
        if solver == 'MCTS':
            print(state)
            print(mdp.Q[tuple(state)])
        # for key, value in mdp.Q.items() :
        #     print(key)
        #     print(value)

    state = mdp.take_action(state, a_opt)
    mdp.state_history.append(state)

    # Calculate all of the metrics!
    if a_opt[1] == 1:
        # balloon comp time:
        time_end = t.process_time()
        comp_time_b.append(time_end - time_start)

        # balloon rewards
        br = mdp.calc_balloon_reward(state[0], state[1], state[2]+1) #balloon reward

        # balloon statistics
        bs = mdp.balloon_statistics(state[0], state[1]) # returns [mu, std]
        balloon_stats.append(bs) # final location stats

        goal_location = mdp.xgoals[mdp.num_goals - (state[2] + 1)]
        avg_dist = abs(goal_location - bs[0])/mdp.num_goals
        bd_avg = bd_avg + avg_dist

    else:
        br = 0

    if (state[1] < ylimits[0]) or (state[1] > ylimits[1]):
        num_leave_ws = num_leave_ws + 1

    ur = mdp.calc_uas_reward(a_opt[0], state[1]) #uas reward
    ar = ur + br # action reward
    reward = reward + ar



time_end = t.process_time()
duration = time_end - time_start

logger.info('Generation and MDP Simulation Completed in {} s'.format(duration))

# Metrics!
# can be commented out to reduce clutter.
logger.info('Solver: {}'.format(solver))
logger.info('reward earned: {}'.format(reward))
logger.info('total computation time: {}'.format(duration))
logger.info('computation time for balloons: {}'.format(comp_time_b))
logger.info('number of unique balloon value queries: {}'.format(mdp.num_unique_queries))
logger.info('balloons final location stats [mu, std]: {}'.format(balloon_stats))
logger.info('average distance to goals: {}'.format(bd_avg))
logger.info('number of times A/C leaves workspace: {}'.format(num_leave_ws))


logger.info('done')

''' Visualize Here!!!! '''

logger.info('visualizing...')
vis = visualizer.animator_mdp()
vis.save = True
vis.init_live_plot(mdp)

for time in range(vis.total_time):
    #t.sleep(0.1)
    vis.measurement_update(time)

gif = input('make gif? (y/n)')
if gif == 'y':
    vis.make_gif()

# Clear the figures folder
clear_folder.clear_figure_folder()
