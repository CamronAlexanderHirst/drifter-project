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
import clear_folder
import time as t


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
m = 20  # # of cells width of field (x)
length = 5  # cell unit width for field estimate
n_samples = 75  # number of balloons propagated at each space


time_start = t.process_time()

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
mdp.set_xgoals([20, 40, 60])
mdp.ygoal = 20
mdp.balloon_dt = 0.25
logger.info('X Goals: {}'.format(mdp.xgoals))
logger.info('Y Goals: {}'.format(mdp.ygoal))
logger.info('Balloon dt: {}'.format(mdp.balloon_dt))
logger.info('generated mdp')

mdp.import_windfield(A)  # import windfield
mdp.import_actionspace([0, 110], [0, 10])  # import action space [xlimits], [ylimits]
logger.info('imported windfield and actionspace')

logger.info('running mdp simulation...')
mdp.initial_state = [5, 0, mdp.num_balloons]  # initialize state
mdp.state_history.append(mdp.initial_state)
state = mdp.initial_state

# planner settings
horizon = 5  # planning horizon
num_actions = 90  # number of actions for MDP to take
solver = 'MCTS'  # either 'Forward', 'Sparce', or 'MCTS'

logger.info('Horizon: {}'.format(horizon))
logger.info('Number of actions: {}'.format(num_actions))
logger.info('Solver: {}'.format(solver))
for i in range(num_actions):
    print(i)
    if solver == 'Forward':
        [a_opt, v_opt] = mdp.selectaction(state, horizon)
    if solver == 'Sparce':
        [a_opt, v_opt] = mdp.selectaction_SPARCE(state, horizon, 1)
    if solver == 'MCTS':
        a_opt = mdp.selectaction_MCTS(state, horizon, 50, .95, 50)
    print("action: ", a_opt)
    print("state: ", state)

    state = mdp.take_action(state, a_opt)
    mdp.state_history.append(state)


time_end = t.process_time()
duration = time_end - time_start

logger.info('Generation and MDP Simulation Completed in {} s'.format(duration))
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
