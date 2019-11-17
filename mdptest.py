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


test_steps = 100
n = 10  # # of cells height of field (y)
m = 18  # # of cells width of field (x)
length = 5  # cell unit width
n_samples = 75  # number of balloons propagated at each space


time_start = t.process_time()

field = gen_random_field.field_generator(n, m, length, 0, 0.15, n_samples, 'Normal')
field.nrm_mean = 0  # Can use matrix here to specify distributions for each measurement
field.nrm_sig = 1

field.nrm_mean_s = 0
field.nrm_sig_s = 0.2
field.nrm_mean_h = 0
field.nrm_sig_h = 0.4
field.sample_heading_speed()
logger.info('generated field')

A = wind_field.wind_field(field.vel, field.loc, length, field.nsamps, field.samples)
# A.xgoal = 18 # Not currently used?
# A.ygoal = 18

# Create MDP object
mdp = mdp.SoarerDrifterMDP(test_steps)
mdp.set_xgoals([11, 16, 25])
mdp.ygoal = 18
mdp.balloon_dt = 0.5
logger.info('X Goals: {}'.format(mdp.xgoals))
logger.info('Y Goals: {}'.format(mdp.ygoal))
logger.info('Balloon dt: {}'.format(mdp.balloon_dt))
logger.info('generated mdp')

mdp.import_windfield(A)  # import windfield
mdp.import_actionspace([0, 42], [0, 5])  # import action space [xlimits], [ylimits]
logger.info('imported windfield and actionspace')

logger.info('running mdp simulation...')
mdp.initial_state = [5, 2, mdp.num_balloons]  # initialize state
mdp.state_history.append(mdp.initial_state)
state = mdp.initial_state
horizon = 5
logger.info('Horizon: {}'.format(horizon))
for i in range(30):
    [a_opt, v_opt] = mdp.selectaction(state, horizon)
    #[a_opt, v_opt] = mdp.selectaction_SPARCE(state, horizon, 1)
    #a_opt = mdp.selectaction_MCTS(state, horizon)
    # print(mdp.Q)
    # print(mdp.N)
    # print(mdp.T)
    #print("state at k:", state)
    #print("opt. action at k:", a_opt)
    state = mdp.take_action(state, a_opt)
    #print("state at k+1:", state)

    mdp.state_history.append(state)

# print('Q: ', mdp.Q)
# print('N: ',mdp.N)
# print('T: ',mdp.T)

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
