"""
Test Script for the MDP
Alex Hirst
"""

import matplotlib
#matplotlib.use('TkAgg') #for macs

import matplotlib.pyplot as plt

from src import mdp
from src import wind_field
from src import gen_random_field
from src import visualizer_mdp
import clear_folder
import time as t


test_steps = 100
n = 10#cell height of field (y)
m = 15 #cell width of field (x)
length = 3 #unit width
n_samples = 50 #number of balloons propagated at each space

field = gen_random_field.field_generator(n ,m , length, 0, 0.15, n_samples, 'Normal')
field.nrm_mean = 0 #Can use matrix here to specify distributions for each measurement
field.nrm_sig = 1 #Can use matrix here to specify distributions for each measurement
field.sample()


A = wind_field.wind_field(field.vel, field.loc, length, field.nsamps, field.samples)
A.xgoal = 18
A.ygoal = 18
#Create MDP object
mdp = mdp.SoarerDrifterMDP(test_steps)
mdp.xgoal = 18
mdp.ygoal = 18
mdp.balloon_dt = 0.5

mdp.import_windfield(A) #import windfield
mdp.import_actionspace([0,42], [0,5]) #import action space [xlimits], [ylimits]


state_log = []
mdp.initial_state = [5,2,1] #initialize state
mdp.state_history.append(mdp.initial_state)
state = [5,2,1]
horizon = 3
for i in range(35):
    [a_opt,v_opt] = mdp.selectaction(state, horizon)
    state = mdp.take_action(state, a_opt)
    mdp.state_history.append(state)


''' Visualize Here!!!! '''

vis = visualizer_mdp.animator_mdp()
vis.save = True
vis.init_live_plot(mdp)

for time in range(vis.total_time):
    t.sleep(0.1)
    vis.measurement_update(time)


input("press enter to create gif")
gif = input('make gif? (y/n)')
if gif == 'y':
    vis.make_gif()

#Clear the figures folder
clear_folder.clear_figure_folder()



print(state_log)
