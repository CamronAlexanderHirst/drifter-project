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
from src import visualizer


test_steps = 100
n = 10#cell height of field (y)
m = 15 #cell width of field (x)
length = 3 #unit width
n_samples = 25 #number of balloons propagated at each space

field = gen_random_field.field_generator(n ,m , length, 0, 0.15, n_samples, 'Normal')
field.nrm_mean = 0 #Can use matrix here to specify distributions for each measurement
field.nrm_sig = 1 #Can use matrix here to specify distributions for each measurement
field.sample()

A = wind_field.wind_field(field.vel, field.loc, length, field.nsamps, field.samples)

#Create MDP object
mdp = mdp.SoarerDrifterMDP(test_steps)
mdp.xgoal = 26
mdp.ygoal = 15
mdp.balloon_dt = 1

mdp.import_windfield(A) #import windfield
mdp.import_actionspace([19,35], [0,5]) #import action space [xlimits], [ylimits]


state_log = []
state = [5,2,1] #initialize state
state_log.append(state)
horizon = 5
for i in range(10):
    [a_opt,v_opt] = mdp.selectaction(state, horizon)
    state = mdp.take_action(state, a_opt)
    state_log.append(state)
    #print(a_opt)
    #print(v_opt)
    #print(state)



print(state_log)
