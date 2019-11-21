#!/usr/bin/env python3

'''Used to test out balloon prop
Author: Alex Hirst
'''
from src import wind_field
from src import gen_random_field
from src import visualizer
import clear_folder
import numpy as np
import time as t


# Generate a nxm field
n = 10  # cell height of field (y)
m = 15   # cell width of field (x)
length = 3
n_samples = 100

field = gen_random_field.field_generator(n, m, length, 0, 0.15, n_samples, 'Normal')
field.nrm_mean = 0  # Can use matrix here to specify distributions for each measurement
field.nrm_sig = 1  # Can use matrix here to specify distributions for each measurement
field.sample()

A = wind_field.wind_field(field.vel, field.loc, length, field.nsamps, field.samples)

A.dt = 0.5
A.t_end = 20
A.y_goal = 20

dx = 1
xstart = 20  # absolute starting point
xend = 22  # abosulte end point
ystart = 2
num_release_pts = 3

# generate start vector
start = []
for i in np.linspace(0, xend-xstart, num_release_pts):
    print('i:' + str(i))
    start.append([xstart + i, ystart])

print(start)
for start in start:
    A.prop_balloon(start[0], start[1], A.y_goal)
    stats = A.calc_util()
    print('Start: ' + str(start))
    print('Mu: ' + str(stats[0]))
    print('Std: ' + str(stats[1]))


# A.plot_orig = True
# A.plot_orig_mean = True
A.plot_samps = True
A.plot_samps_mean = True

input("press enter to plot")

# Plot the wind field
A.plot_wind_field()

savefig = input('save figure? (y/n)')
if savefig == 'y':
    print('saving')
    A.save_plot = True
    A.plot_wind_field()

input("press enter to animate")


# Animate!!!
vis = visualizer.animator()
vis.save = True
vis.init_live_plot(A)

for time in range(len(A.position_history_y_orig)):
    t.sleep(0.1)
    vis.measurement_update(A, time)


input("press enter to create gif")
gif = input('make gif? (y/n)')
if gif == 'y':
    vis.make_gif()

# Clear the figures folder
clear_folder.clear_figure_folder()
