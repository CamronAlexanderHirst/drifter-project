#Used to test out balloon prop

import wind_field
import gen_random_field
import visualizer
import timeit
import numpy as np
import time as t



#test function:

#Generate a nxm field
n = 20#height of field (y)
m = 40 #width of field (x)
length = 1
n_samples = 100

field = gen_random_field.field_generator(n ,m , length, 0, 0.15, n_samples, 'Normal')
field.nrm_mean = 0 #Can use matrix here to specify distributions for each measurement
field.nrm_sig = 1 #Can use matrix here to specify distributions for each measurement
field.sample()

A = wind_field.wind_field(field.vel, field.loc, length, field.nsamps, field.samples)

print('x array:')
print(field.loc[0,:,:])
print('y array:')
print(field.loc[1,:,:])



dt = 0.25
t_end = 12.5
dx = 1
xstart = 8
xend = 12
ystart = 2
num_release_pts = 2

#generate start vector
start = []
for i in np.linspace(0, xend-xstart, num_release_pts):
    print('i:' + str(i))
    start.append([xstart + i, ystart])

print(start)
for start in start:
    A.prop_balloon(start[0], start[1], t_end, dt)
    stats = A.calc_util(5,1)
    print('Start: '+ str(start))
    print('Mu: ' + str(stats[0]))
    print('Std: ' + str(stats[1]))


A.plot_orig = True
A.plot_orig_mean = True
#A.plot_samps = True
#A.plot_samps_mean = True




input("press enter to plot")

#Plot the wind field
A.plot_wind_field()

input("press enter to animate")

#Animate!!!
vis = visualizer.animator()
vis.init_live_plot(A)

for time in range(int(t_end/dt)):
    t.sleep(0.15)
    vis.measurement_update(A, time)



input("press enter to end")
