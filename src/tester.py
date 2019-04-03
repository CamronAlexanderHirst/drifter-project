#Used to test out classes and functions

import wind_field
from gen_random_field import field_generator
from visualizer import animator
import timeit
import numpy as np

start_time = timeit.timeit()
#test function:
n = 20
m = 10
length = 2
n_samples = 100

field = field_generator(n ,m , length, 0, 0.25, n_samples, 'Normal')
field.sample()

A = wind_field.wind_field(field.vel, field.loc, length, field.nsamps, field.samples)

#test out plot:
#import subprocess as subp

start = [8.1,2.5]
dt = 0.1
t_end = 12
A.nrm_mean = 0 #Can use matrix here to specify distributions for each measurement
A.nrm_sig = 1 #Can use matrix here to specify distributions for each measurement

A.prop_balloon(start[0], start[1], t_end, dt)

A.calc_mean()



A.plot_orig_mean = True
A.plot_samps = True
A.plot_samps_mean = True

A.plot_wind_field()

end_time = timeit.timeit()
print(end_time - start_time)

input("press enter to continue")




'''



vis = visualizer()
vis.init_live_plot(n_samples)

for time in range(len(x[0,:])):
    for j in range(n_samples):
        vis.measurement_update(A, j, time)
    '''
