#Used to test out classes and functions

import wind_field
from gen_random_field import generate_wind_field
from visualizer import animator
import timeit

start = timeit.timeit()
#test function:
n = 10
m = 10
length = 2
[vel,loc,length] = generate_wind_field(n ,m , length, 0.25, 0.1)

n_samples = 100
A = wind_field.wind_field(vel, loc, length, n_samples, 'Normal')

#test out plot:
#import subprocess as subp

#print('X grid size: ' + str(n))
#print('Y grid size: ' + str(m))
start = [5.1,2.5]
dt = 0.1
t_end = 12
A.sample_for_prop()
[x,y] = A.prop_balloon(start[0], start[1], t_end, dt, 'samples')
end = timeit.timeit()
print(end)
A.plot_wind_field()

input("press enter to continue")




'''



vis = visualizer()
vis.init_live_plot(n_samples)

for time in range(len(x[0,:])):
    for j in range(n_samples):
        vis.measurement_update(A, j, time)
    '''
