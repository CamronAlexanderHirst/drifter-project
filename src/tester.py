#Used to test out classes and functions

import wind_field
from gen_random_field import generate_wind_field

#test function:
n = 10
m = 10
length = 2
[vel,loc,length] = generate_wind_field(n ,m , length, 0.25, 0.1)

A = wind_field.wind_field(vel, loc, length, 1000, 'Normal')

#test out plot:
#import subprocess as subp

#print('X grid size: ' + str(n))
#print('Y grid size: ' + str(m))
#matrix.plot_wind_field()

A.sample_for_prop()
