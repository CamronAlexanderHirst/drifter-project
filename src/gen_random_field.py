#function to generate a wind field

def generate_wind_field(n, m, length, init_x, dist):

    import numpy as np

    #Initialize wind and measurement location matrices
    Matrix = np.zeros((2,n,m))
    location_Matrix = np.zeros((2,n,m))

    Matrix[1,:,:] = 1
    #define all y-velocities as 1m/s

    Matrix[0,:,0] = init_x
    #define all left hand side x-vels

    for i in range(0,n):
    #over y
        for j in range(1,m):
        #over x
            Matrix[0,i,j] = Matrix[0,i,j-1] + np.random.uniform(-1*dist,dist)

        for j in range(0,m):
            location_Matrix[0,i,j] = i*length
            location_Matrix[1,i,j] = j*length


    vel = Matrix
    loc = location_Matrix

    length = length

    return [vel, loc, length]
