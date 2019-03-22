#Function to create an nxm vector field for wind vectors. Outputs an n x m x 2 matrix for x and y wind speeds at each grid point.
#Alex Hirst


def generate_field(n, m):
    
    import numpy as np

    #Initialize wind matrix
    Matrix = np.zeros((2,n,m))

    Matrix[1,:,:] = 1 
    #define all y-velocities as 1m/s

    Matrix[0,:,0] = 0.25 
    #define all left hand side x-vels

    for i in range(0,n):
    #over y
        for j in range(1,m): 
        #over x
            Matrix[0,i,j] = Matrix[0,i,j-1] + np.random.uniform(-0.05,0.05)

    
    return Matrix

#test function:
n = 5
m = 8
matrix = generate_field(n,m)
print(matrix)
