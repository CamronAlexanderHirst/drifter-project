#Function to create an nxm vector field for wind vectors. Outputs an n x m x 2 matrix for x and y wind speeds at each grid point.
#Alex Hirst

class wind_field:

    def __init__(self, n, m, length, init_x, dist):

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


        self.x_matrix = Matrix[0,:,:]
        self.y_matrix = Matrix[1,:,:]
        self.x_location = location_Matrix[0,:,:]
        self.y_location = location_Matrix[1,:,:]

        self.length = length


    def plot_wind_field(self):
        #This method creates a new figure, and plots the wind vector field on a gridded
        #space. matplotlib dependent. arguement is the grid size in meters. 
        import matplotlib.pyplot as plt
        import numpy as np
        
        xmat = self.x_matrix
        print(xmat)
        ymat = self.y_matrix
        length = self.length
        matsizex = int(xmat.shape[0])
        matsizey = int(xmat.shape[1])

        plt.figure(1)

        plt.quiver(self.x_location, self.y_location, xmat,ymat)
        plt.xticks(np.arange(0, length*(matsizex), length))
        plt.yticks(np.arange(0,length*(matsizey), length))
        plt.xlabel('x coordinates (m)')
        plt.ylabel('y coordiantes (m)')
        
        plt.grid(True ,which = 'major', axis = 'both')
        plt.show()

#test function:
n = 10
m = 10
length = 2
matrix = wind_field(n,m,length, 0.25, 0.1)

#test out plot:
import subprocess as subp

print('X grid size: ' + str(n))
print('Y grid size: ' + str(m))
matrix.plot_wind_field()


