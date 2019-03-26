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
        
        #Retrieve measurement values and locations
        xmat = self.x_matrix
        ymat = self.y_matrix
        length = self.length
        matsizex = int(xmat.shape[0])
        matsizey = int(xmat.shape[1])

        #Plot data on a grid:
        plt.figure(1)

        plt.quiver(self.x_location, self.y_location, xmat,ymat)
        plt.xticks(np.arange(0, length*(matsizex), length))
        plt.yticks(np.arange(0,length*(matsizey), length))
        plt.xlabel('x coordinates (m)')
        plt.ylabel('y coordiantes (m)')
        
        plt.grid(True ,which = 'major', axis = 'both')
        
        #Show plot
        #TODO: How to make this run in background?
        plt.show()

    def get_wind(self, x, y):
       #function to get wind values from interpolating measurment vector field

        import numpy as np

        #get nearest x and y indices:
        xarray = self.x_location[:,0]
        x_idx = (np.abs(xarray - x)).argmin()
        
        if xarray[x_idx] >= x:
            x_high = x_idx
            x_low = x_idx - 1

        else:
            x_high = x_idx + 1
            x_low = x_idx

        yarray = self.y_location[0,:]
        y_idx = (np.abs(yarray - y)).argmin()
        
        if yarray[y_idx]>= y:
            y_high = y_idx
            y_low = y_idx - 1

        else:
            y_high = y_idx + 1
            y_low = y_idx

        x_points = [x_low, x_high]
        y_points = [y_low, y_high]
        
        
        a = np.matrix([float(xarray[x_high]) - x, x - float(xarray[x_low])])

        b_x = np.matrix([[float(self.x_matrix[x_low,y_low]), float(self.x_matrix[x_low,y_high])],[float(self.x_matrix[x_high,y_low]), float(self.x_matrix[x_high,y_high])]])

        b_y = np.matrix([[float(self.y_matrix[x_low,y_low]), float(self.y_matrix[x_low,y_high])], [float(self.y_matrix[x_high,y_low]), float(self.y_matrix[x_high,y_high])]])

        c = np.matrix([[float(yarray[y_high] - y)], [float(y - yarray[y_low])]])

        d = 1/((float(xarray[x_high]-xarray[x_low]))*(float(yarray[y_high]-yarray[y_low])))
        x_value = d*a*b_x*c
        y_value = d*a*b_y*c

        return [x_value, y_value]

        

    def prop_balloon(self, xstart, ystart):
        #this method propagates a balloon through the vector field to determine the uti
        #of releasing the balloon at the starting point.
        
        import scipy.integrate as integrate
        
        [x_value, y_value] = self.get_wind(xstart,ystart)
        
        print(x_value)
        print(y_value)

#test function:
n = 10
m = 10
length = 2
matrix = wind_field(n,m,length, 0.25, 0.1)

#test out plot:
#import subprocess as subp

#print('X grid size: ' + str(n))
#print('Y grid size: ' + str(m))
#matrix.plot_wind_field()

matrix.prop_balloon(2.5,3.5)

