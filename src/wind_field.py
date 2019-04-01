'''Wind Field class, used for storing and propagating
balloons through estimated wind fields.

Alex Hirst
'''

class DistroNotRecognized(Exception):
    message = '\nException: DistroNotRecognizedError:\nDistribution not recognized\
     \nTry either Normal or Uniform '

class WhichSampleNotRecognized(Exception):
     message = '\nException: WhichSampleNotRecognizedError:\nWhich not recognized\
      \nTry either original or samples '

class wind_field:

    def __init__(self, vel, loc, length, nsamps, distro):
        import numpy as np

        #Initialize wind and measurement location matrices
        self.x_matrix = vel[0,:,:] #matrix of x-velocities
        self.y_matrix = vel[1,:,:]
        self.x_location = loc[0,:,:] #Matrix of x-locations
        self.y_location = loc[1,:,:]
        self.nsamps = nsamps
        self.length = length

        if distro == 'Normal':
            self.distro = 'Normal'
        elif distro == 'Uniform':
            self.distro = 'Uniform'
        else:
            raise DistroNotRecognized

    def plot_wind_field(self):
        #This method creates a new figure, and plots the wind vector field on a gridded
        #space. matplotlib dependent.
        import matplotlib.pyplot as plt
        import numpy as np

        #Retrieve measurement values and locations
        xmat = self.x_matrix
        ymat = self.y_matrix
        length = self.length
        matsizex = int(xmat.shape[0])
        matsizey = int(xmat.shape[1])

        #Get x_traj and y_traj (MUST RUN prop_balloon FIRST)
        x_traj = self.position_history_x
        y_traj = self.position_history_y
        nsamps = self.nsamps

        xmean = self.calc_mean(x_traj)
        ymean = y_traj[0,-1]
        #Plot data on a grid:
        plt.figure()
        #plt.ion()

        plt.quiver(self.x_location, self.y_location, xmat,ymat)
        #plt.hold(True)
        #TODO: How to make the trajectory plot?

        for i in range(nsamps):
            plt.plot(x_traj[i,:], y_traj[i,:])


        plt.plot(xmean,ymean,'ro')

        plt.axis([self.x_location[0,0]-1, self.x_location[-1,1]+1, self.y_location[0,0]-1, self.y_location[-1,1]+1])
        plt.xticks(np.arange(0, length*(matsizex), length))
        plt.yticks(np.arange(0,length*(matsizey), length))
        plt.xlabel('x coordinates (m)')
        plt.ylabel('y coordiantes (m)')

        plt.grid(True ,which = 'major', axis = 'both')

        #Show plot
        #TODO: How to make this run in background?
        plt.draw()
        plt.show(block=False)
        #


    def get_wind(self, x, y, n, which):
       #function to get wind values from interpolating measurment vector field
        import numpy as np

        if which == 'samples':
            x_matrix = self.x_samples[n,:,:]
            y_matrix = self.y_samples[n,:,:]
        elif which == 'original':
            x_matrix = self.x_matrix[:,:]
            y_matrix = self.y_matrix[:,:]
        else:
            raise WhichSampleNotRecognized

        #get nearest x and y indices:
        xarray = self.x_location[:,0]
        yarray = self.y_location[0,:]

        x_idx = (np.abs(xarray - x)).argmin()

        if xarray[x_idx] >= x:
            x_high = x_idx
            x_low = x_idx - 1

        else:
            x_high = x_idx + 1
            x_low = x_idx


        y_idx = (np.abs(yarray - y)).argmin()

        if yarray[y_idx]>= y:
            y_high = y_idx
            y_low = y_idx - 1

        else:
            y_high = y_idx + 1
            y_low = y_idx

        x_points = [x_low, x_high]
        y_points = [y_low, y_high]

        #Interpolation:

        a = np.matrix([float(xarray[x_high]) - x, x - float(xarray[x_low])])

        b_x = np.matrix([[float(x_matrix[x_low,y_low]), float(x_matrix[x_low,y_high])], [float(x_matrix[x_high,y_low]), float(x_matrix[x_high,y_high])]])

        b_y = np.matrix([[float(y_matrix[x_low,y_low]), float(y_matrix[x_low,y_high])], [float(y_matrix[x_high,y_low]), float(y_matrix[x_high,y_high])]])

        c = np.matrix([[float(yarray[y_high] - y)], [float(y - yarray[y_low])]])

        d = 1/((float(xarray[x_high]-xarray[x_low]))*(float(yarray[y_high]-yarray[y_low])))
        x_value = float(d*a*b_x*c)
        y_value = float(d*a*b_y*c)

        return [x_value, y_value]

    def prop_balloon(self, xstart, ystart, tend, dt, which):
        '''this method propagates a balloon through the vector field to determine the utility
        of releasing the balloon at the starting point.
        '''

        import numpy as np

        if which == 'samples':
            N = self.nsamps
        elif which == 'original':
            N = 1
        else:
            raise WhichSampleNotRecognized

        t_vect = np.arange(0,tend+dt,dt)
        x_vect = np.zeros((N,1,len(t_vect)))
        y_vect = np.zeros((N,1,len(t_vect)))

        x_vect[:,0,0] = xstart
        y_vect[:,0,0] = ystart

        for n in range(0,N):
            for i in range(1,len(t_vect)):
                [x_vel, y_vel] = self.get_wind(x_vect[n,0,i-1],y_vect[n,0,i-1],n, which)
                x_vect[n,0,i] = x_vect[n,0,i-1] + x_vel*dt
                y_vect[n,0,i] = y_vect[n,0,i-1] + y_vel*dt


        self.position_history_x = np.squeeze(x_vect)
        self.position_history_y = np.squeeze(y_vect)

        return [np.squeeze(x_vect),np.squeeze(y_vect)]

    def sample_for_prop(self):
        '''Takes nsamps samples of the wind field according to
        the distro variable. i.e. if we are simulating 500 balloons,
        this method will create an mxnx500 matrix of wind field measurements.

        Currently the only supported distributions
        are uniform and normal.'''
        import numpy as np

        x_orig = self.x_matrix
        y_orig = self.y_matrix

        N = self.nsamps
        size = self.x_matrix.shape

        size_out = (N,) + size

        x_out = np.zeros(size_out)
        y_out = np.zeros(size_out)

        for i in range(0,N):

            if self.distro == 'Normal':
                x = np.random.normal(0,0.5, size) + x_orig #normal distribution if second argument = 1
                y = y_orig #+ np.random.normal(0,1, size)

            elif self.distro == 'Uniform':
                x = np.random.uniform(-0.1,0.1, size) + x_orig
                y = y_orig #+ np.random.uniform(-0.1,0.1, size)

            x_out[i,:,:] = x
            y_out[i,:,:] = y

        self.x_samples = x_out
        self.y_samples = y_out

    def calc_mean(self, x):
        import numpy as np

        #take last x-values:
        mean = np.mean(x[:,-1])
        return mean
