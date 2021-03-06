'''Wind Field class, used for storing and propagating
balloons through estimated wind fields.

Alex Hirst
'''


class WhichSampleNotRecognized(Exception):
    message = '\nException: WhichSampleNotRecognizedError:\nWhich not recognized\
        \nTry either original or samples '


class wind_field:

    def __init__(self, vel, loc, length, nsamps, samps):

        # Initialize wind and measurement location matrices for vector field.
        self.x_matrix = vel[0, :, :]  # matrix of x-velocities
        self.y_matrix = vel[1, :, :]  # matrix of y_velocities
        self.x_location = loc[0, :, :]  # Matrix of x-locations
        self.y_location = loc[1, :, :]  # matrix of y-locations

        self.nsamps = nsamps  # Number of samples (balloons) propagated
        self.length = length  # length of one grid cell

        # Parameters for plot_wind_field settings
        self.plot_samps = False
        self.plot_samps_mean = False
        self.plot_orig = False
        self.plot_orig_mean = False

        # matrices of vector field realizations
        self.x_samples = samps[0]
        self.y_samples = samps[1]

        self.matsizex = int(self.x_matrix.shape[1])
        self.matsizey = int(self.x_matrix.shape[0])

        self.save_plot = False

        self.tend = 20
        self.dt = 1
        self.y_goal = 10

    def plot_wind_field(self):
        # This method creates a new figure, and plots the wind vector field on
        # a gridded space. matplotlib dependent.
        import matplotlib.pyplot as plt
        import numpy as np

        # Retrieve measurement values and locations
        xmat = self.x_matrix
        ymat = self.y_matrix
        length = self.length

        self.calc_mean()

        # Get x_traj and y_traj (MUST RUN prop_balloon FIRST)

        self.fig = plt.figure(figsize=(8,6))
        self.ax = self.fig.add_subplot(111)  # create a subplot
        self.ax.axis('equal')

        plt.quiver(self.x_location, self.y_location, xmat, ymat)

        if self.plot_samps:
            x_traj = self.position_history_x_samps
            y_traj = self.position_history_y_samps
            for i in range(self.nsamps):
                plt.plot(x_traj[i], y_traj[i])

        if self.plot_samps_mean:
            plt.plot(self.xmean_samps,
                     self.position_history_y_samps[0][-1], 'ro', markersize=10)

        if self.plot_orig:
            x_traj = self.position_history_x_orig
            y_traj = self.position_history_y_orig
            plt.plot(x_traj, y_traj)

        if self.plot_orig_mean:
            plt.plot(self.xmean_orig, self.position_history_y_orig[-1], 'bo',
                     markersize=8)

        # plt.axis([self.x_location[0, 0]-1, self.x_location[1, -1]+1,
        #           self.y_location[0, 0]-1, self.y_location[-1, 1]+1])

        xt = np.arange(0, length*(self.matsizex), length)
        yt = np.arange(0, length*(self.matsizey), length)

        self.ax.set_xticks([x for x in xt])
        self.ax.set_xticklabels([str(x/1000) for x in xt])
        self.ax.set_yticks([x for x in xt])
        self.ax.set_yticklabels([str(x/1000) for x in yt])

        x_lim_min = self.x_location[0, 0]-1
        y_lim_min = self.y_location[0, 0]-1
        x_lim_max = self.x_location[1, -1]+1
        y_lim_max = self.y_location[-1, 0]+1
        plt.xlim([x_lim_min, x_lim_max])
        plt.ylim([y_lim_min, y_lim_max])

        plt.xlabel('$Z_1$ (km)')
        plt.ylabel('$Z_2$ (km)')
        plt.title('Example Drifter Propagation')

        plt.grid()

        # Show plot
        # TODO: How to make this run in background?
        plt.draw()
        if self.save_plot is True:
            print('saved fig')
            plt.savefig('./outputs/plot.png')
        plt.show(block=False)

    def get_wind(self, x, y, n, which):
        # function to get wind values from interpolating measurment vector field
        import numpy as np

        if which == 'samples':
            x_matrix = self.x_samples[n, :, :]
            y_matrix = self.y_samples[n, :, :]
        elif which == 'original':
            x_matrix = self.x_matrix[:, :]
            y_matrix = self.y_matrix[:, :]
        else:
            raise WhichSampleNotRecognized

        # get nearest x and y indices:
        xarray = self.x_location[0, :]
        yarray = self.y_location[:, 0]

        x_idx = (np.abs(xarray - x)).argmin()

        if xarray[x_idx] >= x:
            x_high = x_idx
            x_low = x_idx - 1

        else:
            x_high = x_idx + 1
            x_low = x_idx

        y_idx = (np.abs(yarray - y)).argmin()

        if yarray[y_idx] >= y:
            y_high = y_idx
            y_low = y_idx - 1

        else:
            y_high = y_idx + 1
            y_low = y_idx

        # Bilinear Interpolation step:
        # See bilinear interpolation page on wikipedia

        a = np.matrix([float(xarray[x_high]) - x, x - float(xarray[x_low])])

        b_x = np.matrix([[float(x_matrix[y_low, x_low]),
                          float(x_matrix[y_high, x_low])],
                         [float(x_matrix[y_low, x_high]),
                          float(x_matrix[y_high, x_high])]])

        b_y = np.matrix([[float(y_matrix[y_low, x_low]),
                          float(y_matrix[y_high, x_low])],
                         [float(y_matrix[y_low, x_high]),
                          float(y_matrix[y_high, x_high])]])

        c = np.matrix([[float(yarray[y_high] - y)], [float(y - yarray[y_low])]])

        d = 1/((float(xarray[x_high]-xarray[x_low]))*(float(yarray[y_high]-yarray[y_low])))

        x_value = float(d*a*b_x*c)
        y_value = float(d*a*b_y*c)

        return [x_value, y_value]

    def prop_balloon(self, xstart, ystart, y_goal):
        '''this method propagates a balloon through the vector field to
        determine the utility of releasing the balloon at the starting point.
        xstart and ystart are the absolute starting positions of the balloon

        y_goal was added for additional functionality.

        TODO:
        - make an input the y_goal position, prop the balloons till they
        reach that final y_goal. store trajectories in lists, not in numpy
        arrays so that each list can be a different length.
        '''

        import numpy as np

        dt = self.dt
        tend = int((self.y_goal-ystart))  # to get constant y_end
        # Propogate samples:
        N = self.nsamps

        t_list = []
        x_list = []
        y_list = []
        #print("proping wind field")
        for n in range(0, N):
            x = [xstart]
            y = [ystart]
            while y[-1] < y_goal:
                [x_vel, y_vel] = self.get_wind(x[-1], y[-1], n, 'samples')
                x.append(x[-1] + x_vel*dt)
                y.append(y[-1] + y_vel*dt)
            x_list.append(x)
            y_list.append(y)

        # position history of balloon propagated through sampled fields
        self.position_history_x_samps = x_list  # x-position
        self.position_history_y_samps = y_list  # y-position

        # Propagate original:
        N = 1
        t_list = []
        x_list = []
        y_list = []

        for n in range(0, N):
            x = [xstart]
            y = [ystart]
            while y[-1] < y_goal:
                [x_vel, y_vel] = self.get_wind(x[-1], y[-1], n, 'samples')
                x.append(x[-1] + x_vel*dt)
                y.append(y[-1] + y_vel*dt)
            x_list.append(x)
            y_list.append(y)

        # position history of balloon propagated through original field
        self.position_history_x_orig = x_list  # x-position
        self.position_history_y_orig = y_list  # y-position
        #print("done")

    def calc_mean(self):
        import numpy as np

        # take last x-values:
        x_orig = self.position_history_x_orig
        x_samps = self.position_history_x_samps
        l = [x[-1] for x in x_samps]

        self.xmean_samps = np.mean(l)
        self.xmean_orig = np.mean(x_orig[0][-1])

    def calc_util(self):
        from scipy.stats import norm

        x_samps = self.position_history_x_samps
        l = [x[-1] for x in x_samps]

        mu, std = norm.fit(l)

        return [mu, std]
