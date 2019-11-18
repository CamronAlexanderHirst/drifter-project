'''class to generate a wind field
Author: Alex Hirst
'''


class DistroNotRecognized(Exception):
    message = '\nException: DistroNotRecognizedError:\nDistribution not recognized\
     \nTry either Normal or Uniform '


class field_generator:

    def __init__(self, n, m, length, init_x, dist, nsamps, distro):
        import numpy as np

        # Initialize wind and measurement location matrices
        Matrix = np.zeros((2, n, m))
        location_Matrix = np.zeros((2, n, m))

        Matrix[1, :, :] = 1
        # define all y-velocities as 1m/s

        Matrix[0, :, 0] = init_x
        # define all left hand side x-vels (rest are unifromly sampled)

        for i in range(0, n):
            # over y
            for j in range(1, m):
                # over x
                # generated x - velocities
                Matrix[0, i, j] = Matrix[0, i, j-1] + np.random.uniform(-1*dist, dist)

            for j in range(0, m):
                location_Matrix[1, i, j] = i*length  # matrix of y positions
                location_Matrix[0, i, j] = j*length  # matrix of x positions

        self.vel = Matrix
        self.loc = location_Matrix

        # Initialize wind and measurement location matrices
        self.x_matrix = self.vel[0, :, :]  # matrix of x-velocities
        self.y_matrix = self.vel[1, :, :]
        self.x_location = self.loc[0, :, :]  # Matrix of x-locations
        self.y_location = self.loc[1, :, :]

        # pick if we want a normal or uniformly distributed vector field
        if distro == 'Normal':
            self.distro = 'Normal'
        elif distro == 'Uniform':
            self.distro = 'Uniform'
        else:
            raise DistroNotRecognized

        self.nsamps = nsamps  # number of wind field samples
        self.length = length  # cell unit width

        # initialize sampling distributions (can change to matrices to make different for each pt)
        # to change, just change in outer script
        self.nrm_sig = 1
        self.nrm_mean = 0
        self.uni_mean = 0
        self.uni_rng = 0.1

    def sample(self):
        '''Takes nsamps samples of the wind field according to
        the distro variable. i.e. if we are simulating 500 balloons,
        this method will create two 500xmxn matrix of wind field measurements.
        One for x direction and one for y directon.

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

        for i in range(0, N):

            if self.distro == 'Normal':
                x = np.random.normal(self.nrm_mean, self.nrm_sig, size) + x_orig  # normal distribution if second argument = 1
                y = y_orig  # + np.random.normal(0, 1, size)

            elif self.distro == 'Uniform':
                x = np.random.uniform(self.uni_mean-self.uni_rng, self.uni_mean+self.uni_rng, size) + x_orig
                y = y_orig  # + np.random.uniform(-0.1,0.1, size)

            x_out[i, :, :] = x
            y_out[i, :, :] = y

        self.samples = [x_out, y_out]

    def sample_heading_speed(self):
        '''Takes nsamps samples of the wind field according to
        the distro variable. i.e. if we are simulating 500 balloons,
        this method will create two 500xmxn matrix of wind field measurements.
        One for x direction and one for y directon.

        This function differentiates from above sample method by
        reasoning over heading and speed variation, a change that was
        made in the SciTech paper.

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

        # convert x_orig and y_orig to heading and speed matrices.
        speed_orig = (x_orig**2 + y_orig**2)/2
        heading_orig = np.arctan2(x_orig, y_orig)  # checked in terminal

        for i in range(0, N):

            if self.distro == 'Normal':
                ds = np.random.normal(self.nrm_mean_s, self.nrm_sig_s, size)
                dh = np.random.normal(self.nrm_mean_h, self.nrm_sig_h, size)

            elif self.distro == 'Uniform':
                ds = np.random.uniform(self.uni_mean_s-self.uni_rng_s,
                                       self.uni_mean_s+self.uni_rng_s, size)
                dh = np.random.uniform(self.uni_mean_h-self.uni_rng_h,
                                       self.uni_mean_h+self.uni_rng_h, size)

            heading = heading_orig + dh
            speed  = speed_orig + ds

            x = speed * np.sin(heading)
            y = speed * np.cos(heading)

            x_out[i, :, :] = x
            y_out[i, :, :] = y

        self.samples = [x_out, y_out]
