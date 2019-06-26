''' Create a .gif file of balloon propagation
Author: Alex Hirst
'''

from src import wind_field
from src import gen_random_field
from src import visualizer
import numpy as np
import time as t
from matplotlib.animation import FuncAnimation


def update(i):
    label = 'timestep {0}'.format(i)
    print(label)

    for j in range(i):
        return line, ax


if __name__ == '__main__':
    # test function:

    # Generate a nxm field
    n = 20  # cell height of field (y)
    m = 30  # cell width of field (x)
    length = 3
    n_samples = 25

    # Create a random field object
    field = gen_random_field.field_generator(n, m, length, 0, 0.15, n_samples, 'Normal')
    field.nrm_mean = 0  # Can use matrix here to specify distributions for each measurement
    field.nrm_sig = 1  # Can use matrix here to specify distributions for each measurement
    field.sample()  # take n_samples realizations of stoichastic wind field.

    # Initilize wind field object with randomly generated field.
    A = wind_field.wind_field(field.vel, field.loc, length, field.nsamps, field.samples)

    dt = 0.25  # timestep size for balloon propagation
    t_end = 30  # Final time for balloon propagation

    dx = 1  # distance between release points
    xstart = 15  # absolute starting point
    xend = 45  # abosulte end point
    ystart = 2  # starting y-value
    num_release_pts = 5  # number of release points

    # generate start vector for simulations
    start = []
    for i in np.linspace(0, xend-xstart, num_release_pts):
        print('i:' + str(i))
        start.append([xstart + i, ystart])

    x_history = np.zeros((len(start), n_samples, int(t_end/dt)+1))
    y_history = np.zeros((len(start), n_samples, int(t_end/dt)+1))
    mu_history = np.zeros((len(start), 1))
    std_history = np.zeros((len(start), 1))

    i = 0
    for start in start:
        A.prop_balloon(start[0], start[1], t_end, dt)
        stats = A.calc_util(5, 1)
        print('Start: ' + str(start))
        print('Mu: ' + str(stats[0]))
        print('Std: ' + str(stats[1]))

        mu_history[i, 0] = stats[0]
        std_history[i, 0] = stats[1]
        x_history[i, :, :] = A.position_history_x_samps
        y_history[i, :, :] = A.position_history_y_samps

        i = 1+1

    input("press enter to animate")

    # Animate!!!
    vis = visualizer.animator()
    vis.init_live_plot(A)

    for time in range(int(t_end/dt)):
        t.sleep(0.15)
        vis.measurement_update(A, time)

    input("press enter to create gif")
    fig = vis.fig
    ax = vis.ax
    line, = ax.pl

    anim = FuncAnimation(fig, update, frames=np.arange(0, int(t_end/dt)+1), interval=200)
    anim.save('balloon.gif', dpi=80, writer='imagemagick')
