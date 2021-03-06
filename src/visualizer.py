'''
Visualizer for Live Plot
Alex Hirst and John Jackson
'''

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

#matplotlib.use('TkAgg')  # for macs


class animator:
    """
    animator class for the balloonproptest.py script. Will only display the
    wind field and balloon trajectories.
    """

    def __init__(self):

        self.line_list = []  # create a line list
        self.trail_length = 500
        self.save = False

    def init_live_plot(self, wind_field):

        # Initialize live plot
        self.fig = plt.figure(figsize=(4,6))
        self.ax = self.fig.add_subplot(111)  # create a subplot
        self.ax.axis('equal')

        # Starting point is plotted as a diamond
        self.ax.plot(wind_field.position_history_x_orig[0], wind_field.position_history_y_orig[0], 'bD')

        # plot vector field
        xmat = wind_field.x_matrix
        ymat = wind_field.y_matrix
        plt.quiver(wind_field.x_location, wind_field.y_location, xmat, ymat)

        length = wind_field.length
        x_lim_min = wind_field.x_location[0, 0]-1
        y_lim_min = wind_field.y_location[0, 0]-1

        x_lim_max = wind_field.x_location[1, -1]+1
        y_lim_max = wind_field.y_location[-1, 0]+1

        xt = np.arange(0, length*(wind_field.matsizex), length)
        yt = np.arange(0, length*(wind_field.matsizey), length)


        self.ax.set_xticks([x for x in xt])
        self.ax.set_xticklabels([str(x/1000) for x in xt])
        self.ax.set_yticks([x for x in xt])
        self.ax.set_yticklabels([str(x/1000) for x in yt])

        plt.xlim([x_lim_min, x_lim_max])
        plt.ylim([y_lim_min, y_lim_max])

        plt.xlabel('$Z_1$ (km)')
        plt.ylabel('$Z_2$ (km)')
        plt.title(title)

        plt.grid()

        num_agents = wind_field.nsamps

        for i in range(num_agents):
            line, = self.ax.plot([], [])
            self.line_list.append(line)

        plt.show(block=False)
        plt.tight_layout()

    def measurement_update(self, wind_field, time):
        # This updates all agents at a time
        posx = wind_field.position_history_x_samps
        posy = wind_field.position_history_y_samps
        nsamps = wind_field.nsamps

        for j in range(nsamps):
            if time <= self.trail_length:
                self.line_list[j].set_data(posx[j][0:time], posy[j][0:time])
            else:
                self.line_list[j].set_data(posx[j][0:time][-self.trail_length:], posy[j][0:time][-self.trail_length:])

        self.fig.canvas.draw()
        self.fig.canvas.flush_events()

        self.timesteps = time

        if self.save is True:
            plt.savefig('./figures/'+str(time)+'.png')

    def make_gif(self):

        print('making gif...')
        import imageio
        images = []

        filenames = []
        for i in range(self.timesteps+1):
            filenames.append('./figures/'+str(i)+'.png')

        for filename in filenames:
            images.append(imageio.imread(filename))
        imageio.mimsave('./outputs/movie.gif', images, duration=0.2)

        print('done')


class animator_mdp:
    """
    animator for the mdp solver
    """

    def __init__(self):

        self.line_list = []  # create a line list
        self.trail_length = 500
        self.save = False

    def simple_plot(self, mdp, title):
        self.wind_field = mdp.field
        self.num_ballons = mdp.num_goals
        wind_field = self.wind_field

        self.state_history = mdp.state_history  # aircraft state history
        self.release_traj = mdp.release_traj

        # create subplot
        self.fig = plt.figure(figsize=(9, 5))
        self.ax = self.fig.add_subplot(111)  # create a subplot
        self.ax.axis('equal')


        # plot vector field
        xmat = wind_field.x_matrix
        ymat = wind_field.y_matrix
        plt.quiver(wind_field.x_location, wind_field.y_location, xmat, ymat, width=0.0025)
        self.ax.set_axisbelow(True)
        plt.grid()

        length = wind_field.length
        x_lim_min = wind_field.x_location[0, 0]-1
        y_lim_min = wind_field.y_location[0, 0]-1

        x_lim_max = wind_field.x_location[1, -1]+1
        y_lim_max = wind_field.y_location[-1, 0]+1

        xt = np.arange(0, length*(wind_field.matsizex), length)
        yt = np.arange(0, length*(wind_field.matsizey), length)

        self.ax.set_xticks([x for x in xt])
        self.ax.set_xticklabels([str(x/1000) for x in xt])
        self.ax.set_yticks([x for x in xt])
        self.ax.set_yticklabels([str(x/1000) for x in yt])

        plt.xlim([x_lim_min, x_lim_max])
        plt.ylim([y_lim_min, y_lim_max])

        plt.xlabel('$Z_1$ (km)')
        plt.ylabel('$Z_2$ (km)')
        plt.title(title)

        # set every other label off
        for label in self.ax.xaxis.get_ticklabels()[1::2]:
            label.set_visible(False)
        for label in self.ax.yaxis.get_ticklabels()[1::2]:
            label.set_visible(False)


        # find state where balloon is realeased
        i = 0
        old_state = self.state_history[0][2]
        self.release_times = {}
        for state in self.state_history:
            if state[2] == (old_state - 1):
                self.release_times[self.num_ballons - old_state] = i
                old_state = state[2]
                if state[2] == 0:
                    break
            i = i + 1

        # This updates all agents at a time
        wind_field = self.wind_field
        release_times = self.release_times
        state_history = self.state_history
        nsamps = wind_field.nsamps

        x = []
        y = []
        for state in state_history[:]:
            x.append(state[0])
            y.append(state[1])

        self.state_list = self.ax.plot([], [], 'bo', markersize=4)
        self.state_list[0].set_data(x, y)

        release_times = self.release_times
        for release_num in release_times:
            release_time = release_times[release_num]

            posx = self.release_traj[release_num][0]
            posy = self.release_traj[release_num][1]

            for j in range(nsamps):
                    line = self.ax.plot(posx[j], posy[j], zorder=2)

        for i in range(len(mdp.xgoals)):
            self.ax.plot(mdp.xgoals[i], mdp.ygoals[i], 'rD', markersize=10, zorder=3)

        # bounding box
        xmin, xmax, ymin, ymax = mdp.xmin+1000, mdp.xmax-500, mdp.ymin, mdp.ymax
        bbox = [[xmin, xmax, xmax, xmin, xmin],
                [ymin, ymin, ymax, ymax, ymin]]
        self.ax.plot(bbox[0], bbox[1], color='orange', zorder=1)

        plt.show(block=False)


    def init_live_plot(self, mdp):

        self.wind_field = mdp.field
        self.num_ballons = mdp.num_goals
        wind_field = self.wind_field

        self.state_history = mdp.state_history  # aircraft state history
        self.release_traj = mdp.release_traj

        # find state where balloon is realeased
        i = 0
        old_state = self.state_history[0][2]
        self.release_times = {}
        for state in self.state_history:
            if state[2] == (old_state - 1):
                self.release_times[self.num_ballons - old_state] = i
                old_state = state[2]
                if state[2] == 0:
                    break
            i = i + 1


        y_pos = wind_field.position_history_y_samps
        self.total_time = i + max([len(b) for b in y_pos]) + 1
        if len(self.state_history) > self.total_time:
            self.total_time = len(self.state_history)

        # y_release = self.state_history[i][1]
        # self.wind_field.position_history_y_samps = y_release + self.wind_field.position_history_y_samps
        # Initialize live plot

        self.fig = plt.figure()
        self.ax = self.fig.add_subplot(111)  # create a subplot
        self.ax.axis('equal')

        # Starting point is plotted as a diamond
        for i in range(len(mdp.xgoals)):
            self.ax.plot(mdp.xgoals[i], mdp.ygoals[i], 'rD', markersize=10, zorder=1)

        # plot vector field
        xmat = wind_field.x_matrix
        ymat = wind_field.y_matrix
        plt.quiver(wind_field.x_location, wind_field.y_location, xmat, ymat)

        length = wind_field.length
        x_lim_min = wind_field.x_location[0, 0]-1
        y_lim_min = wind_field.y_location[0, 0]-1

        x_lim_max = wind_field.x_location[1, -1]+1
        y_lim_max = wind_field.y_location[-1, 0]+1

        plt.xticks(np.arange(0, length*(wind_field.matsizex), length))
        plt.yticks(np.arange(0, length*(wind_field.matsizey), length))

        plt.xlim([x_lim_min, x_lim_max])
        plt.ylim([y_lim_min, y_lim_max])

        plt.xlabel('X')
        plt.ylabel('Y')
        plt.grid()

        num_agents = wind_field.nsamps

        self.line_dict = {}

        for release_num in self.release_times:
            self.line_dict[release_num] = []

        for release_num in self.release_times:
            for i in range(num_agents):
                line, = self.ax.plot([], [], zorder=1)
                self.line_dict[release_num].append(line)

        self.state_list = self.ax.plot([], [], 'bo', markersize=8)

        plt.show(block=False)
        plt.tight_layout()

    def measurement_update(self, time):
        # This updates all agents at a time
        wind_field = self.wind_field
        release_times = self.release_times
        state_history = self.state_history

        nsamps = wind_field.nsamps

        x = []
        y = []
        for state in state_history[0:min([time, len(state_history)])]:
            x.append(state[0])
            y.append(state[1])

        self.state_list[0].set_data(x, y)

        for release_num in release_times:
            release_time = release_times[release_num]

            posx = self.release_traj[release_num][0]
            posy = self.release_traj[release_num][1]

            if time > release_time:
                time2 = time - release_time
                for j in range(nsamps):
                    if time <= self.trail_length:
                        self.line_dict[release_num][j].set_data(posx[j][0:time2], posy[j][0:time2])
                    else:
                        self.line_dict[release_num][j].set_data(posx[j][0:time2][-self.trail_length:], posy[j][0:time2][-self.trail_length:])

        self.fig.canvas.draw()
        self.fig.canvas.flush_events()

        self.timesteps = time

        if self.save is True:
            plt.savefig('./figures/'+str(time)+'.png')

    def make_gif(self):

        print('making gif...')
        import imageio
        images = []

        filenames = []
        for i in range(self.timesteps+1):
            filenames.append('./figures/'+str(i)+'.png')

        for filename in filenames:
            images.append(imageio.imread(filename))
        imageio.mimsave('./outputs/movie.gif', images, duration=0.15)

        print('done')
