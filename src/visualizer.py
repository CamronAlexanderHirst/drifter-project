'''
Visualizer for Live Plot
Alex Hirst and John Jackson
'''

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

class animator:

    def __init__(self):

            self.line_list = [] #create a line list
            self.trail_length = 10

    def init_live_plot(self, num_agents):

            self.fig = plt.figure()

            self.ax = self.fig.add_subplot(121) #create a subplot
            # Starting point is plotted as a diamond
            self.ax.plot(0, 0, 'kD')

            x_lim_min = -100
            y_lim_min = -100
            x_lim_max = 100
            y_lim_max = 100

            plt.xlim([x_lim_min, x_lim_max])
            plt.ylim([y_lim_min, y_lim_max])

            plt.xlabel('X Distance (m)')
            plt.ylabel('Y Distance (m)')
            plt.grid()

            for i in range(num_agents):
                line, = self.ax.plot([], [], '-k')
                self.line_list.append(line)

            plt.show(block=False)
            plt.tight_layout()

    def measurement_update(self, wind_field, j, time):

            # This updates jth agent (balloon) at a time
            posx = np.array(wind_field.position_history_x[j ,:])
            posy = np.array(wind_field.position_history_y[j ,:])

            if time <= self.trail_length:
                self.line_list[j].set_data(posx[0:time], posy[0:time])
            else:
                self.line_list[j].set_data(posx[0:time][-self.trail_length:], posy[0:time][-self.trail_length:])

            self.fig.canvas.draw()
            self.fig.canvas.flush_events()
