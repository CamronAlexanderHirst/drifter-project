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
            self.trail_length = 50

    def init_live_plot(self, wind_field):

            self.fig = plt.figure()

            self.ax = self.fig.add_subplot(121) #create a subplot
            # Starting point is plotted as a diamond
            self.ax.plot(0, 0, 'kD')

            x_lim_min = wind_field.x_location[0,0]-1
            y_lim_min = wind_field.y_location[0,0]-1

            x_lim_max = wind_field.x_location[1,-1]+1
            y_lim_max = wind_field.y_location[-1,0]+1

            print(y_lim_max)

            plt.xlim([x_lim_min, x_lim_max])
            plt.ylim([y_lim_min, y_lim_max])

            plt.xlabel('X Distance (m)')
            plt.ylabel('Y Distance (m)')
            plt.grid()

            num_agents = wind_field.nsamps

            for i in range(num_agents):
                line, = self.ax.plot([], [], '-k')
                self.line_list.append(line)

            plt.show(block=False)
            plt.tight_layout()

    def measurement_update(self, wind_field, j, time):
            # This updates jth agent (balloon) at a time
            posx = np.array(wind_field.position_history_x_samps[j ,:])
            posy = np.array(wind_field.position_history_y_samps[j ,:])

            if time <= self.trail_length:
                self.line_list[j].set_data(posx[0:time], posy[0:time])
            else:
                self.line_list[j].set_data(posx[0:time][-self.trail_length:], posy[0:time][-self.trail_length:])

            self.fig.canvas.draw()
            self.fig.canvas.flush_events()
