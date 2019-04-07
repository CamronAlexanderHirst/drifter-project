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
            self.trail_length = 500
            self.save = False

    def init_live_plot(self, wind_field):

            #Initialize live plot

            self.fig = plt.figure(figsize=(10,10))

            self.ax = self.fig.add_subplot(111) #create a subplot

            # Starting point is plotted as a diamond
            self.ax.plot(wind_field.position_history_x_orig[0], wind_field.position_history_y_orig[0], 'bD')

            #plot vector field
            xmat = wind_field.x_matrix
            ymat = wind_field.y_matrix
            plt.quiver(wind_field.x_location, wind_field.y_location, xmat,ymat)

            length = wind_field.length
            x_lim_min = wind_field.x_location[0,0]-1
            y_lim_min = wind_field.y_location[0,0]-1

            x_lim_max = wind_field.x_location[1,-1]+1
            y_lim_max = wind_field.y_location[-1,0]+1

            plt.xticks(np.arange(0, length*(wind_field.matsizex), length))
            plt.yticks(np.arange(0,length*(wind_field.matsizey), length))

            plt.xlim([x_lim_min, x_lim_max])
            plt.ylim([y_lim_min, y_lim_max])

            plt.xlabel('X')
            plt.ylabel('Y')
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
                    self.line_list[j].set_data(posx[j,0:time], posy[j,0:time])
                else:
                    self.line_list[j].set_data(posx[j,0:time][-self.trail_length:], posy[j,0:time][-self.trail_length:])

            self.fig.canvas.draw()
            self.fig.canvas.flush_events()

            self.timesteps = time

            if self.save == True:
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
