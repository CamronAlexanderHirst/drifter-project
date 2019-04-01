"""
MDP - Markov Decision Process class for the Soarer-Drifter system
Author: John Jackson
"""

import numpy as np

import matplotlib
matplotlib.use('TkAgg')

import matplotlib.pyplot as plt


class SoarerDrifterMDP:

    def __init__(self):

        self.size = (20, 100)   # Size of the discrete world NxM

        # Discretized control space - (sUAS control, balloon control)
        # sUAS: 0->"turn" left, 1->forward, 2->"turn" right
        # balloon: 0->no release, 1->release
        self.control_space = [[0, 0], [1, 0], [2, 0],
                              [0, 1], [1, 1], [2, 1]]

        self.transition_probability = np.array([[0.75, 0.125, 0.125],
                                                [0.125, 0.75, 0.125],
                                                [0.125, 0.125, 0.75]])

        # sUAS Parameters
        self.x0 = np.array([10, 0])  # Starting cell of the sUAS
        self.x_old = self.x0
        self.state_history = [np.copy(self.x0)]

    def take_action(self, action):

        suas_action = action[0]

        # Get the transition row based on the suas action taken
        transition_row = np.copy(self.transition_probability[suas_action, :])

        # Adjust row based on whether or not we are at the edge
        if self.x_old[0] == 0:
            transition_row[0] = 0.
            transition_row = transition_row / np.sum(transition_row)
        elif self.x_old[0] == (self.size[0] - 1):
            transition_row[2] = 0.
            transition_row = transition_row / np.sum(transition_row)

        # Take action here!
        x_new = self.suas_control(transition_row)
        # Calcualte the reward!
        reward = self.calculate_reward(x_new, action)
        self.x_old = x_new
        self.state_history.append(np.copy(x_new))

        return reward

    def suas_control(self, transition_row):

        result = np.random.multinomial(1, transition_row)
        control_result = np.argmax(result)

        x_new = self.x_old
        x_new[1] += 1

        if control_result == 0:
            x_new[0] -= 1
        elif control_result == 2:
            x_new[0] += 1
        else:
            # Continue level!
            pass

        return x_new

    def calculate_reward(self, x_new, action):

        balloon_action = action[1]

        if balloon_action > 0:
            # progpgate balloon things here
            balloon_reward = 11.
        else:
            balloon_reward = -1.

        # Control action costs
        suas_control_cost = -1.

        total_reward = balloon_reward + suas_control_cost

        return total_reward


if __name__ == '__main__':

        # Quick Test for MDP
        mdp_tester = SoarerDrifterMDP()
        test_steps = 100

        plt.figure()
        plt.xlabel('Forward Direction')
        plt.ylabel('Lateral Direction')
        plt.grid()

        for i in range(1, test_steps):
            # Fly level!
            mdp_tester.take_action([1, 0])
            plt.plot(mdp_tester.state_history[-1][1], mdp_tester.state_history[-1][0], 'kD')

        plt.show()
