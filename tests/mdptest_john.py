"""
Test Script for the MDP
"""

import matplotlib
import matplotlib.pyplot as plt

from src import mdp
matplotlib.use('TkAgg')


test_steps = 100
mdp_process = mdp.SoarerDrifterMDP(test_steps)

plt.figure()
plt.xlabel('Forward Direction')
plt.ylabel('Lateral Direction')
plt.grid()

# Fly Level
for i in range(1, test_steps):
    # Fly level!
    mdp_process.take_action([1, 0])
    plt.plot(mdp_process.state_history[-1][1], mdp_process.state_history[-1][0], 'kD')

mdp_process.reset_mdp()
# Turn Left
for i in range(1, test_steps):
    # Fly level!
    mdp_process.take_action([0, 0])
    plt.plot(mdp_process.state_history[-1][1], mdp_process.state_history[-1][0], 'rD')

mdp_process.reset_mdp()
# Turn Right
for i in range(1, test_steps):
    # Fly level!
    mdp_process.take_action([2, 0])
    plt.plot(mdp_process.state_history[-1][1], mdp_process.state_history[-1][0], 'bD')

plt.show()
