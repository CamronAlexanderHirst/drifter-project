################################################################################
# Figures Script for SciTech 2020
################################################################################

import pandas as pd
import matplotlib
from matplotlib import rcParams
rcParams['font.family'] = 'serif'
rcParams['font.sans-serif'] = ['Times New Roman']
import matplotlib.pyplot as plt
import numpy as np

# Path to the pandas dataframe pickle
datafile = './logs/benchmark/drifter_mdp_23_15_24_11_2019.zip'
dataframe = pd.read_pickle(datafile)
# figure_directory = '/Users/John/Documents/CU Boulder/Conference Papers/sUAS Decision Making/drifter-project/figs/'
figure_directory = './figs/'

solvers = ['TFS', 'SS', 'MCTS']
df_TFS = dataframe[dataframe['Solver'] == 'TFS']
df_SS = dataframe[dataframe['Solver'] == 'SS']
df_MCTS = dataframe[dataframe['Solver'] == 'MCTS']

################################################################################
# Computation Time
################################################################################
fig = plt.figure(figsize=(8, 6), dpi=80, facecolor='w', edgecolor='k')
plt.title("Total Computation Time", fontsize=14)
# plt.xlabel("Solver")
plt.ylabel("Mean Time (s)", fontsize=14)
plt.xlim(0.5,3.5)
ax = fig.add_subplot(111)
ax.yaxis.grid(True, color='lightgray')
ax.set_axisbelow(True)

ax.bar([1, 2, 3],
        [np.mean(df_TFS.CompTime),
         np.mean(df_SS.CompTime),
         np.mean(df_MCTS.CompTime)],
         yerr=[2*np.std(df_TFS.CompTime),
               2*np.std(df_SS.CompTime),
               2*np.std(df_MCTS.CompTime)],
        error_kw=dict(lw=4),
        color='silver',
        width=0.5)

ax.set_xticks([1, 2, 3])
ax.set_xticklabels(['TFS', 'SS', 'MCTS'], fontsize=14)
plt.savefig(figure_directory + 'comptimes.png', dpi=600)

################################################################################
# Reward
################################################################################
fig = plt.figure(figsize=(8, 6), dpi=80, facecolor='w', edgecolor='k')
plt.title("Cumulative Reward", fontsize=14)
# plt.xlabel("Solver")
plt.ylabel("Mean Reward", fontsize=14)
plt.xlim(0.5,3.5)
ax = fig.add_subplot(111)
ax.yaxis.grid(True, color='lightgray')
ax.set_axisbelow(True)

ax.bar([1, 2, 3],
        [np.mean(df_TFS.Reward),
         np.mean(df_SS.Reward),
         np.mean(df_MCTS.Reward)],
         yerr=[2*np.std(df_TFS.Reward),
               2*np.std(df_SS.Reward),
               2*np.std(df_MCTS.Reward)],
        error_kw=dict(lw=4),
        color='silver',
        width=0.5)

ax.set_xticks([1, 2, 3])
ax.set_xticklabels(['TFS', 'SS', 'MCTS'], fontsize=14)
plt.savefig(figure_directory + 'rewards.png', dpi=600)

################################################################################
# Unique Queries
################################################################################
fig = plt.figure(figsize=(8, 6), dpi=80, facecolor='w', edgecolor='k')
plt.title("Unique Balloon Queries", fontsize=14)
# plt.xlabel("Solver")
plt.ylabel("Queries", fontsize=14)
plt.xlim(0.5,3.5)
ax = fig.add_subplot(111)
ax.yaxis.grid(True, color='lightgray')
ax.set_axisbelow(True)

ax.bar([1, 2, 3],
        [np.mean(df_TFS.UniqueQueries),
         np.mean(df_SS.UniqueQueries),
         np.mean(df_MCTS.UniqueQueries)],
         yerr=[2*np.std(df_TFS.UniqueQueries),
               2*np.std(df_SS.UniqueQueries),
               2*np.std(df_MCTS.UniqueQueries)],
        error_kw=dict(lw=4),
        color='silver',
        width=0.5)

ax.set_xticks([1, 2, 3])
ax.set_xticklabels(['TFS', 'SS', 'MCTS'], fontsize=14)
plt.savefig(figure_directory + 'queries.png', dpi=600)

################################################################################
# Balloon Distance
################################################################################
fig = plt.figure(figsize=(8, 6), dpi=80, facecolor='w', edgecolor='k')
plt.title("Balloon Distance From Goal", fontsize=14)
# plt.xlabel("Solver")
plt.ylabel("Mean Distance (m)", fontsize=14)
plt.xlim(0.5,3.5)
ax = fig.add_subplot(111)
ax.yaxis.grid(True, color='lightgray')
ax.set_axisbelow(True)

ax.bar([1, 2, 3],
        [np.mean(df_TFS.BalloonDistance),
         np.mean(df_SS.BalloonDistance),
         np.mean(df_MCTS.BalloonDistance)],
         yerr=[2*np.std(df_TFS.BalloonDistance),
               2*np.std(df_SS.BalloonDistance),
               2*np.std(df_MCTS.BalloonDistance)],
        error_kw=dict(lw=4),
        color='silver',
        width=0.5)

ax.set_xticks([1, 2, 3])
ax.set_xticklabels(['TFS', 'SS', 'MCTS'], fontsize=14)
plt.savefig(figure_directory + 'distance.png', dpi=600)

################################################################################
# Workspace Violations
################################################################################
fig = plt.figure(figsize=(8, 6), dpi=80, facecolor='w', edgecolor='k')
plt.title("Aircraft Workspace Violations", fontsize=14)
# plt.xlabel("Solver")
plt.ylabel("Violations", fontsize=14)
plt.xlim(0.5,3.5)
ax = fig.add_subplot(111)
ax.yaxis.grid(True, color='lightgray')
ax.set_axisbelow(True)

ax.bar([1, 2, 3],
        [np.mean(df_TFS.ACViolations),
         np.mean(df_SS.ACViolations),
         np.mean(df_MCTS.ACViolations)],
         yerr=[2*np.std(df_TFS.ACViolations),
               2*np.std(df_SS.ACViolations),
               2*np.std(df_MCTS.ACViolations)],
        error_kw=dict(lw=4),
        color='silver',
        width=0.5)

ax.set_xticks([1, 2, 3])
ax.set_xticklabels(['TFS', 'SS', 'MCTS'], fontsize=14)
plt.savefig(figure_directory + 'violations.png', dpi=600)
