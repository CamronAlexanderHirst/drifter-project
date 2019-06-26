'''
Reward Calculator Class for use in MDP
Author: Alex Hirst
'''


def generator(xlimits, ylimits):
    xmin = xlimits[0]
    xmax = xlimits[1]

    ymin = ylimits[0]
    ymax = ylimits[1]

    actionspace = []

    for x in range(xmax - xmin + 1):
        for y in range(ymax - ymin + 1):
            for b in range(0, 2):  # number of balloons
                actionspace.append([xmin + x, ymin + y, b])

    return actionspace
