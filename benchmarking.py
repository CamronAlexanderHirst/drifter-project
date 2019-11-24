#!/usr/bin/python3
"""
Script to run several benchmarking tests on the MDP solvers. The Script
logic is as follows:
- Sample a wind field
- solver the planning problem with each solver using "optimal" tuning Parameters
- Fields to record: time duration, number of unique field propagations, reward,
average distance from average final location of balloons to goal location,
number of instances traveled outside of permitted workspace.

This will be run for 50 iterations. The statistics of the planners will be shown
in plots. The data will be logged in a log file.
"""
import logging
import datetime
from src import mdp as MDP
from src import wind_field
from src import gen_random_field
from src import visualizer
import matplotlib
import clear_folder
import time as t


# logs results and test setup to a log file.
logFormatter = logging.Formatter(
        '%(asctime)s [%(threadName)-12.12s] [%(levelname)-5.5s]  %(message)s')
logger = logging.Logger(logFormatter)
hdlr = logging.FileHandler(datetime.datetime.now().strftime(
        'logs/benchmark/drifter_mdp_%H_%M_%d_%m_%Y.log'))
logger.addHandler(hdlr)
consoleHandler = logging.StreamHandler()
consoleHandler.setFormatter(logFormatter)
logger.addHandler(consoleHandler)
logger.setLevel(logging.INFO)


solver_stats = {}
solvers = ['TFS', 'SS', 'MCTS']
for solver in solvers:
    solver_stats[solver] = []

number_experiments = 1  # number of simulations ran for each solver
logger.info('mdp solver benchmarking tests')
logger.info('Solvers: {}'.format(solvers))
logger.info('Number of test runs: {}'.format(number_experiments))

for i in range(number_experiments):
    logger.info('Test number: {} of {}'.format(i, number_experiments))
    test_steps = 100 # currently unused, number of actions to take?
    n = 10  # # of cells height of field (y)
    m = 25  # # of cells width of field (x)
    length = 5  # cell unit width for field estimate
    n_samples = 75  # number of balloons propagated at each space
    # logger.info('field params: n, m, length, n_samples: {}'.format(n, m, length, n_samples))
    # create a random wind field, this is our estimate of the field.
    field = gen_random_field.field_generator(n, m, length, 0, 0.25, n_samples, 'Normal')

    # speed statistics
    field.nrm_mean_s = 0
    field.nrm_sig_s = 0.25
    # heading statistics
    field.nrm_mean_h = 0
    field.nrm_sig_h = 0.25
    # logger.info('field distribution params: {}'.format(field.nrm_mean_s, field.nrm_sig_s,
    #                                                   field.nrm_mean_h, field.nrm_sig_h) )
    field.sample_heading_speed() # sample n_samples fields with heading and speed stats
    A = wind_field.wind_field(field.vel, field.loc, length, field.nsamps, field.samples)

    x_goals = [20, 30, 40, 50]  # x position of goals
    y_goals = [30, 30, 30, 30]  # y position of goals
    logger.info('x_goals: {}'.format(x_goals))
    logger.info('y_goals: {}'.format(y_goals))
    for solver in solvers:
        logger.info('solver: {}'.format(solver))

        time_start = t.process_time()
        reward = 0  # total reward recieved in this run

        # Create MDP object
        mdp = MDP.SoarerDrifterMDP(test_steps)  # initialize MDP object
        mdp.set_goals(x_goals, y_goals)  # x, y locations of goals
        mdp.balloon_dt = 0.25  # to prop balloon, dt
        mdp.import_windfield(A)  # import windfield
        xlimits = [0, 110]
        ylimits = [0, 10]
        mdp.import_actionspace(xlimits, ylimits)  # import action space [xlimits], [ylimits]
        mdp.initial_state = [10, 0, mdp.num_goals]  # initialize state
        mdp.state_history.append(mdp.initial_state)  # initialize state history

        num_actions = 90  # number of actions for MDP to take
        state = mdp.initial_state  # initialize state

        # setup variables to store the metrics:
        reward = 0  # store reward
        b = 0  # balloon index
        comp_time_b = []  # balloon computation time list
        balloon_stats = []  # balloon final location statistics list
        bd_avg = 0 # mean distance to goal
        num_leave_ws = 0  # number of times A/C leaves ws

        for i in range(num_actions):

            # set up variables to store metrics:


            if solver == 'TFS':
                [a_opt, v_opt] = mdp.selectaction(state, 5)
            elif solver == 'SS':
                [a_opt, v_opt] = mdp.selectaction_SPARCE(state, 5, 2, .95)
            elif solver == 'MCTS':
                a_opt = mdp.selectaction_MCTS(state, 5, 100, .95, 100)


            state = mdp.take_action(state, a_opt)
            mdp.state_history.append(state)


            # Calculate all the metrics!
            if a_opt[1] == 1:
                # balloon comp time:
                time_end = t.process_time()
                comp_time_b.append(time_end - time_start)

                # balloon rewards
                br = mdp.calc_balloon_reward(state[0], state[1], state[2]+1) #balloon reward

                # balloon statistics
                bs = mdp.balloon_statistics(state[0], state[1]) # returns [mu, std]
                balloon_stats.append(bs) # final location stats

                goal_location = mdp.ygoals[mdp.num_goals - (state[2] + 1)]
                avg_dist = abs(goal_location - bs[0])/mdp.num_goals
                bd_avg = bd_avg + avg_dist

            else:
                br = 0

            if (state[1] < ylimits[0]) or (state[1] > ylimits[1]):
                num_leave_ws = num_leave_ws + 1

            ur = mdp.calc_uas_reward(a_opt[0], state[1]) #uas reward
            ar = ur + br # action reward
            reward = reward + ar



        time_end = t.process_time()
        duration = time_end - time_start



        # FOR JOHN: Here are all of the metrics for this experiment
        logger.info('Solver: {}'.format(solver))
        logger.info('reward earned: {}'.format(reward))
        logger.info('total computation time: {}'.format(duration))
        logger.info('computation time for balloons: {}'.format(comp_time_b))
        logger.info('number of unique balloon value queries: {}'.format(mdp.num_unique_queries))
        logger.info('balloons final location stats [mu, std]: {}'.format(balloon_stats))
        logger.info('total average distance to goals: {}'.format(bd_avg))
        logger.info('number of times A/C leaves workspace: {}'.format(num_leave_ws))
