"""
Test Script for the MDP
Alex Hirst
"""

import matplotlib
matplotlib.use('TkAgg') #for macs

import matplotlib.pyplot as plt
import logging
import datetime
import pickle

from src.mdp import SoarerDrifterMDP
from src import wind_field
from src import gen_random_field
from src import visualizer
import clear_folder
import time as t

class DrifterExperiment(object):

    def __init__(self, test_steps=100, n=10, m=15, length=3, n_samples=75, experiment_tag=None):

        logFormatter = logging.Formatter('%(asctime)s [%(threadName)-12.12s] [%(levelname)-5.5s]  %(message)s')
        self.logger = logging.Logger(logFormatter)
        if experiment_tag is not None:
            hdlr = logging.FileHandler(datetime.datetime.now().strftime('results/drifter_mdp_{}.log'.format(experiment_tag)))

        else:
            hdlr = logging.FileHandler(datetime.datetime.now().strftime('results/drifter_mdp_%H_%M_%d_%m_%Y.log'))
        self.experiment_tag = experiment_tag
        self.logger.addHandler(hdlr)
        consoleHandler = logging.StreamHandler()
        consoleHandler.setFormatter(logFormatter)
        self.logger.addHandler(consoleHandler)
        self.logger.setLevel(logging.INFO)

        self.test_steps = test_steps
        self.n = n
        self.m = m
        self.length = length
        self.n_samples = n_samples

    def run_drifter_experiment(self):

        test_steps = self.test_steps
        n = self.n#cell height of field (y)
        m = self.m #cell width of field (x)
        length = self.length #unit width
        n_samples = self.n_samples #number of balloons propagated at each space

        time_start = t.process_time()

        field = gen_random_field.field_generator(n, m, length, 0, 0.15, n_samples, 'Normal')
        field.nrm_mean = 0 #Can use matrix here to specify distributions for each measurement
        field.nrm_sig = 1 #Can use matrix here to specify distributions for each measurement
        field.sample()
        self.logger.info('generated field')

        A = wind_field.wind_field(field.vel, field.loc, length, field.nsamps, field.samples)
        # A.xgoal = 18 # Not currently used?
        # A.ygoal = 18

        #Create MDP object
        mdp = SoarerDrifterMDP(test_steps)
        mdp.set_xgoals([11, 16, 25])
        mdp.ygoal = 18
        mdp.balloon_dt = 0.5
        self.logger.info('X Goals: {}'.format(mdp.xgoals))
        self.logger.info('Y Goals: {}'.format(mdp.ygoal))
        self.logger.info('Balloon dt: {}'.format(mdp.balloon_dt))
        self.logger.info('generated mdp')

        mdp.import_windfield(A) #import windfield
        mdp.import_actionspace([0,42], [0,5]) #import action space [xlimits], [ylimits]
        self.logger.info('imported windfield and actionspace')

        self.logger.info('running mdp simulation...')
        mdp.initial_state = [5, 2, mdp.num_balloons] #initialize state
        mdp.state_history.append(mdp.initial_state)
        state = mdp.initial_state
        horizon = 5
        self.logger.info('Horizon: {}'.format(horizon))
        for i in range(self.m):
            [a_opt, v_opt] = mdp.selectaction(state, horizon)
            state = mdp.take_action(state, a_opt)
            mdp.state_history.append(state)

        time_end = t.process_time()
        duration = time_end - time_start

        self.logger.info('Generation and MDP Simulation Completed in {} s'.format(duration))

        if self.experiment_tag is not None:
            self.logger.info('Saving the MDP for this experiment.')

            filename = 'results/{}.mdp'.format(self.experiment_tag)

            try:
                with open(filename, 'wb') as file:
                    pickle.dump(mdp, file)
                self.logger.info('Successfully saved the MPD.')
            except FileNotFoundError:
                self.logger.error('Could not save the MDP!')


if __name__ == '__main__':

    drifter_experiment1 = DrifterExperiment(experiment_tag='meow')
    drifter_experiment1.run_drifter_experiment()
    # drifter_experiment2 = DrifterExperiment(test_steps=5)
    # drifter_experiment2.run_drifter_experiment()
