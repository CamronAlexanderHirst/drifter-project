"""
Primary script for defining and running experiments.
"""

from experiment import DrifterExperiment

example_experiment = DrifterExperiment(test_steps=50, length=4, experiment_tag='meow')
example_experiment.run_drifter_experiment()
