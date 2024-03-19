import numpy as np

from plant_disease_model.control import SensitivityBasedControlTarget


class RandomSubPopControl(SensitivityBasedControlTarget):
    def __init__(self, trial_settings):
        super().__init__(trial_settings)
        self.rng = np.random.default_rng(seed=101010)

    def update_priorities(self, _1, _2):
        # Randomly generate the priority list for control
        self.priorities = self.rng.dirichlet([1] * self.num_nodes)
