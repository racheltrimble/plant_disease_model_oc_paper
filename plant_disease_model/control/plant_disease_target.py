import numpy as np
from control_ablations.ablation_infra import CEPATarget
from control_ablations.ablation_infra import CEPATrialSettings
from control_ablations.generic_targets import FixedControlTarget
from plant_disease_model.control import ControllerPlotter
from plant_disease_model.simulator import GraphFilter


class PlantNetworkPlotter(CEPATarget):
    def __init__(self, trial_settings: CEPATrialSettings):
        super().__init__(trial_settings)
        self.check_settings_are_valid()

    def tune(self):
        raise NotImplementedError()

    def train(self):
        raise NotImplementedError()

    def evaluate(self):
        raise NotImplementedError()

    @staticmethod
    def get_valid_controller_settings():
        raise NotImplementedError

    def plot(self):
        graph_filter_string = self.run_settings.plot_settings.get("graph_filter", None)
        graph_filter = GraphFilter.from_string(graph_filter_string)
        i_only = self.run_settings.plot_settings.get("i_only", True)
        max_iterations = self.run_settings.plot_settings.get("max_display_iterations", None)
        t_max = self.run_settings.plot_settings.get("t_max", None)
        plot_actions = self.run_settings.plot_settings.get("plot_actions", True)
        targets_to_plot = self.run_settings.plot_settings.get("targets_to_plot", None)
        if targets_to_plot is None or self.test_name in targets_to_plot:
            for iteration in self.run_settings.plot_settings["iterations"]:
                plotter = ControllerPlotter(self.test_name, self.plot_display_name, iteration)
                plotter.plot_eval(plot_filter=graph_filter,
                                  display=False,
                                  i_only=i_only,
                                  max_iterations=max_iterations,
                                  plot_actions=plot_actions,
                                  t_max=t_max)


class PlantNetworkFixedControlTarget(FixedControlTarget, PlantNetworkPlotter):

    def translate_action_description_to_action(self, description):
        if description == "equal":
            action = self.get_equal_vector()
        elif description == "equal_per_host":
            action = self.get_equal_per_host_vector()
        elif description == "equal_cull":
            action = self.get_equal_cull_vector()
        elif description == "equal_per_host_cull":
            action = self.get_equal_per_host_cull_vector()
        elif description == "no_action":
            action = self.get_no_action_vector()
        else:
            print("Unsupported action specified for fixed control")
            assert 0
        return action

    def get_no_action_vector(self):
        return np.array([0.0] * self.action_length, np.float32)

    def get_equal_vector(self):
        # This assumes cull vs thin are the same cost.
        action_budget = 100 / self.env_params.cull_cost
        action_for_each_option = action_budget / self.action_length
        return np.array([action_for_each_option] * self.action_length, np.float32)

    def get_equal_cull_vector(self):
        # Split budget equally among cull actions.
        action_budget = 100 / self.env_params.cull_cost
        cull_action = action_budget / self.action_length * 2

        action_vector = np.zeros(self.action_length)
        action_vector[0::2] = cull_action
        # Thin should always be 0.
        action_vector[1::2] = 0

        action_vector = np.array(action_vector, np.float32)
        return action_vector

    def get_equal_per_host_vector(self):
        populations = self.get_population_vector()
        # This assumes cull vs thin are the same cost.
        action_budget = 100 / self.env_params.cull_cost
        action = action_budget * populations / populations.sum()
        return action

    def get_equal_per_host_cull_vector(self):
        action = self.get_equal_per_host_vector()

        # Double the cull and mask the thin actions
        action = action * 2
        action[1::2] = 0.0
        return action

    def get_population_vector(self):
        node_setups = self.sim_setup["node_setups"]
        # This assumes equal number of actions per host and no global actions.
        multiple = int(self.action_length/len(node_setups))
        populations = np.zeros(self.action_length, np.float32)
        for index in range(len(node_setups)):
            populations[index*multiple:index*multiple+multiple] = node_setups[index]["n"]
        return populations

    def get_valid_controller_settings(self):
        settings = super().get_valid_controller_settings()
        return settings + [
                           "n_eval_episodes",
                           "n_evals"
                           ]
