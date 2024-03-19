import numpy as np

from control_ablations.ablation_infra import CEPATrialSettings
from control_ablations.generic_targets import NoLearningControlTarget
from plant_disease_model.simulator import get_kernel_from_name
from plant_disease_model.control.plant_disease_target import PlantNetworkPlotter


class NoLearningPlantControlTarget(NoLearningControlTarget, PlantNetworkPlotter):
    def __init__(self, trial_settings: CEPATrialSettings):
        super().__init__(trial_settings)
        self.num_nodes = len(self.sim_setup["node_setups"])
        self.n0s = self.controller_settings.get("n0s", 1000)

        # transmission between nodes. Indexed as source, dest.
        self.betas = np.zeros((self.num_nodes, self.num_nodes))
        self.gammas = np.zeros(self.num_nodes)

        node_locations = self.sim_setup["node_locations"]
        aerial_setups = self.sim_setup["aerial_setups"]
        kernel = get_kernel_from_name(aerial_setups['kernel'])

        for start_index in range(self.num_nodes):
            self.gammas[start_index] = self.sim_setup["node_setups"][start_index]["gamma"]
            for end_index in range(self.num_nodes):
                # Don't count aerial transmission to yourself.
                if start_index == end_index:
                    self.betas[start_index, end_index] = self.sim_setup["node_setups"][start_index]["beta"]
                else:
                    x_dist = node_locations[start_index]['x'] - node_locations[end_index]['x']
                    y_dist = node_locations[start_index]['y'] - node_locations[end_index]['y']
                    d = np.sqrt(x_dist**2 + y_dist**2)
                    self.betas[start_index, end_index] = aerial_setups['beta'] * kernel.kernel(d)

    def translate_observation_from_vector_to_model(self, obs_vector):
        raw_s = obs_vector[0:self.num_nodes]
        raw_i = obs_vector[self.num_nodes:self.num_nodes * 2]
        return raw_s, raw_i

    def translate_action_from_model_to_vector(self, culls):
        action_vector = np.zeros(2 * self.num_nodes)
        cull_values = culls
        no_thin_value = 0

        action_vector[0::2] = cull_values
        # Thin should always be 0.
        action_vector[1::2] = no_thin_value
        action_vector = np.array(action_vector, np.float32)
        assert(not np.isnan(action_vector).any())

        return action_vector


# For use with fully observable cull vs thin environments.
class SensitivityBasedControlTarget(NoLearningPlantControlTarget, PlantNetworkPlotter):
    def __init__(self, trial_settings: CEPATrialSettings):
        super().__init__(trial_settings)
        self.timestep = 1.0
        self.first_obs = True
        self.priorities = None
        self.rate_based = self.sim_setup["rate_based_control"]
        self.rate_correction = self.controller_settings.get("rate_correction", False)
        self.overshoot_proportion = self.controller_settings.get("overshoot_proportion", 1.0)
        self.recalculate_sensitivities = self.controller_settings.get("recalculate_sensitivities", True)
        self.reversed_order = self.controller_settings.get("reversed_order", False)
        self.use_proportional_not_order = self.controller_settings.get("use_proportional_not_order", False)
        self.n0s = np.zeros(self.num_nodes)
        for idx, node in enumerate(self.sim_setup["node_setups"]):
            self.n0s[idx] = node["n"]

        # Expecting to see S and I for each of the nodes (fully observable)
        stack_stride_per_node = 2
        self.stack_stride = self.num_nodes * stack_stride_per_node

    def reset(self):
        self.first_obs = True

    # Sensitivity based control will use up resources in order of sensitivity
    def get_policy_action(self, observation):
        s_vector, i_vector = self.translate_observation_from_vector_to_model(observation)

        if self.recalculate_sensitivities or self.first_obs:
            self.update_priorities(s_vector, i_vector)
            self.first_obs = False
        if self.use_proportional_not_order:
            out = self.get_proportional_action(s_vector, i_vector)
        else:
            out = self.get_prioritised_action(s_vector, i_vector)
        return out

    def get_prioritised_action(self, s_vector, i_vector):
        # argsort goes from smallest to largest, we want largest to smallest.
        if self.reversed_order:
            sensitivity_order_list = np.argsort(self.priorities)
        else:
            sensitivity_order_list = np.argsort(self.priorities)[::-1]

        cull_budget = 100 / self.env_params.cull_cost
        i_increase_rate = self.get_i_increase_rate(s_vector, i_vector)
        culls = np.zeros(self.num_nodes)
        sensitivity_index = 0
        while cull_budget > 0 and sensitivity_index < self.num_nodes:
            target_node = sensitivity_order_list[sensitivity_index]
            desired_culls_for_target = i_vector[target_node]
            if self.rate_based and self.rate_correction:
                desired_culls_for_target += i_increase_rate[target_node] * self.timestep * self.overshoot_proportion
            culls_for_target = min(cull_budget, desired_culls_for_target)
            culls[target_node] = culls_for_target
            cull_budget -= culls_for_target
            sensitivity_index += 1
        if cull_budget > 0 and culls.sum() > 0:
            # Use up the extra cull budget proportional to the culls already allocated.
            culls += cull_budget * culls / culls.sum()
        out = self.translate_action_from_model_to_vector(culls)
        return out

    def get_proportional_action(self, _, i_vector):
        cull_budget = 100 / self.env_params.cull_cost
        i_exists_mask = i_vector > 0
        masked_priorities = self.priorities * i_exists_mask
        if masked_priorities.sum() == 0:
            culls = masked_priorities
        else:
            culls = masked_priorities / masked_priorities.sum() * cull_budget
        out = self.translate_action_from_model_to_vector(culls)
        return out

    def update_priorities(self, s_vector, _):
        self.priorities = self.get_r0_matrix(s_vector).sum(axis=0)

    def get_r0_matrix(self, s_vector):
        # Think broadcasting is right by default here...
        return (self.betas * s_vector).transpose()

    def get_i_increase_rate(self, s_vector, i_vector):
        # This currently only includes the increase in I. This gives a conservative result as
        # it doesn't include infected hosts that die. A bit of conservatism is good as we want to
        # have good rates of local eradication but this should really be separately tunable.
        i_increase_rate = np.zeros(self.num_nodes)
        for idx in range(self.num_nodes):
            i_increase_rate[idx] = s_vector[idx] * (i_vector * self.betas[:, idx]).sum()
        return i_increase_rate

    def get_valid_controller_settings(self):
        return [
                "rate_correction",
                "recalculate_sensitivities",
                "overshoot_proportion",
                "reversed_order",
                "use_proportional_not_order"
               ]


class PrioritiseByHostClassTarget(SensitivityBasedControlTarget):
    def __init__(self, trial_settings: CEPATrialSettings):
        super().__init__(trial_settings)
        self.use_fraction_rather_than_count = self.controller_settings.get("use_fraction_rather_than_count", False)

    def get_valid_controller_settings(self):
        settings = super().get_valid_controller_settings()
        return settings + ["use_fraction_rather_than_count"]

    def get_priorities_with(self, numerator, denominator):
        if self.use_fraction_rather_than_count:
            self.priorities = np.divide(numerator,
                                        denominator,
                                        out=np.zeros_like(numerator, dtype=np.float64),
                                        where=denominator != 0)
        else:
            self.priorities = numerator


class PrioritiseByITarget(PrioritiseByHostClassTarget):
    def update_priorities(self, s_vector, i_vector):
        self.get_priorities_with(i_vector, s_vector + i_vector)


class PrioritiseBySTarget(PrioritiseByHostClassTarget):
    def update_priorities(self, s_vector, i_vector):
        self.get_priorities_with(s_vector, s_vector + i_vector)