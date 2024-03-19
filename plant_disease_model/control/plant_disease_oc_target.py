import numpy as np
import pandas as pd
from casadi import MX

from control_ablations.generic_targets import OptimalControlTarget, MPCOptimalControlTarget
from plant_disease_model.control.plant_disease_target import PlantNetworkPlotter
from plant_disease_model.simulator import get_kernel_from_name
from plant_disease_model.control.pd_oc_io import PDOCIO


class PlantNetworkOCTarget(OptimalControlTarget, PlantNetworkPlotter):
    def __init__(self, trial_settings):
        super().__init__(trial_settings)
        self.io = PDOCIO(self.test_name)
        # Need to set a default for externally supplied dynamics.
        self.host_scaling = 1.0
        # Sets the ratio between action changes (set by problem constraints)
        # and the ODE solver (selected to allow optimal control convergence given the dynamics)
        self.ode_resolution_multiplier = 6
        self.direct_rates = self.controller_settings.get("direct_rates", False)
        # This sets the proportion of the budget that the OC thinks it has available.
        # The rest of the budget is added on to the cull actions to "overdrive" them
        # This helps to account for variability in the stochastic system.
        self.oc_allowance = self.controller_settings.get("oc_allowance", 1.0)
        self.budget = 100
        self.time_horizon = 20
        # Similarly, this sets up the beta values seen by the optimal control to
        # be worse than they actually are.
        self.beta_overestimate = self.controller_settings.get("beta_overestimate", 0.0)
        self.i_lower_bound = self.controller_settings.get("i_lower_bound", 0.0)

    def generate_dynamics_from_sim_setup(self):
        final_reward_only = self.controller_settings.get("final_reward_only", True)
        final_sum_to_infinity = self.controller_settings.get("final_sum_to_infinity", None)
        inverted_i_reward = self.controller_settings.get("inverted_i_reward", None)
        discretised = self.controller_settings.get("discretised", True)
        rate_limit = self.controller_settings.get("rate_limit", 1.0)
        node_setups = self.sim_setup['node_setups']
        node_locations = self.sim_setup['node_locations']
        additional_scaling = self.controller_settings.get("host_scaling", 1.0)
        num_nodes = len(node_setups)
        kernel = get_kernel_from_name(self.sim_setup['aerial_setups']['kernel']).kernel
        raw_node_pops = [node_setup['n'] for node_setup in node_setups]
        self.host_scaling = 1/max(raw_node_pops) * additional_scaling
        print(f"host_scaling is {self.host_scaling}")
        cost_scaling = 100
        print(f"cost_scaling is {cost_scaling}")

        # Generate transmission matrix and node_pops
        betas = np.zeros((num_nodes, num_nodes))
        node_pops = []
        n_max = 0
        for source in range(num_nodes):
            # Note - scaling is applied here so doesn't need repeated through the rest of the code.
            new_node_pop = node_setups[source]['n'] * self.host_scaling
            node_pops += [new_node_pop]
            n_max = max(n_max, new_node_pop)
            beta_multiplier = (1 + self.beta_overestimate) / self.host_scaling
            for dest in range(num_nodes):
                if source == dest:
                    betas[source, dest] = node_setups[source]['beta'] * beta_multiplier
                else:
                    xdiff = node_locations[source]['x'] - node_locations[dest]['x']
                    ydiff = node_locations[source]['y'] - node_locations[dest]['y']
                    distance = np.sqrt(xdiff**2 + ydiff**2)
                    betas[source, dest] = kernel(distance) * self.sim_setup['aerial_setups']['beta'] * beta_multiplier

        # Declare model variables
        s_list = []
        i_list = []
        control_list = []
        control_line_type = []
        state_list = []
        state_line_type = []
        initial_state = []

        for node in range(num_nodes):
            s = MX.sym('S' + str(node))
            i = MX.sym('I' + str(node))
            s_list.append(s)
            i_list.append(i)
            state_list += [s, i]
            state_line_type += ['--', '.']
            control_list.append(MX.sym('cull' + str(node)))
            control_list.append(MX.sym('thin' + str(node)))
            control_line_type += ['-.', 'o']
            node_i0 = node_setups[node]['initial_infected'] * self.host_scaling
            node_s0 = node_pops[node] - node_i0
            initial_state += [node_s0, node_i0]

        # Actions are spend on each action on each node per turn
        # Budgets and costs all normalised to 1.
        budget = self.budget / cost_scaling * self.oc_allowance
        cull_and_thin_per_node = 2
        cull_cost = self.env_params.cull_cost / cost_scaling
        thin_cost = self.env_params.thin_cost / cost_scaling

        print(f"cull cost:{cull_cost}")
        print(f"thin cost:{thin_cost}")

        # Model equations
        equations = []
        objective_per_step_cost = 0
        state_bounds_upper = []
        state_bounds_lower = []
        control_bounds_upper = []
        control_bounds_lower = []
        control_initial_point_per_step = []
        # Express betas in terms of source and destination of pathogen
        for dest_node in range(num_nodes):
            gamma = node_setups[dest_node]['gamma']
            s1 = s_list[dest_node]
            i1 = i_list[dest_node]
            cull = control_list[dest_node * cull_and_thin_per_node]
            thin = control_list[dest_node * cull_and_thin_per_node + 1]
            if self.direct_rates:
                s_to_r = thin
                i_to_r = gamma * i1 + cull
            else:
                s_to_r = thin * s1
                i_to_r = gamma * i1 + cull * i1
            s_to_i = 0
            # 1 is source, 2 is dest
            for source_node in range(num_nodes):
                i2 = i_list[source_node]
                s_to_i += s1 * i2 * betas[source_node, dest_node]
            equations += [-s_to_i - s_to_r, s_to_i - i_to_r]
            # Objective term
            # Problem is framed as a minimisation so want to minimise -S
            i_sum = 0
            if not final_reward_only:
                if inverted_i_reward is not None:
                    i_sum += i1
                else:
                    objective_per_step_cost -= s1

            state_bounds_upper += [np.inf, np.inf]
            state_bounds_lower += [0.0, self.i_lower_bound]
            control_initial_point_per_step += [0.0, 0.0]

            control_bounds_lower += [0.0] * cull_and_thin_per_node
            # No point in controlling more than there is in discretised version of
            # the problem.
            if discretised:
                control_bounds_upper += [1.0] * cull_and_thin_per_node
            else:
                # Note - this is an arbitrary constraint.
                # Can be thought of as "how many sweeps will controllers do within timestep".
                control_bounds_upper += [rate_limit] * cull_and_thin_per_node

        if not final_reward_only and inverted_i_reward is not None:
            epsilon = 0.00001
            objective_per_step_cost -= inverted_i_reward / (i_sum + epsilon) * self.host_scaling

        def inequality_constraint(state, control):
            b_sum = 0
            for node_idx in range(num_nodes):
                local_s = state[node_idx * 2]
                local_i = state[node_idx * 2 + 1]
                local_cull = control[node_idx * cull_and_thin_per_node]
                local_thin = control[node_idx * cull_and_thin_per_node + 1]
                if self.direct_rates:
                    b_sum += local_cull * cull_cost / self.host_scaling
                    b_sum += local_thin * thin_cost / self.host_scaling
                else:
                    b_sum += local_cull * cull_cost * local_i / self.host_scaling
                    b_sum += local_thin * thin_cost * local_s / self.host_scaling
            return [(b_sum, 0, budget)]

        if final_reward_only:
            def final_cost(state):
                cost = 0
                for node_idx in range(num_nodes):
                    local_s = state[node_idx * 2]
                    cost -= local_s
                return cost
        # This is equivalent to the sum to infinity for 0.95 discount factor.
        elif final_sum_to_infinity is not None:
            def final_cost(state):
                cost = 0
                for node_idx in range(num_nodes):
                    local_s = state[node_idx * 2]
                    cost -= local_s
                return cost * final_sum_to_infinity
        else:
            final_cost = None

        control_intervals = 20
        casadi_settings = {'time_horizon': self.time_horizon,
                           'control_intervals': control_intervals,
                           'ode_resolution_multiplier': self.ode_resolution_multiplier,
                           'integrator_type': 'cvodes',
                           'use_midpoint': True,
                           'state_list': state_list,
                           'control_list': control_list,
                           'dynamics_list': equations,
                           'objective': objective_per_step_cost,
                           'initial_state': initial_state,
                           'control_bounds_lower': control_bounds_lower,
                           'control_bounds_upper': control_bounds_upper,
                           'inequality_constraint_function': inequality_constraint,
                           'state_bounds_lower': state_bounds_lower,
                           'state_bounds_upper': state_bounds_upper,
                           'control_initial_trajectories': [control_initial_point_per_step] * control_intervals,
                           'final_cost': final_cost}

        return casadi_settings

    def train(self):
        super().train()
        self.save_df()

    def save_df(self):
        state_df, action_df = self.generate_deterministic_df()
        filename_csv = self.io.deterministic_df_path()
        state_df.to_csv(filename_csv)

        filename_action = self.io.deterministic_action_df_path()
        action_df.to_csv(filename_action)

    # Converts the optimal state trajecory and optimal control trajectory to be the same
    # df format as that generated by the Network object.
    def generate_deterministic_df(self):
        oc_control = self.io.load_trajectory()
        oc_states = self.io.load_state_trajectory()
        state_df = pd.DataFrame()
        action_df = pd.DataFrame()
        num_nodes = len(self.sim_setup['node_setups'])
        for node in range(num_nodes):
            time_list = np.linspace(0, self.time_horizon, oc_states.shape[1])
            s_list = oc_states[node * 2]
            i_list = oc_states[node * 2 + 1]
            r_list = self.sim_setup['node_setups'][node]['n'] - s_list - i_list
            d = {'t': time_list,
                 'nS': s_list,
                 'nI': i_list,
                 'nR': r_list,
                 'test_iteration': len(time_list) * [0],
                 'idx': len(time_list) * [node]
                 }
            new_data = pd.DataFrame.from_dict(d, orient='index').transpose()
            state_df = pd.concat([state_df, new_data])

            action_time_list = np.linspace(0, self.time_horizon, oc_control.shape[1])
            d = {'t': action_time_list,
                 'inspect': len(action_time_list)*[np.nan],
                 'cull': oc_control[2 * node],
                 'thin': oc_control[2 * node + 1],
                 'test_iteration': len(action_time_list)*[0],
                 'idx': len(action_time_list)*[node]
                 }
            new_data = pd.DataFrame.from_dict(d, orient='index').transpose()
            action_df = pd.concat([action_df, new_data])
        return state_df, action_df

    # This is for debug only.
    def convert_to_env_states(self, states):
        return states / self.host_scaling

    def get_valid_controller_settings(self):
        settings = super().get_valid_controller_settings()
        settings += ['host_scaling',
                     'final_reward_only',
                     'final_sum_to_infinity',
                     "inverted_i_reward",
                     'direct_rates',
                     'discretised',
                     'rate_limit',
                     'oc_allowance',
                     'beta_overestimate',
                     'i_lower_bound']
        return settings

    # This function and the one below combine to allow scaling of the action based
    # on the predicted state from the optimal control OR from the state that is
    # currently present in the system.
    # This function is also intended to account for any action scaling due to state
    # scaling input to the optimal control.
    # However, in this case, budget and cost were both scaled. Effect of action defined as
    # proportional to hosts so no descaling required.
    # Optimal control generates actions as a fraction of the S/I population so converting this
    # to number of controls based on the predicted state of the system.
    # Takes the whole optimal control trajectory and state trajectory and scales it.
    def convert_to_env_actions(self, optimal_control, states):
        # num_nodes = len(self.sim_setup['node_setups'])
        s_vector = states[0::2]
        i_vector = states[1::2]
        env_actions = np.zeros_like(optimal_control)
        for action_idx, action_vector in enumerate(optimal_control):
            node = int(action_idx / 2)
            # Cull vector should be multiplied by I
            if action_idx % 2 == 0:
                if self.direct_rates:
                    unscaled_cull = action_vector
                else:
                    unscaled_cull = action_vector * i_vector[node, :-1:self.ode_resolution_multiplier]
                # direct rates includes the host scaling natively. Otherwise, there is host scaling
                # in the multiplying state (I in this case). Hence, both options require scaling.
                # As a first approximation, we reapply the oc_allowance here to scale up the selected actions.
                env_actions[action_idx] = unscaled_cull / self.host_scaling / self.oc_allowance
            if action_idx % 2 == 1:
                if self.direct_rates:
                    unscaled_thin = action_vector
                else:
                    unscaled_thin = action_vector * s_vector[node, :-1:self.ode_resolution_multiplier]
                env_actions[action_idx] = unscaled_thin / self.host_scaling
        # Add extra budget proportionally onto the cull actions
        cull_spend = env_actions[0::2].sum(axis=0) * self.env_params.cull_cost
        thin_spend = env_actions[1::2].sum(axis=0) * self.env_params.thin_cost
        if self.oc_allowance != 1.0:
            culls = env_actions[0::2]
            extra_money = self.budget - cull_spend - thin_spend
            extra_culls = extra_money / self.env_params.cull_cost
            # If the overall spend on culling is basically zero, just spread evenly
            mask = np.isclose(cull_spend, 0, atol=0.01)
            culls[:, mask] = extra_culls[mask] / (len(env_actions) / 2)
            increase_factor = (extra_money + cull_spend) / cull_spend
            culls[:, ~mask] = culls[:, ~mask] * increase_factor[~mask]
            env_actions[0::2] = culls
            env_actions[0::2] = np.maximum(env_actions[0::2], 0)
        return env_actions

    def scale_action(self, action, observation):
        scaled_action = action.astype(np.float32)
        scaled_action = np.maximum(0.0, scaled_action)
        return scaled_action


class PlantNetworkMPCTarget(MPCOptimalControlTarget, PlantNetworkOCTarget):
    def __init__(self, trial_settings):
        super().__init__(trial_settings)
        self.num_nodes = len(self.sim_setup['node_setups'])
        n0_list = [node["n"] for node in self.sim_setup['node_setups']]
        self.n0s = np.array(n0_list)

    def observation_to_initial_state(self, observation):
        # MPC state is arranged as S, I, S, I...
        # observation is all S followed by all I
        initial_state = np.zeros(2 * self.num_nodes)
        scaling = self.host_scaling

        initial_state[::2] = observation[0:self.num_nodes] * scaling
        initial_state[1::2] = observation[self.num_nodes:2*self.num_nodes] * scaling

        return initial_state
