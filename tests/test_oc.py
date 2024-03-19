from control_ablations.generic_targets import OptimalControlCollocation
from control_ablations.ablation_infra import RunSettings, CEPATrialSettings
from plant_disease_model.experiments import build_cull_or_thin_grid
from plant_disease_model.control \
    import PlantNetworkOCTarget, PlantNetworkMPCTarget, PlantNetworkTargetSettings
from plant_disease_model.simulator \
    import get_constant_kernel, make_ct_env, CullOrThinEnvParams, build_scatter_network
from plant_disease_model.control import ControllerPlotter

import numpy as np
from matplotlib import pyplot as plt


def get_uopt_from_target_with_scaling(scaling, plot=False, direct_rates=False):
    target_settings = build_cull_or_thin_grid(x_nodes=1, y_nodes=2)
    print(target_settings.sim_setup)
    target_settings.set_display_name_addendum(f"test_scaling_{scaling}")
    target_settings.controller = {"type": "casadi_oc",
                                  "settings": {"host_scaling": scaling,
                                               "direct_rates": direct_rates}}
    run_settings = RunSettings()
    run_settings.add_training(iterations=0)
    run_settings.add_eval(iterations=0)
    ts = CEPATrialSettings("test_oc_from_target", run_settings, target_settings)
    target = PlantNetworkOCTarget(ts)

    casadi_inputs = target.generate_dynamics_from_sim_setup()
    opt = OptimalControlCollocation(**casadi_inputs)
    success, u_opt, x_opt = opt.run()
    if direct_rates:
        u_opt = u_opt / scaling
    if plot:
        state_line_styles = ['--', '.'] * 2
        control_line_styles = ['-.'] * 4
        opt.plot(state_line_styles, control_line_styles)
    assert success
    return u_opt


def test_cull_vs_thin_optimal_control_target_scaling():
    uopt_1 = get_uopt_from_target_with_scaling(1.0)
    uopt_0_8 = get_uopt_from_target_with_scaling(0.8)
    assert(np.isclose(uopt_1, uopt_0_8).all())


def test_cull_vs_thin_optimal_control_target_scaling_direct():
    uopt_1 = get_uopt_from_target_with_scaling(1.0, direct_rates=True)
    uopt_0_8 = get_uopt_from_target_with_scaling(0.8, direct_rates=True)
    assert(np.isclose(uopt_1, uopt_0_8, atol=0.02).all())


def get_uopt_from_target_with_ode_resolution_multiplier(multiplier):
    target_settings = build_cull_or_thin_grid(x_nodes=1, y_nodes=2)
    print(target_settings.sim_setup)
    target_settings.set_display_name_addendum("test")
    target_settings.controller = {"type": "casadi_oc", "settings": {}}
    run_settings = RunSettings()
    run_settings.add_training(iterations=0)
    run_settings.add_eval(iterations=0)
    ts = CEPATrialSettings("test_oc_from_target", run_settings, target_settings)
    target = PlantNetworkOCTarget(ts)

    casadi_inputs = target.generate_dynamics_from_sim_setup()
    casadi_inputs["ode_resolution_multiplier"] = multiplier
    opt = OptimalControlCollocation(**casadi_inputs)
    success, u_opt, x_opt = opt.run()
    assert success
    return u_opt


def test_different_control_and_ode_resolutions():
    uopt_1 = get_uopt_from_target_with_ode_resolution_multiplier(1)
    uopt_2 = get_uopt_from_target_with_ode_resolution_multiplier(2)
    assert(np.isclose(uopt_1, uopt_2, atol=0.02).all())


def generate_multi_node_as_sim_setup(number_of_nodes, node_population=1.0, cull_vs_thin=True):
    # Currently only supporting cull vs thin.
    assert cull_vs_thin
    initial_infected_vector = np.zeros(number_of_nodes)
    initial_infected_vector[0] = 0.8 * node_population
    node_setups = []
    node_locations = {}
    # All nodes collocated.
    x_vec = np.zeros(number_of_nodes)
    y_vec = np.zeros(number_of_nodes)
    # All nodes with same population
    n_vec = np.ones(number_of_nodes) * node_population
    for node in range(number_of_nodes):
        node_locations[node] = {'x': float(x_vec[node]), 'y': float(y_vec[node])}
        assert (initial_infected_vector[node] <= n_vec[node])

        settings = {'n': n_vec[node],
                    'beta': float(1.0 / n_vec[node]),
                    'gamma': 0.2,
                    'initial_infected': initial_infected_vector[node]
                    }
        node_setups.append(settings)
    kernel = get_constant_kernel(1.0).name

    setup = {'node_setups': node_setups,
             'node_locations': node_locations,
             'link_setups': [],
             'aerial_setups': {'kernel': kernel, 'beta': 0.25 / node_population},
             'initial_infected_random': 0}

    # Scale cull cost to be equivalent to 5 of each action over 200 hosts.
    # Thin cost and cull cost equal.
    cull_cost = 500
    thin_cost = cull_cost

    env_params = CullOrThinEnvParams(integral_reward=True,
                                     cull_cost=cull_cost,
                                     thin_cost=thin_cost)

    controller = {"type": "casadi_oc",
                  "settings": {'host_scaling': 1.0}}
    target_settings = PlantNetworkTargetSettings(env_params,
                                                 setup,
                                                 controller,
                                                 make_ct_env,
                                                 "simple cull or thin")
    return target_settings


cull_vs_thin_ramp_up_nodes = 4


def test_multi_node_via_target():
    print("Run the multinode system entirely within the target")
    target_settings = generate_multi_node_as_sim_setup(number_of_nodes=cull_vs_thin_ramp_up_nodes)
    target_settings.set_display_name_addendum("multi_node_via_target")
    run_settings = RunSettings()
    run_settings.add_training(iterations=0)
    ts = CEPATrialSettings("test_oc_cull_vs_thin", run_settings, target_settings)
    target = PlantNetworkOCTarget(ts)
    target.run()


# Aiming to generate dynamics of similar magnitudes to the redwood creek work.
#     - Timestep - T = 7 weeks. (N/A) 4 timesteps a year
#     - Beta internal - 0.00506 DONE
#     - Beta external  - 0.00506 * normalisation. (To normalise…)
#     - N - max 100 per 250m square (0.0625km^2). Assume 10 per small square so 840 per large square.
#     - Reward - sum of S hosts. DONE
#     - Gamma - 0 (no R) DONE.
#     - Kernel. Exponential normalised. Sigma of 1310
#     - budget. 1.0 with control rate of 50km^2 per year. = ~10 squares per year over 120 cells.
#       -> 1/12 per year per square = 1/48 per timestep per square (0.02)
def build_bussell_equivalent(number_of_nodes=10, seed=203957):
    initial_infected_vector = np.zeros(number_of_nodes)
    n_max = 800
    n_average = n_max / 2
    initial_infected_vector[0] = 200
    beta = 0.00506 * n_average
    setup, total_population = build_scatter_network(number_of_nodes,
                                                    n_max,
                                                    seed,
                                                    scaled_beta=beta,
                                                    beta_aerial=beta/4/n_average,  # No normalisation so nudging down.
                                                    gamma=0,
                                                    initial_infected_random=0,
                                                    initial_infected_vector=initial_infected_vector)

    cull_rate = 0.02 * number_of_nodes
    cull_cost = 100/cull_rate
    thin_cost = cull_cost

    env_params = CullOrThinEnvParams(integral_reward=True,
                                     cull_cost=cull_cost,
                                     thin_cost=thin_cost)
    controller = {"type": "casadi_oc",
                  "settings": {'host_scaling': 1.0}}
    target_settings = PlantNetworkTargetSettings(env_params,
                                                 setup,
                                                 controller,
                                                 make_ct_env,
                                                 "redwood")
    return target_settings


def test_bussell_matching_system():
    target_settings = build_bussell_equivalent(number_of_nodes=2)

    # Running without settings makes the target generate the dynamics from
    # the env settings.
    target_settings.set_display_name_addendum("test")
    run_settings = RunSettings()
    run_settings.add_training(iterations=0)
    ts = CEPATrialSettings("test_redwood", run_settings, target_settings)
    target = PlantNetworkOCTarget(ts)
    target.run()


def test_target_collocation_constraints():
    num_nodes = 9
    control_timesteps = 20
    ode_multiplier = 6
    timesteps = control_timesteps * ode_multiplier
    target_settings = build_cull_or_thin_grid(x_nodes=3, y_nodes=3)

    # Running without settings makes the target generate the dynamics from
    # the env settings.
    target_settings.controller = {"type": "casadi_oc",
                                  "settings": {'host_scaling': 1.0}}
    target_settings.set_display_name_addendum("test")
    run_settings = RunSettings()
    run_settings.add_training(iterations=0)
    ts = CEPATrialSettings("test_oc_cull_vs_thin", run_settings, target_settings)
    target = PlantNetworkOCTarget(ts)
    casadi_settings = target.generate_dynamics_from_sim_setup()
    opt = OptimalControlCollocation(**casadi_settings,
                                    polynomial_degree=1)

    # opt.setup_collocation()
    opt.generate_midpoint_nlp()
    # "What can we wobble about to optimise?" 2 control variable per node * 120 timesteps = 2400
    control_variables = num_nodes * 2 * control_timesteps

    # Midpoint combines collocation and continuity...
    # 1 collocation point for 2 states per node at 120 timesteps
    # These are bundled in vectors of num_nodes * 2
    # collocation_points_vector = timesteps
    # collocation_points = timesteps + num_nodes * 2
    collocation_points_vector = 0
    collocation_points = 0

    # 1 continuity point for 2 variables per node at 120 timesteps
    # These are bundled in vectors of num_nodes * 2
    continuity_points_vector = timesteps
    continuity_points = timesteps * num_nodes * 2

    assert(len(opt.optimiser_input_variables_per_timestep)
           == control_variables + collocation_points_vector + continuity_points_vector)
    # 1 reward
    assert(opt.reward_accumulator.shape == (1, 1))
    # bounds as per inputs...
    assert(len(opt.optimiser_input_lower_bounds_per_timestep)
           == control_variables + continuity_points + collocation_points)
    # budget constraint per timestep
    budget_constraints = control_timesteps
    # collocations per timestep
    # continuity per timestep Last continuity is questionable but means the last state can't just go awol...
    assert(len(opt.g) == collocation_points_vector + continuity_points_vector + budget_constraints)


def test_oc_matching_system_tiny():
    target_settings = build_cull_or_thin_grid(x_nodes=1, y_nodes=2)

    # Running without settings makes the target generate the dynamics from
    # the env settings.
    target_settings.controller = {"type": "casadi_oc",
                                  "settings": {'host_scaling': 1.0}}
    target_settings.set_display_name_addendum("test")
    run_settings = RunSettings()
    run_settings.add_training(iterations=0)
    ts = CEPATrialSettings("test_oc_cull_vs_thin", run_settings, target_settings)
    target = PlantNetworkOCTarget(ts)
    target.run()


def test_oc_matching_system():
    target_settings = build_cull_or_thin_grid(x_nodes=1, y_nodes=2)

    # Running without settings makes the target generate the dynamics from
    # the env settings.
    target_settings.controller = {"type": "casadi_oc",
                                  "settings": {'host_scaling': 1.0}}
    target_settings.set_display_name_addendum("test")
    run_settings = RunSettings()
    run_settings.add_training(iterations=0)
    # run_settings.add_eval(iterations=0)
    # run_settings.add_plotting(iterations=0)
    ts = CEPATrialSettings("test_oc_cull_vs_thin", run_settings, target_settings)
    target = PlantNetworkOCTarget(ts)
    target.run()


def test_oc_grid_system():
    target_settings = build_cull_or_thin_grid(x_nodes=2,
                                              y_nodes=2)

    # Running without settings makes the target generate the dynamics from
    # the env settings.
    target_settings.controller = {"type": "casadi_oc",
                                  "settings": {'host_scaling': 1.0}}
    target_settings.set_display_name_addendum("test")
    run_settings = RunSettings()
    run_settings.add_training(iterations=0)
    ts = CEPATrialSettings("test_oc_cull_vs_thin", run_settings, target_settings)
    target = PlantNetworkOCTarget(ts)
    target.run()


def test_integral_and_constrained_trajectories_are_similar(display=False):
    target_settings = build_cull_or_thin_grid(x_nodes=1,
                                              y_nodes=2)
    target_settings.controller = {"type": "casadi_oc",
                                  "settings": {"final_reward_only": False}
                                  }
    target_settings.run_limits = {"tuning_valid": True,
                                  "training_valid": True,
                                  "multiple_evals": False}

    target_settings.set_display_name_addendum("test")
    run_settings = RunSettings()
    run_settings.add_training(iterations=0)
    ts = CEPATrialSettings("test_oc_cull_vs_thin", run_settings, target_settings)
    target = PlantNetworkOCTarget(ts)
    casadi_settings = target.generate_dynamics_from_sim_setup()
    oc = OptimalControlCollocation(**casadi_settings,
                                   initialisation_plot_path=target.io.optimiser_init_plot_path())

    # Check output and stop if optimiser has not converged.
    success, optimal_control, state = oc.run()

    if not success:
        assert 0

    integration_resolution = 6
    state_trajectory_oc, controls_oc = oc.get_integrated_trajectories(optimal_control.transpose(),
                                                                      integration_resolution)
    # Check generated controls meet constraints.
    state_array = np.array(state_trajectory_oc)[:, :, 0]

    # Plot integrated states against constrained states
    plt.cla()
    for per_state, per_state_integrated, idx in zip(state, state_array, range(len(state))):
        plt.plot(per_state[::6], label=f"constrained_{idx}")
        plt.plot(per_state_integrated[::integration_resolution], label=f"integral_{idx}")
    plt.legend()
    if display:
        plt.show()

    for index, control_point in enumerate(np.array(controls_oc).transpose()):
        matching_state = state_array[:, index * integration_resolution]
        oc.check_control_against_constraints(control_point, matching_state)

    assert (np.isclose(per_state, per_state_integrated, atol=4e-5).all())


def test_oc_grid_system_cant_be_easily_beat(display=False):
    target_settings = build_cull_or_thin_grid(x_nodes=2,
                                              y_nodes=2)
    target_settings.controller = {"type": "casadi_oc",
                                  "settings": {"final_reward_only": False}
                                  }
    target_settings.run_limits = {"tuning_valid": True,
                                  "training_valid": True,
                                  "multiple_evals": False}

    target_settings.set_display_name_addendum("test")
    run_settings = RunSettings()
    run_settings.add_training(iterations=0)
    ts = CEPATrialSettings("test_oc_cull_vs_thin", run_settings, target_settings)
    target = PlantNetworkOCTarget(ts)
    casadi_settings = target.generate_dynamics_from_sim_setup()
    oc = OptimalControlCollocation(**casadi_settings,
                                   initialisation_plot_path=target.io.optimiser_init_plot_path())

    # Check output and stop if optimiser has not converged.
    success, optimal_control, state = oc.run()

    if not success:
        assert 0

    integration_resolution = 6
    state_trajectory_oc, controls_oc = oc.get_integrated_trajectories(optimal_control.transpose(),
                                                                      integration_resolution)
    # Check generated controls meet constraints.
    state_array = np.array(state_trajectory_oc)[:, :, 0]

    for index, control_point in enumerate(np.array(controls_oc).transpose()):
        matching_state = state_array[:, index * integration_resolution]
        oc.check_control_against_constraints(control_point, matching_state)

    perturbed_controls = np.zeros_like(optimal_control)
    perturbed_controls[0::2] = optimal_control[0::2]
    state_trajectory_pet, controls_pet = oc.get_integrated_trajectories(perturbed_controls.transpose(),
                                                                        oc.ode_resolution_multiplier)

    # Check perturbed controls meet constraints.
    perturbed_state_array = np.array(state_trajectory_pet)[:, :, 0]
    for index, control_point in enumerate(np.array(controls_pet).transpose()):
        if index != 0:  # skip the nans
            matching_state = perturbed_state_array[:, index * 6]
            oc.check_control_against_constraints(control_point, matching_state)

    # Plot oc states against perturbed states
    plt.cla()
    states_per_node = 2
    for per_state_perturbed, per_state_oc, idx in zip(perturbed_state_array, state_array, range(len(state))):
        node = int(idx / states_per_node)
        if idx % states_per_node == 0:
            label = "S"
        else:
            label = "I"
        plt.plot(per_state_perturbed[::integration_resolution], label=f"perturbed_node_{label}{node}")
        plt.plot(per_state_oc[::integration_resolution], label=f"OC_node_{label}{node}")
    plt.legend()
    plt.title("States")
    plt.figure(2)
    for per_control_perturbed, per_control_oc, idx in zip(perturbed_controls,
                                                          optimal_control,
                                                          range(len(optimal_control))):
        plt.plot(per_control_perturbed, ".", label=f"perturbed_{idx}")
        plt.plot(per_control_oc, "-", label=f"OC_{idx}")
    plt.title("Controls")
    plt.legend()
    if display:
        plt.show()

    s_trajectory = np.array(state_trajectory_oc)[0::2, :, 0]
    s_trajectory_sum = s_trajectory.sum()
    perturbed = np.array(state_trajectory_pet)[0::2, :, 0]
    assert(perturbed.shape == s_trajectory.shape)
    perturbed_sum = perturbed.sum()
    assert(s_trajectory_sum > perturbed_sum)


def test_optimal_control_mostly_uses_budget():
    target_settings = build_cull_or_thin_grid(x_nodes=2,
                                              y_nodes=2)
    target_settings.controller = {"type": "casadi_oc",
                                  "settings": {"final_reward_only": False,
                                               # "direct_rates": True
                                               }
                                  }
    target_settings.run_limits = {"tuning_valid": True,
                                  "training_valid": True,
                                  "multiple_evals": False}

    target_settings.set_display_name_addendum("test")
    run_settings = RunSettings()
    run_settings.add_training(iterations=0)
    ts = CEPATrialSettings("test_oc_cull_vs_thin", run_settings, target_settings)
    target = PlantNetworkOCTarget(ts)
    casadi_settings = target.generate_dynamics_from_sim_setup()
    oc = OptimalControlCollocation(**casadi_settings,
                                   initialisation_plot_path=target.io.optimiser_init_plot_path())

    # Check output and stop if optimiser has not converged.
    success, optimal_control, state = oc.run()
    if not success:
        assert 0
    # oc.plot(state_line_type=["."]*4, control_line_type=["-"]*4)
    cull_cost = target.env_params.cull_cost / 100
    thin_cost = target.env_params.thin_cost / 100
    for idx, timestep in enumerate(optimal_control.transpose()):
        state_t = state[:, idx * oc.get_ode_resolution_multiplier()]
        s_vals = state_t[::2]
        i_vals = state_t[1::2]
        cull_vals = timestep[::2]
        thin_vals = timestep[1::2]
        spend = s_vals * thin_vals * thin_cost + i_vals * cull_vals * cull_cost
        assert spend.sum() > 0.95 or np.isclose(cull_vals, 1.0).all()
        assert spend.sum() < 1.01

    optimal_control_converted = target.convert_to_env_actions(optimal_control, state)

    for idx, timestep in enumerate(optimal_control_converted.transpose()):
        timestep_spend = timestep.sum() * target_settings.env_params.cull_cost
        assert (timestep_spend < 101)
        # Either the optimal control should be spending as much as possible
        # or it should be culling all the Is it thinks there are.
        culls = timestep[::2]
        state_i = state[1::2, idx * oc.get_ode_resolution_multiplier()] / target.host_scaling
        remaining_i = (state_i - culls) > 2
        assert (timestep_spend > 95) or not remaining_i.any()


def test_optimal_control_mostly_uses_budget_direct():
    target_settings = build_cull_or_thin_grid(x_nodes=2,
                                              y_nodes=2)
    target_settings.controller = {"type": "casadi_oc",
                                  "settings": {"final_reward_only": False,
                                               "direct_rates": True
                                               }
                                  }
    target_settings.run_limits = {"tuning_valid": True,
                                  "training_valid": True,
                                  "multiple_evals": False}
    target_settings.set_display_name_addendum("test")
    run_settings = RunSettings()
    run_settings.add_training(iterations=0)
    ts = CEPATrialSettings("test_oc_cull_vs_thin", run_settings, target_settings)
    target = PlantNetworkOCTarget(ts)

    casadi_settings = target.generate_dynamics_from_sim_setup()
    oc = OptimalControlCollocation(**casadi_settings,
                                   initialisation_plot_path=target.io.optimiser_init_plot_path())

    # Check output and stop if optimiser has not converged.
    success, optimal_control, state = oc.run()
    if not success:
        assert 0
    # oc.plot(state_line_type=["."]*4, control_line_type=["-"]*4)
    cull_cost = target.env_params.cull_cost / 100
    thin_cost = target.env_params.thin_cost / 100
    for idx, timestep in enumerate(optimal_control.transpose()):
        state_t = state[:, idx * oc.get_ode_resolution_multiplier()]
        i_vals = state_t[1::2]
        cull_vals = timestep[::2]
        thin_vals = timestep[1::2]
        spend = thin_vals * thin_cost + cull_vals * cull_cost
        assert spend.sum() > 0.95 or np.logical_or(cull_vals > i_vals, np.isclose(cull_vals, i_vals)).all()
        assert spend.sum() < 1.01

    optimal_control_converted = target.convert_to_env_actions(optimal_control, state)

    for idx, timestep in enumerate(optimal_control_converted.transpose()):
        timestep_spend = timestep.sum() * target_settings.env_params.cull_cost
        assert (timestep_spend < 101)
        # Either the optimal control should be spending as much as possible
        # or it should be culling all the Is it thinks there are.
        culls = timestep[::2]
        state_i = state[1::2, idx * oc.get_ode_resolution_multiplier()] / target.host_scaling
        remaining_i = (state_i - culls) > 2
        assert (timestep_spend > 95) or not remaining_i.any()


def test_mpc_basic():
    target_settings = build_cull_or_thin_grid(x_nodes=2,
                                              y_nodes=2,
                                              cost_scaling=2.0)

    # Running without settings makes the target generate the dynamics from
    # the env settings.
    target_settings.controller = {"type": "casadi_mpc",
                                  "settings": {'host_scaling': 1.0}}
    target_settings.set_display_name_addendum("test")
    run_settings = RunSettings()
    run_settings.add_training(iterations=0)
    ts = CEPATrialSettings("test_mpc_cull_vs_thin", run_settings, target_settings)
    target = PlantNetworkOCTarget(ts)
    target.run()


def test_mpc_beats_oc():
    target = run_for_comparison(mpc=True, cost_scaling=2.0)
    # Get performance
    plotter = ControllerPlotter(target.test_name, target.plot_display_name, 0)
    rewards_mpc = plotter.read_reward_file()

    # Run OC
    target = run_for_comparison(mpc=False, cost_scaling=2.0)
    # Get performance
    plotter = ControllerPlotter(target.test_name, target.plot_display_name, 0)
    rewards_oc = plotter.read_reward_file()

    assert np.abs(np.array(rewards_mpc).sum()) > np.abs(np.array(rewards_oc).sum())


def run_for_comparison(mpc=True, cost_scaling=4.0):
    # Run MPC
    target_settings = build_cull_or_thin_grid(x_nodes=2,
                                              y_nodes=2,
                                              cost_scaling=cost_scaling,
                                              seed=10101)
    target_settings.run_limits["training_valid"] = True
    if mpc:
        target_settings.controller = {"type": "casadi_mpc",
                                      "settings": {'host_scaling': 1.0,
                                                   "final_reward_only": False}}
        target_settings.set_display_name_addendum("test_mpc")
        test_name = "test_mpc_cull_vs_thin"
    else:
        target_settings.controller = {"type": "casadi_oc",
                                      "settings": {'host_scaling': 1.0,
                                                   "final_reward_only": False}}
        target_settings.set_display_name_addendum("test_oc")
        test_name = "test_oc_cull_vs_thin"
    run_settings = RunSettings()
    run_settings.add_training(iterations=0)
    run_settings.add_eval(iterations=0, example_plot_repeats=20)
    run_settings.add_plotting(iterations=0)
    ts = CEPATrialSettings(test_name, run_settings, target_settings)
    if mpc:
        target = PlantNetworkMPCTarget(ts)
    else:
        target = PlantNetworkOCTarget(ts)
    target.run()
    return target


def get_first_actions_for_comparison(mpc=True):
    target = run_for_comparison(mpc=mpc)
    # Get first actions for 5 tests.
    plotter = ControllerPlotter(target.test_name, target.plot_display_name, 0)
    action_df = plotter.get_action_df()
    actions_at_0 = action_df.loc[action_df['t'] == 0]
    actions_at_0 = actions_at_0.loc[np.logical_not(np.isnan(actions_at_0['cull']))]
    full_data = np.zeros((20, 4))
    for iteration in range(20):
        full_data[iteration] = actions_at_0.loc[actions_at_0['test_iteration'] == iteration]['cull'].values
    assert (full_data == full_data[0]).all()
    return full_data


def test_mpc_and_oc_have_same_first_actions():
    # Run 5 MPC and 5 OC runs. All the first actions should be the same.
    mpc_data = get_first_actions_for_comparison(mpc=True)
    oc_data = get_first_actions_for_comparison(mpc=False)

    assert (oc_data == mpc_data[0]).all()
