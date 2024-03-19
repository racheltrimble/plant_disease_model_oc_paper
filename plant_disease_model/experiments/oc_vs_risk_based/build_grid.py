import numpy as np
from plant_disease_model.simulator import build_grid_network, make_ct_env, CullOrThinEnvParams
from plant_disease_model.control import PlantNetworkTargetSettings


fixed_horizon = 20


def build_cull_or_thin_grid(x_nodes=3,
                            y_nodes=3,
                            n_max=200,
                            seed=203957,
                            cost_scaling=1.0,
                            frequency_dependent_betas=True,
                            random_start=False
                            ):
    number_of_nodes = x_nodes * y_nodes
    initial_infected_seed_population = 4 * number_of_nodes
    initial_infected_vector = np.zeros(number_of_nodes)
    if random_start:
        initial_infected_random = initial_infected_seed_population
    else:
        initial_infected_random = 0
        infected_share_per_node = 20
        # Note this could theoretically specify more infected hosts than exist...
        node = 0
        while initial_infected_seed_population > 0 and node < number_of_nodes:
            initial_infected_vector[node] = infected_share_per_node
            initial_infected_seed_population -= infected_share_per_node
            node += 1
    if frequency_dependent_betas:
        scaled_beta = 1.0
        print("Using beta of 1.0 / N for each node")
    else:
        scaled_beta = 1.0 / n_max * 2
        print(f"Using beta value of {scaled_beta} for every node")
    setup, total_population = build_grid_network(x_nodes=x_nodes,
                                                 y_nodes=y_nodes,
                                                 spacing=1.0,
                                                 biggest_spatial_beta_ratio=0.2,
                                                 n_max=n_max,
                                                 seed=seed,
                                                 initial_infected_random=initial_infected_random,
                                                 initial_infected_vector=initial_infected_vector,
                                                 scaled_beta=scaled_beta,
                                                 gamma=0.2,
                                                 rate_based_control=True,
                                                 frequency_dependent_betas=frequency_dependent_betas)

    culls = total_population / 8
    cull_cost = 100 / culls * cost_scaling
    thin_cost = cull_cost

    env_params = CullOrThinEnvParams(integral_reward=True,
                                     cull_cost=cull_cost,
                                     thin_cost=thin_cost,
                                     fixed_horizon=fixed_horizon,
                                     max_rate_multiplier_per_node=3.0,
                                     env_format="OC")

    controller = {"type": "sensitivity_control",
                  "settings": {}}
    target_settings = PlantNetworkTargetSettings(env_params,
                                                 setup,
                                                 controller,
                                                 make_ct_env,
                                                 "")

    target_settings.run_limits = {"tuning_valid": False,
                                  "training_valid": False,
                                  "multiple_evals": False}

    return target_settings
