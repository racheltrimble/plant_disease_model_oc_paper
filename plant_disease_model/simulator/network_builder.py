import numpy as np
from plant_disease_model.simulator.kernels import get_power_law_kernel


def setup_location_from_x_y_n_vec(x_vec,
                                  y_vec,
                                  n_vec,
                                  initial_infected_vector,
                                  scaled_beta=None,
                                  gamma=0.2,
                                  frequency_dependent_betas=True):
    number_of_nodes = x_vec.shape[0]
    node_setups = []
    node_locations = {}
    for node in range(number_of_nodes):
        node_locations[node] = {'x': float(x_vec[node]), 'y': float(y_vec[node])}
        assert (initial_infected_vector[node] <= n_vec[node])
        # frequency_dependent_betas determines if the provided beta value is scaled by N (frequency dependent
        # or passed through unchanged (density dependent).
        if frequency_dependent_betas:
            beta = float(scaled_beta / n_vec[node])
        else:
            beta = scaled_beta
        settings = {'n': int(n_vec[node]),
                    'beta': beta,
                    'gamma': gamma,
                    'initial_infected': int(initial_infected_vector[node])
                    }
        node_setups.append(settings)
    return node_setups, node_locations


def calculate_average_connectivity(node_layout):
    total = 0
    for start in node_layout.values():
        for end in node_layout.values():
            d = np.sqrt((start['x'] - end['x'])**2 + (start['y'] - end['y'])**2)
            if d != 0:
                total += 1 / d**3
    num_nodes = len(node_layout)
    return total / num_nodes


def get_aerial_setups(scaled_beta,
                      n_max,
                      smallest_distance,
                      biggest_spatial_beta_ratio,
                      node_layout,
                      frequency_dependent_betas=True):
    kernel_obj = get_power_law_kernel(2)
    kernel = kernel_obj.name
    ratio_for_smallest_distance = kernel_obj.kernel(smallest_distance)
    if frequency_dependent_betas:
        local_beta = scaled_beta/(n_max / 2)
    else:
        local_beta = scaled_beta
    # Scale to ensure that the ratio of internal to external spread is reasonable between
    # closest node pairs.
    spatial_beta = local_beta * biggest_spatial_beta_ratio / ratio_for_smallest_distance

    # Additionally scale for overall levels of transmission in system
    # Want systems with high overall connectivity to have small betas to make behaviours
    # approx equivalent.
    # All normalised to a 2x2 grid.
    scaling_for_number_of_nodes_in_system = calculate_average_connectivity(node_layout)
    scaling_for_2x2_grid = 2.35355
    spatial_beta = float(spatial_beta * scaling_for_2x2_grid / scaling_for_number_of_nodes_in_system)
    aerial_setups = {'kernel': kernel, 'beta': spatial_beta}
    print(f"Spatial beta: {spatial_beta}")
    return aerial_setups


def build_scatter_network(number_of_nodes=10,
                          n_max=200,
                          seed=203957,
                          initial_infected_random=10,
                          initial_infected_vector=None,
                          scaled_beta=1.0,
                          beta_aerial=0.02,
                          gamma=0.2):
    if initial_infected_vector is None:
        initial_infected_vector = np.zeros(number_of_nodes)
    rng = np.random.default_rng(seed=seed)
    x_range = (0, 10)
    y_range = (0, 10)
    # Generate positions and population sizes randomly in the space
    x_vec = rng.uniform(x_range[0], x_range[1], number_of_nodes)
    y_vec = rng.uniform(y_range[0], y_range[1], number_of_nodes)
    n_vec = rng.integers(0, n_max, number_of_nodes)
    node_setups, node_locations = setup_location_from_x_y_n_vec(x_vec,
                                                                y_vec,
                                                                n_vec,
                                                                initial_infected_vector,
                                                                scaled_beta=scaled_beta,
                                                                gamma=gamma)
    kernel = get_power_law_kernel(2).name

    setup = {'node_setups': node_setups,
             'node_locations': node_locations,
             'link_setups': [],
             'aerial_setups': {'kernel': kernel, 'beta': beta_aerial},
             'initial_infected_random': initial_infected_random}
    return setup, float(n_vec.sum())


def build_grid_network(x_nodes=2,
                       y_nodes=2,
                       spacing=1.0,
                       biggest_spatial_beta_ratio=0.1,
                       n_max=200,
                       seed=203957,
                       initial_infected_random=10,
                       initial_infected_vector=None,
                       scaled_beta=1.0,
                       gamma=0.2,
                       rate_based_control=False,
                       frequency_dependent_betas=True):
    number_of_nodes = x_nodes * y_nodes
    if initial_infected_vector is None:
        initial_infected_vector = np.zeros(number_of_nodes)
    rng = np.random.default_rng(seed=seed)

    # Generate positions according to grid and population sizes randomly
    x_vec = np.zeros(number_of_nodes)
    y_vec = np.zeros(number_of_nodes)
    for x_pos in range(x_nodes):
        for y_pos in range(y_nodes):
            x_vec[x_pos * y_nodes + y_pos] = x_pos * spacing
            y_vec[x_pos * y_nodes + y_pos] = y_pos * spacing
    n_vec = rng.integers(0, n_max, number_of_nodes)
    node_setups, node_locations = setup_location_from_x_y_n_vec(x_vec,
                                                                y_vec,
                                                                n_vec,
                                                                initial_infected_vector,
                                                                scaled_beta=scaled_beta,
                                                                gamma=gamma,
                                                                frequency_dependent_betas=frequency_dependent_betas)

    aerial_setups = get_aerial_setups(scaled_beta,
                                      n_max,
                                      spacing,
                                      biggest_spatial_beta_ratio,
                                      node_locations,
                                      frequency_dependent_betas=frequency_dependent_betas)

    setup = {'node_setups': node_setups,
             'node_locations': node_locations,
             'link_setups': [],
             'aerial_setups': aerial_setups,
             'initial_infected_random': initial_infected_random,
             'rate_based_control': rate_based_control}
    return setup, float(n_vec.sum())


def build_a_over_n_by_n_network(number_of_subpops,
                                grid_size_x,
                                grid_size_y,
                                grid_spacing,
                                n_max,
                                seed,
                                initial_infected_vector,
                                rate_based_control=False):
    rng = np.random.default_rng(seed=seed)
    assert number_of_subpops < grid_size_x * grid_size_y

    scaled_beta = 1.0
    biggest_spatial_beta_ratio = 0.1

    # Generate positions according to grid and population sizes randomly
    x_vec = np.zeros(number_of_subpops)
    y_vec = np.zeros(number_of_subpops)
    taken = np.zeros((grid_size_x, grid_size_y))
    for subpop in range(number_of_subpops):
        got_empty_space = False
        while not got_empty_space:
            trial_space_x = rng.integers(0, grid_size_x)
            trial_space_y = rng.integers(0, grid_size_y)
            if not taken[trial_space_x, trial_space_y]:
                got_empty_space = True
                x_vec[subpop] = trial_space_x * grid_spacing
                y_vec[subpop] = trial_space_y * grid_spacing
    n_vec = rng.integers(0, n_max, number_of_subpops)

    node_setups, node_locations = setup_location_from_x_y_n_vec(x_vec,
                                                                y_vec,
                                                                n_vec,
                                                                initial_infected_vector,
                                                                scaled_beta=scaled_beta,
                                                                gamma=0.2)
    aerial_setups = get_aerial_setups(scaled_beta,
                                      n_max,
                                      grid_spacing,
                                      biggest_spatial_beta_ratio,
                                      node_locations)
    setup = {'node_setups': node_setups,
             'node_locations': node_locations,
             'link_setups': [],
             'aerial_setups': aerial_setups,
             'initial_infected_random': 0,
             'rate_based_control': rate_based_control}

    return setup, float(n_vec.sum())


def build_n_clusters_x_by_y_network(number_of_clusters,
                                    grid_size_x,
                                    grid_size_y,
                                    grid_spacing,
                                    radius,
                                    n_max,
                                    seed,
                                    initial_infected_vector,
                                    rate_based_control=False):

    rng = np.random.default_rng(seed=seed)
    number_of_subpops = grid_size_y * grid_size_x * number_of_clusters

    scaled_beta = 1.0
    biggest_spatial_beta_ratio = 0.1
    # Generate positions according to circle
    # Cluster bottom left hand corners sit on circle.
    x_vec = np.zeros(number_of_subpops)
    y_vec = np.zeros(number_of_subpops)
    circle_centre_x = radius
    circle_centre_y = radius

    for cluster in range(number_of_clusters):
        theta = 2 * np.pi * cluster / number_of_clusters
        cluster_centre_x = radius * np.cos(theta) + circle_centre_x
        cluster_centre_y = radius * np.sin(theta) + circle_centre_y
        cluster_offset = cluster * grid_size_y * grid_size_x
        for x_pos in range(grid_size_x):
            x_offset = x_pos * grid_size_y
            for y_pos in range(grid_size_y):
                index = cluster_offset + x_offset + y_pos
                x_vec[index] = cluster_centre_x + x_pos * grid_spacing
                y_vec[index] = cluster_centre_y + y_pos * grid_spacing

    n_vec = rng.integers(0, n_max, number_of_subpops)

    node_setups, node_locations = setup_location_from_x_y_n_vec(x_vec,
                                                                y_vec,
                                                                n_vec,
                                                                initial_infected_vector,
                                                                scaled_beta=scaled_beta,
                                                                gamma=0.2)
    aerial_setups = get_aerial_setups(scaled_beta,
                                      n_max,
                                      grid_spacing,
                                      biggest_spatial_beta_ratio,
                                      node_locations)
    setup = {'node_setups': node_setups,
             'node_locations': node_locations,
             'link_setups': [],
             'aerial_setups': aerial_setups,
             'initial_infected_random': 0,
             'rate_based_control': rate_based_control}

    return setup, float(n_vec.sum())


def build_n_clusters_x_by_y_network_with_bridges(number_of_clusters,
                                                 grid_size_x,
                                                 grid_size_y,
                                                 grid_spacing,
                                                 radius,
                                                 n_max,
                                                 seed,
                                                 initial_infected_vector,
                                                 rate_based_control=False):

    rng = np.random.default_rng(seed=seed)
    number_of_subpops = grid_size_y * grid_size_x * number_of_clusters

    scaled_beta = 1.0
    biggest_spatial_beta_ratio = 0.1
    # Generate positions according to circle
    # Cluster bottom left hand corners sit on circle.
    x_vec = np.zeros(number_of_subpops)
    y_vec = np.zeros(number_of_subpops)
    circle_centre_x = radius
    circle_centre_y = radius

    for cluster in range(number_of_clusters):
        theta = 2 * np.pi * cluster / number_of_clusters
        theta_bridge = 2 * np.pi * (cluster + 0.5) / number_of_clusters
        cluster_centre_x = radius * np.cos(theta) + circle_centre_x
        cluster_centre_y = radius * np.sin(theta) + circle_centre_y
        bridge_position_x = radius * np.cos(theta_bridge) + circle_centre_x
        bridge_position_y = radius * np.sin(theta_bridge) + circle_centre_y
        cluster_offset = cluster * grid_size_y * grid_size_x
        for x_pos in range(grid_size_x):
            x_offset = x_pos * grid_size_y
            for y_pos in range(grid_size_y):
                if not (x_pos == grid_size_x - 1 and y_pos == grid_size_y - 1):
                    index = cluster_offset + x_offset + y_pos
                    x_vec[index] = cluster_centre_x + x_pos * grid_spacing
                    y_vec[index] = cluster_centre_y + y_pos * grid_spacing
        # Add the bridging node
        index = cluster_offset
        index += (grid_size_x - 1) * grid_size_y
        index += (grid_size_y - 1)
        x_vec[index] = bridge_position_x
        y_vec[index] = bridge_position_y

    n_vec = rng.integers(0, n_max, number_of_subpops)

    node_setups, node_locations = setup_location_from_x_y_n_vec(x_vec,
                                                                y_vec,
                                                                n_vec,
                                                                initial_infected_vector,
                                                                scaled_beta=scaled_beta,
                                                                gamma=0.2)
    aerial_setups = get_aerial_setups(scaled_beta,
                                      n_max,
                                      grid_spacing,
                                      biggest_spatial_beta_ratio,
                                      node_locations)
    setup = {'node_setups': node_setups,
             'node_locations': node_locations,
             'link_setups': [],
             'aerial_setups': aerial_setups,
             'initial_infected_random': 0,
             'rate_based_control': rate_based_control}

    return setup, float(n_vec.sum())
