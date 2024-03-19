from plant_disease_model.simulator import Network, get_cauchy_kernel
from plant_disease_model.simulator import Node
from plant_disease_model.simulator.kernels import get_no_spread_kernel
from plant_disease_model.simulator.plotter import *
from plant_disease_model.simulator import Simulator

rng = np.random.default_rng()


# Minimal kernel for test - d converted to _ as unused.


# Basic compile and run check
def test_one_node(graph=False, one_axis=False):
    beta = 5/100
    gamma = 1
    n = 100

    node_setups = ({'n': n,
                    'beta': beta,
                    'gamma': gamma,
                    'initial_infected': 1,
                    'idx': '0'},
                   )

    link_setups = []

    aerial_setups = {'beta': 1,
                     'kernel': get_no_spread_kernel().name}

    node_locations = {0: {'x': 0, 'y': 0}}

    test = Network(node_setups, node_locations, link_setups, aerial_setups)
    sim = Simulator(test, 4, 100)
    sim.run()
    if graph:
        if one_axis:
            resolution = 100
        else:
            resolution = None

        dfs = {index: node.get_df(resolution=resolution) for index, node in enumerate(test.nodes)}
        plot_df(dfs, title="Single node network (beta = 0.05, gamma = 1)", one_axis=one_axis, filename="test_graph.png")

    assert 1


def test_two_nodes(graph=False, one_axis=True):
    beta = 5/100
    gamma = 1
    upstream = 100
    downstream = 100

    node_setups = ({'n': upstream,
                    'beta': beta,
                    'gamma': gamma,
                    'initial_infected': 1,
                    'idx': '0'},
                   {'n': downstream,
                    'beta': beta,
                    'gamma': gamma,
                    'initial_infected': 0,
                    'idx': '1'})
    node_locations = {0: {'x': 0, 'y': 0},
                      1: {'x': -6, 'y': 27}}
    # Define trade rate
    t = 5

    # Format is start, end, rate for each link
    # Start and end are tuples with the first string defining if the label is a node, a source or a sink.
    # The second value is an index into the node, source or sink setup tuple.
    # Network sources and sinks are implied and implemented within the nodes.
    link_setups = ({"start_type": 'source', "start_id": 0, "end_type": 'Node', "end_id": 0, "rate": t},
                   {"start_type": 'Node',   "start_id": 0, "end_type": 'Node', "end_id": 1, "rate": t},
                   {"start_type": 'Node',   "start_id": 1, "end_type": 'sink', "end_id": 0, "rate": t})

    kernel = get_cauchy_kernel(alpha=1, gamma=1).name

    aerial_setups = {'beta': 0.1,
                     'kernel': kernel}

    test = Network(node_setups, node_locations, link_setups, aerial_setups)
    sim = Simulator(test, sim_duration=4, repeats=100)
    sim.run()
    if graph:
        if one_axis:
            resolution = 100
        else:
            resolution = None

        dfs = {index: node.get_df(resolution=resolution) for index, node in enumerate(test.nodes)}
        plot_df(dfs, title="Two node network with aerial and trade", one_axis=one_axis, filename="test_graph.png")

    assert 1


def run_to_df(node, duration=4, repeats=100, resolution=100):
    sim = Simulator(node, sim_duration=duration, repeats=repeats)
    sim.run()
    return node.get_df(resolution=resolution)


def test_one_node_net_compared_to_node(graph=False):
    beta = 5/100
    gamma = 1
    n = 100

    # Run node model
    test_setup = {'n': n,
                  'beta': beta,
                  'gamma': gamma,
                  'initial_infected': 1,
                  'idx': 0}
    test = Node(test_setup, rng)
    node_df = run_to_df(test)

    # Run network model
    node_setups = (test_setup,)
    link_setups = []
    node_locations = {0: {'x': 0, 'y': 0}}
    aerial_setups = {"beta": 0,
                     "kernel": get_no_spread_kernel().name}

    test_net = Network(node_setups, node_locations, link_setups, aerial_setups)
    net_df = run_to_df(test_net)
    if graph:
        plot_df({0: node_df, 1: net_df}, "Comparing single node net to node",
                filter_stochastic_threshold=5,
                filename="test_graph.png")
    compare_results(node_df, net_df, filter_stochastic_threshold=5)


def test_two_node_disconnected_net_compared_to_node(graph=False):
    beta = 5/100
    gamma = 1
    n = 100

    # Run node model
    test_setup = {'n': n,
                  'beta': beta,
                  'gamma': gamma,
                  'initial_infected': 1,
                  'idx': 0}
    test = Node(test_setup, rng)
    node_df = run_to_df(test, repeats=200)

    # Run network model
    node_setups = (test_setup, test_setup)
    link_setups = []
    node_locations = {0: {'x': 0, 'y': 0},
                      1: {'x': 0, 'y': 1}}
    aerial_setups = {"beta": 0,
                     "kernel": get_no_spread_kernel().name}

    test_net = Network(node_setups, node_locations, link_setups, aerial_setups)
    net_df = run_to_df(test_net, repeats=200)
    net_df_split = split_by_node(net_df)
    # Think this should compare the average of the two nodes to the single node.
    if graph:
        plot_df(net_df_split, "Just the two node network", filter_stochastic_threshold=5, filename="test_graph1.png")
        plot_df({0: node_df, 1: net_df_split[0]},
                "Comparing node to two node disconnected net",
                filter_stochastic_threshold=5,
                filename="test_graph2.png")
        plot_df({0: node_df, 1: net_df_split[1]},
                "Comparing node to two node disconnected net",
                filter_stochastic_threshold=5,
                filename="test_graph3.png")
    # Override the node id on node 1 to allow the comparison
    net_df_split[1] = net_df_split[1].assign(idx=0)
    compare_results(node_df, net_df_split[0], filter_stochastic_threshold=5)
    compare_results(node_df, net_df_split[1], filter_stochastic_threshold=5)


def test_aerial_spread_does_something(graph=False):
    beta = 5/100
    gamma = 1
    n = 100

    # Run node model
    test_setup = {'n': n,
                  'beta': beta,
                  'gamma': gamma,
                  'initial_infected': 1,
                  'idx': 0}
    test = Node(test_setup, rng)
    node_df = run_to_df(test, repeats=200)

    # Run network model
    node_setups = (test_setup, test_setup)
    link_setups = []
    node_locations = {0: {'x': 0, 'y': 0},
                      1: {'x': 0, 'y': 1}}
    aerial_setups = {"beta": 1,
                     "kernel": get_cauchy_kernel(alpha=1, gamma=1).name}

    test_net = Network(node_setups, node_locations, link_setups, aerial_setups)
    net_df = run_to_df(test_net, repeats=200)
    net_df0 = net_df.loc[net_df['idx'] == 0]
    compare_results(node_df, net_df0, check_different=True)
    if graph:
        plot_df({0: node_df, 1: net_df0}, "Comparing aerial spread to no aerial spread", filename="test_graph.png")


def test_coupled_net_looks_like_big_node_aerial(graph=False):
    beta = 5/100
    gamma = 1
    n = 100
    repeats = 200
    # Run node model
    test_setup = {'n': 2 * n,
                  'beta': beta,
                  'gamma': gamma,
                  'initial_infected': 2,
                  'idx': 0}
    test = Node(test_setup, rng)
    node_df = run_to_df(test, repeats=repeats)

    # Run network model
    test_setup = {'n': n,
                  'beta': beta,
                  'gamma': gamma,
                  'initial_infected': 1}
    node_setups = (test_setup, test_setup)
    link_setups = []
    node_locations = {0: {'x': 0, 'y': 0},
                      1: {'x': 0, 'y': 0}}
    aerial_setups = {"beta": beta,
                     "kernel": get_cauchy_kernel(alpha=1, gamma=1).name}

    test_net = Network(node_setups, node_locations, link_setups, aerial_setups)
    net_df = run_to_df(test_net, repeats=repeats, resolution=100)
    net_df_summed = sum_over_network(net_df)
    # Override the node id to allow the comparison
    net_df_summed = net_df_summed.assign(idx=0)
    if graph:
        plot_df({0: node_df, 1: net_df_summed},
                "Comparing single large node to nodes linked by aerial spread",
                filename="test_graph.png")
    compare_results(node_df, net_df_summed, filter_stochastic_threshold=5)


def test_coupled_net_looks_like_big_node_trade(graph=False):
    beta = 2.5/100
    gamma = 1
    n = 100
    resolution = 100
    # Run node model
    test_setup = {'n': 2 * n,
                  'beta': beta,
                  'gamma': gamma,
                  'initial_infected': 2,
                  'idx': 0}
    test = Node(test_setup, rng)
    node_df = run_to_df(test, resolution=resolution)

    # Run network model
    # beta is multiplied by two because any one infected host is only spreading to
    # half the population at any one time (even if it mixes around).
    test_setup = {'n': n,
                  'beta': beta*2,
                  'gamma': gamma,
                  'initial_infected': 1}
    node_setups = (test_setup, test_setup)
    # Have very high trade between node 0 and 1.
    link_setups = ({"start_type": 'Node', "start_id": 1, "end_type": 'Node', "end_id": 0, "rate": 100},
                   {"start_type": 'Node', "start_id": 0, "end_type": 'Node', "end_id": 1, "rate": 100})

    node_locations = {0: {'x': 0, 'y': 0},
                      1: {'x': 0, 'y': 0}}
    aerial_setups = {"beta": 0,
                     "kernel": get_no_spread_kernel().name}

    test_net = Network(node_setups, node_locations, link_setups, aerial_setups)
    net_df = run_to_df(test_net, resolution=resolution)
    net_df_summed = sum_over_network(net_df)
    # Override the node id to allow the comparison
    net_df_summed = net_df_summed.assign(idx=0)
    if graph:
        plot_df({0: node_df, 1: net_df_summed},
                "Comparing single large node to nodes linked by aerial spread",
                "test_graph.png")
    compare_results(node_df, net_df_summed, filter_stochastic_threshold=5)


def get_test_net(rate_based_control=False, disease_spreads=True):
    if disease_spreads:
        beta = 5/100
    else:
        beta = 0
    node_setups = ({'n': 100,
                    'beta': beta,
                    'gamma': 1,
                    'initial_infected': 1},
                   {'n': 300,
                    'beta': beta,
                    'gamma': 1,
                    'initial_infected': 10}
                   )
    node_locations = {0: {'x': 0, 'y': 0},
                      1: {'x': 1, 'y': 0}}
    link_setups = []
    aerial_setups = {'kernel': get_cauchy_kernel(1, 1).name, 'beta': beta * 20}
    net = Network(node_setups,
                  node_locations,
                  link_setups,
                  aerial_setups,
                  rate_based_control=rate_based_control)
    net.reset()
    return net


def test_inspect_all_finds_any_infected():
    net = get_test_net()
    result = net.inspect(np.array([100, 300], dtype=np.int32), 0)
    assert((result == [1, 10]).all())


def test_inspect_zero_finds_nothing():
    net = get_test_net()
    result = net.inspect(np.array([0, 0], dtype=np.int32), 0)
    assert((result == [0, 0]).all())


def test_inspect_more_than_there_are_inspects_all():
    net = get_test_net()
    result = net.inspect(np.array([1000, 999], dtype=np.int32), 0)
    assert((result == [1, 10]).all())


def test_treating_all_removes_all():
    net = get_test_net()
    net.treat(np.array([1, 10], dtype=np.int32), time=0)
    result = net.inspect(np.array([1000, 1000], dtype=np.int32), 0)
    assert((result == [0, 0]).all())


def test_treating_some_removes_some():
    net = get_test_net()
    net.treat(np.array([0, 3], dtype=np.int32), time=0)
    net.treat(np.array([1, 3], dtype=np.int32), time=1)
    result = net.inspect(np.array([1000, 1000], dtype=np.int32), time=2)
    assert((result == [0, 4]).all())


def test_treating_none_does_nothing():
    net = get_test_net()
    net.treat(np.array([0, 0], dtype=np.int32), time=0)
    result = net.inspect(np.array([1000, 1000], dtype=np.int32), 0)
    assert((result == [1, 10]).all())


def test_treat_more_than_there_are_inspects_all():
    net = get_test_net()
    net.treat(np.array([1000, 1000], dtype=np.int32), time=0)
    result = net.inspect(np.array([1000, 1000], dtype=np.int32), 0)
    assert((result == [0, 0]).all())


def test_rate_based_control_needs_time():
    net = get_test_net(rate_based_control=True, disease_spreads=False)
    net.treat(np.array([1000, 1000], dtype=np.int32), time=0)
    net.thin(np.array([1000, 1000], dtype=np.int32), time=0)
    result = net.get_s_array()
    assert (result == [99, 290]).all()
    result = net.get_i_array()
    assert (result == [1, 10]).all()
    net.progress(1)
    result = net.get_s_array()
    assert (result < [99, 290]).all()
    result = net.get_i_array()
    assert (result < [1, 10]).all()


def test_random_initial_inspection_adds_infected():
    node_setups = ({'n': 100,
                    'beta': 5 / 100,
                    'gamma': 1,
                    'initial_infected': 1},
                   {'n': 300,
                    'beta': 5 / 100,
                    'gamma': 1,
                    'initial_infected': 10}
                   )
    node_locations = {0: {'x': 0, 'y': 0},
                      1: {'x': 1, 'y': 0}}
    link_setups = []
    aerial_setups = {'kernel': get_cauchy_kernel(1, 1).name, 'beta': 1}
    net = Network(node_setups, node_locations, link_setups, aerial_setups, initial_infected_random=5)
    # Start a few tests test
    net.reset()
    net.stash_results(1)
    net.reset()
    net.stash_results(1)
    net.reset()
    net.stash_results(1)
    net.reset()
    # II should be 1 and 10 (+ 5)
    df = net.get_df()
    i0t0 = df[(df['idx'] == 0) & (df['t'] == 0) & (df['test_iteration'] == 0)]['nI'].item()
    i1t0 = df[(df['idx'] == 1) & (df['t'] == 0) & (df['test_iteration'] == 0)]['nI'].item()
    i0t1 = df[(df['idx'] == 0) & (df['t'] == 0) & (df['test_iteration'] == 1)]['nI'].item()
    i1t1 = df[(df['idx'] == 1) & (df['t'] == 0) & (df['test_iteration'] == 1)]['nI'].item()
    i0t2 = df[(df['idx'] == 0) & (df['t'] == 0) & (df['test_iteration'] == 2)]['nI'].item()
    i1t2 = df[(df['idx'] == 1) & (df['t'] == 0) & (df['test_iteration'] == 2)]['nI'].item()

    # Checks we get the right number of infected and that it's not always the same number (will
    # sometimes randomly fail but ok for sanity)
    assert(i0t0 + i1t0 == 16)
    assert(i0t1 + i1t1 == 16)
    assert(i0t2 + i1t2 == 16)
    assert(not((i0t0 == i0t1) and (i0t0 == i0t2)))


def test_s_i_and_r_matrices_are_consistent():
    node_setups = ({'n': 100,
                    'beta': 5 / 100,
                    'gamma': 1,
                    'initial_infected': 1},
                   {'n': 300,
                    'beta': 5 / 100,
                    'gamma': 1,
                    'initial_infected': 10}
                   )
    node_locations = {0: {'x': 0, 'y': 0},
                      1: {'x': 1, 'y': 0}}
    link_setups = []
    aerial_setups = {"beta": 0,
                     "kernel": get_no_spread_kernel().name}
    net = Network(node_setups, node_locations, link_setups, aerial_setups)
    net.reset()
    assert (net.get_s_array() == np.array([99, 290])).all()
    assert (net.get_i_array() == np.array([1, 10])).all()
    assert (net.get_r_array() == np.array([0, 0])).all()
    totals = np.array([100, 300])
    assert (net.get_total_i() == 11)
    while net.get_total_i() > 0:
        totals_per_node = net.get_s_array() + net.get_i_array() + net.get_r_array()
        assert (totals == totals_per_node).all()
        assert net.get_total_s() == net.get_s_array().sum()
        assert net.get_total_i() == net.get_i_array().sum()
        assert net.get_total_r() == net.get_r_array().sum()
        net.progress(0.2)


def test_thinning_all_removes_all():
    net = get_test_net()
    net.thin(np.array([99, 290], dtype=np.int32), time=0)
    result = net.get_s_array()
    assert((result == [0, 0]).all())


def test_thinning_some_removes_some():
    net = get_test_net()
    net.thin(np.array([0, 3], dtype=np.int32), time=0)
    net.thin(np.array([1, 3], dtype=np.int32), time=1)
    result = net.get_s_array()
    assert((result == [98, 284]).all())


def test_thinning_none_does_nothing():
    net = get_test_net()
    net.thin(np.array([0, 0], dtype=np.int32), time=0)
    result = net.get_s_array()
    assert((result == [99, 290]).all())


def test_thinning_more_than_there_are_thins_all():
    net = get_test_net()
    net.thin(np.array([1000, 1000], dtype=np.int32), time=0)
    result = net.get_s_array()
    assert((result == [0, 0]).all())
