import numpy as np
from plant_disease_model.simulator \
    import CullOrThinEnvOCFormat, get_cauchy_kernel, CullOrThinEnvParams
from stable_baselines3.common.env_checker import check_env


def test_with_stable_baselines_checker():
    setup = {'node_setups': ({'n': 100,
                              'beta': 5/100,
                              'gamma': 1,
                              'initial_infected': 1},
                             ),
             'node_locations': {0: {'x': 0, 'y': 0}},
             'link_setups': [],
             'aerial_setups': {'kernel': get_cauchy_kernel(1, 1).name, 'beta': 1},
             'rate_based_control': True}
    env = CullOrThinEnvOCFormat(setup)
    env.disable_spend_checks()
    check_env(env, warn=True, skip_render_check=True)

    # Copying checking suggestion from here:
    # https://stable-baselines3.readthedocs.io/en/master/guide/rl_tips.html#tips-and-tricks-when-creating-a-custom-environment
    _ = env.reset()
    n_steps = 10
    for _ in range(n_steps):
        # Random action
        action = env.action_space.sample()
        obs, reward, done, info = env.step(action)
        # S and I per node for one node
        assert(len(obs) == 2)
        if done:
            _ = env.reset()


default_setup = {'node_setups': ({'n': 100,
                                  'beta': 5/100,
                                  'gamma': 1,
                                  'initial_infected': 1},
                                 {'n': 100,
                                  'beta': 5 / 100,
                                  'gamma': 1,
                                  'initial_infected': 1},
                                 {'n': 100,
                                  'beta': 5 / 100,
                                  'gamma': 1,
                                  'initial_infected': 1}
                                 ),
                 'node_locations': {0: {'x': 0, 'y': 0},
                                    1: {'x': 0, 'y': 1},
                                    2: {'x': 1, 'y': 0}},
                 'link_setups': [],
                 'aerial_setups': {'kernel': get_cauchy_kernel(1, 1).name, 'beta': 1},
                 'rate_based_control': True}


def test_done_happens():
    env_params = {'integral_reward': True}
    env_params = CullOrThinEnvParams(**env_params)

    env = CullOrThinEnvOCFormat(default_setup, env_params=env_params)
    env.reset()

    done = False
    step = 0
    # Fast epidemic so should be done within 20 steps
    step_check = 20
    while not done:
        observation, reward, done, _ = env.step(np.array([0] * 3 * 2, dtype=np.float32))
        step += 1
        assert (reward <= 0)
        if step > step_check:
            assert False
        else:
            print(step)


def test_done_happens_oc():
    env_params = CullOrThinEnvParams()
    setup = default_setup.copy()
    setup["rate_based_control"] = True

    env = CullOrThinEnvOCFormat(setup, env_params=env_params)
    env.reset()
    # Note - can't run check env on this as it doesn't natively enforce the budget constraints (just checks).

    done = False
    step = 0
    # Fast epidemic so should be done within 20 steps
    step_check = 20
    while not done:
        observation, reward, done, _ = env.step(np.array([0, 1, 0, 1, 1, 0], dtype=np.float32))
        step += 1
        assert (reward <= 0)
        if step > step_check:
            assert False
        else:
            print(step)


def test_nothing_dying_gives_large_negative_reward():
    hosts_per_node = 100
    steps = 10
    # No initial infection so nothing dies
    setup = {'node_setups': ({'n': hosts_per_node,
                              'beta': 5 / 100,
                              'gamma': 1,
                              'initial_infected': 0},
                             {'n': hosts_per_node,
                              'beta': 5 / 100,
                              'gamma': 1,
                              'initial_infected': 0}
                             ),
             'node_locations': {0: {'x': 0, 'y': 0},
                                1: {'x': 0, 'y': 1}},
             'link_setups': [],
             'aerial_setups': {'kernel': get_cauchy_kernel(1, 1).name, 'beta': 1},
             'rate_based_control': True}
    env = CullOrThinEnvOCFormat(setup)
    env.reset()
    total = 0
    # Lots of culling (should have no effect). No thinning
    for n in range(steps):
        observation, reward, done, _ = env.step(np.array([5, 0, 5, 0], dtype=np.float32))
        total += reward
    assert(total == - steps * hosts_per_node * 2)


def test_everything_dying_gives_reward_of_zero():
    # Everything infected so will die eventually
    setup = {'node_setups': ({'n': 100,
                              'beta': 5 / 100,
                              'gamma': 1,
                              'initial_infected': 100},
                             {'n': 100,
                              'beta': 5 / 100,
                              'gamma': 1,
                              'initial_infected': 100}
                             ),
             'node_locations': {0: {'x': 0, 'y': 0},
                                1: {'x': 0, 'y': 1}},
             'link_setups': [],
             'aerial_setups': {'kernel': get_cauchy_kernel(1, 1).name, 'beta': 1},
             'rate_based_control': True}
    env_params = {'integral_reward': False}
    env_params = CullOrThinEnvParams(**env_params)
    env = CullOrThinEnvOCFormat(setup, env_params=env_params)
    env.reset()
    done = False
    total = 0
    # Cull and thin lots in one node and nothing in the other
    while not done:
        observation, reward, done, _ = env.step(np.array([5, 5, 0, 0], dtype=np.float32))
        total += reward
    assert(np.isclose(total, 0))


def test_time_logging_is_aligned():
    # Everything infected so will die eventually
    setup = {'node_setups': ({'n': 100,
                              'beta': 5 / 100,
                              'gamma': 1,
                              'initial_infected': 100},
                             {'n': 100,
                              'beta': 5 / 100,
                              'gamma': 1,
                              'initial_infected': 100}
                             ),
             'node_locations': {0: {'x': 0, 'y': 0},
                                1: {'x': 0, 'y': 1}},
             'link_setups': [],
             'aerial_setups': {'kernel': get_cauchy_kernel(1, 1).name, 'beta': 1},
             'rate_based_control': True}
    env = CullOrThinEnvOCFormat(setup)
    env.reset()
    done = False
    total = 0
    # Inspect lots in one node and nothing in the other
    while not done:
        observation, reward, done, _ = env.step(np.array([5, 5, 0, 0], dtype=np.float32))
        total += reward
    env.reset()
    env_level = env.get_df()
    max_time = env_level['t'].max()
    net_df = env.net.get_df()
    max_time_net = net_df['t'].max()
    assert (np.isclose(max_time, max_time_net, atol=1.0))


def run_fixed_horizon(horizon):
    env_params = CullOrThinEnvParams(fixed_horizon=horizon)
    env = CullOrThinEnvOCFormat(default_setup, env_params=env_params)

    # Can't easily constrain random gym actions to comply to budget.
    env.disable_spend_checks()
    check_env(env)
    env.enable_spend_checks()

    env.reset()
    done = False
    step = 0
    while not done:
        observation, reward, done, _ = env.step(np.array([0]*3*2, dtype=np.float32))
        step += 1
        assert (reward <= 0)
    return step


def test_done_as_specified_for_fixed_horizon():
    step = run_fixed_horizon(1)
    assert step == 1
    step = run_fixed_horizon(3)
    assert step == 3
    step = run_fixed_horizon(100)
    assert step == 100


def test_nothing_dying_gives_max_reward_oc():
    # No initial infection so nothing dies
    setup = {'node_setups': ({'n': 100,
                              'beta': 5 / 100,
                              'gamma': 1,
                              'initial_infected': 0},
                             {'n': 100,
                              'beta': 5 / 100,
                              'gamma': 1,
                              'initial_infected': 0}
                             ),
             'node_locations': {0: {'x': 0, 'y': 0},
                                1: {'x': 0, 'y': 1}},
             'link_setups': [],
             'aerial_setups': {'kernel': get_cauchy_kernel(1, 1).name, 'beta': 1},
             'rate_based_control': True}
    env = CullOrThinEnvOCFormat(setup)
    env.reset()
    total = 0
    # Lots of culling (should have no effect). No thinning
    steps = 10
    population = 200
    for n in range(steps):
        observation, reward, done, _ = env.step(np.array([5, 0, 5, 0], dtype=np.float32))
        total += reward
    assert(total == - steps * population)


# Reward is now multiplied by 5 to give better scaling.
def test_everything_dying_gives_reward_of_zero():
    # Everything infected so will die eventually
    setup = {'node_setups': ({'n': 100,
                              'beta': 5 / 100,
                              'gamma': 1,
                              'initial_infected': 100},
                             {'n': 100,
                              'beta': 5 / 100,
                              'gamma': 1,
                              'initial_infected': 100}
                             ),
             'node_locations': {0: {'x': 0, 'y': 0},
                                1: {'x': 0, 'y': 1}},
             'link_setups': [],
             'aerial_setups': {'kernel': get_cauchy_kernel(1, 1).name, 'beta': 1},
             'rate_based_control': True}

    env_params = {'integral_reward': False}
    env_params = CullOrThinEnvParams(**env_params)
    env = CullOrThinEnvOCFormat(setup, env_params=env_params)
    env.reset()
    done = False
    # Inspect lots in one node and nothing in the other
    while not done:
        observation, reward, done, _ = env.step(np.array([5, 5, 0, 0], dtype=np.float32))
    assert(reward == 0)


def test_culling_and_thinning_work_as_expected_oc():
    # No disease so only culling and thinning have effect
    I_A = 10
    I_B = 20
    S_A = 90
    S_B = 80

    setup = {'node_setups': ({'n': I_A + S_A,
                              'beta': 0,
                              'gamma': 0,
                              'initial_infected': I_A},
                             {'n': I_B + S_B,
                              'beta': 0,
                              'gamma': 0,
                              'initial_infected': I_B}
                             ),
             'node_locations': {0: {'x': 0, 'y': 0},
                                1: {'x': 0, 'y': 1}},
             'link_setups': [],
             'aerial_setups': {'kernel': get_cauchy_kernel(1, 1).name, 'beta': 0},
             'rate_based_control': True}

    # Need to run with fixed horizon or all
    env_params = CullOrThinEnvParams(fixed_horizon=20, cull_cost=0.1, thin_cost=0.1)

    env = CullOrThinEnvOCFormat(setup, env_params=env_params)
    env.reset()
    # do nothing
    observation, reward, done, _ = env.step(np.array([0, 0, 0, 0], dtype=np.float32))
    assert (observation == np.array([S_A, S_B, I_A, I_B])).all()
    # cull some stuff
    cull = 4
    observation, reward, done, _ = env.step(np.array([cull, 0, cull, 0], dtype=np.float32))
    I_A -= cull
    I_B -= cull
    buffer = 2
    # Bigger than lower range
    bigger_than_lower = observation >= np.array([S_A, S_B, I_A, I_B]) - buffer
    smaller_than_upper = observation <= np.array([S_A, S_B, I_A, I_B]) + buffer
    assert np.logical_and(bigger_than_lower, smaller_than_upper).all()
    # thin some stuff
    thin_a = 4
    thin_b = 8
    observation, reward, done, _ = env.step(np.array([0, thin_a, 0, thin_b], dtype=np.float32))
    S_A -= thin_a
    S_B -= thin_b
    bigger_than_lower = observation >= np.array([S_A, S_B, I_A, I_B]) - buffer
    smaller_than_upper = observation <= np.array([S_A, S_B, I_A, I_B]) + buffer
    assert np.logical_and(bigger_than_lower, smaller_than_upper).all()
