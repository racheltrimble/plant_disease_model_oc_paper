import numpy as np
from scipy.optimize import newton

from plant_disease_model.simulator import Node, Simulator

rng = np.random.default_rng()


def test_basics():
    test_setup = {'n': 100,
                  'beta': 5/100,
                  'gamma': 1,
                  'initial_infected': 1,
                  'idx': 0}
    test = Node(test_setup, rng)
    sim = Simulator(test, sim_duration=4, repeats=100)
    sim.run()
    assert (len(test.get_time_to_proportion_infected_dist(50/100)) == 2)

    test_setup = {'n': 100,
                  'beta': 5/100,
                  'gamma': 1,
                  'initial_infected': 0,
                  'idx': 0}
    test = Node(test_setup, rng)
    sim = Simulator(test, sim_duration=4, repeats=100)
    sim.run()
    assert 1


def test_zero_infection_always_zero():
    test_setup = {'n': 100,
                  'beta': 5/100,
                  'gamma': 1,
                  'initial_infected': 0,
                  'idx': 0}
    test = Node(test_setup, rng)
    sim = Simulator(test, sim_duration=4, repeats=100)
    sim.run()
    _, count_disqualified = test.get_time_to_proportion_infected_dist(1 / 100)
    assert (count_disqualified == 100)


# Equilibrium R (r_infinity) as per Keeling & Rohani equation 2.7
def get_equilib_f(s0, beta_n, gamma):
    r0 = beta_n/gamma

    def f(x):
        return 1 - x - s0 * np.exp(-x * r0)
    return f


def r0_compare(n, beta, gamma, repeats=100):
    # Find the expected equilibrium using Newton Raphson
    # Note that the K&R equations are all expressed as per capita so scaling in and out of this
    # as required.
    s0 = (n - 1)/n
    expected_r_infinity = newton(get_equilib_f(s0, beta*n, gamma), n/2, maxiter=100) * n

    print(n, beta, gamma)
    test_setup = {'n': n,
                  'beta': beta,
                  'gamma': gamma,
                  'initial_infected': 1,
                  'idx': 0}
    # sim_duration of 0 is special case meaning - go until rates fall to zero.
    test = Node(test_setup, rng)
    sim = Simulator(test, sim_duration=0, repeats=repeats)
    sim.run()
    susceptible_results = np.array(test.get_final_susceptible())

    # Expecting to see some die out due to stochasticity.
    # Rounding up to account for small n.
    die_out = np.ceil(susceptible_results > n*0.95)

    # If R0, is > 0, exclude the points with stochastic fade from the average for comparison:
    n_die_out = np.logical_not(die_out)

    # Expecting most to kill off the entire population (minus some stochastic boundary)
    epidemic = susceptible_results <= n*0.05

    # Expected die outs are determined by R0 (K&R equation 6.4)
    expected_die_outs = repeats * gamma/(beta*n)

    # If R0 > 1 expecting most of the results to be epidemic. Else, expecting most to die out.
    if beta*n > gamma:
        assert (epidemic.any())
        assert(die_out.sum() < epidemic.sum())
        # Need to exclude die outs from the averaging when comparing to deterministic case.
        sim_result = n - susceptible_results[n_die_out].mean()
        diff = sim_result - expected_r_infinity
        print("Diff: ", diff, "Sim result: ", sim_result, "Expected R(infinity): ", expected_r_infinity)
        assert (np.sign(diff) * diff < n * 1 / 100)

        sim_result = die_out.sum()
        diff = sim_result - expected_die_outs
        print("Diff: ", diff, "Sim result: ", sim_result, "Expected R(infinity): ", expected_die_outs)
        assert (np.sign(diff) * diff < repeats * 5 / 100)
    else:
        assert (die_out.any())
        assert(die_out.sum() > epidemic.sum())
        # Can't say anything more for small r0 (stochastic-deterministic equivalence is dodgy).


# Behaviour for unforced epidemic agrees with R0 (R0 > 1) to within 5% over 1000 runs.
# (few fixed, few random settings)
def test_big_r0():
    r0_compare(n=100, beta=5/100, gamma=1, repeats=1000)


def test_big_r0_extended():
    r0_compare(n=1000, beta=10/100, gamma=1)
    for i in range(3):
        # This only works for large N and large R0 (ref K&R 6.6.3)
        n = int(rng.integers(100, 1000))
        beta = rng.random()*10/100
        # Forcing gamma to be smaller than beta n by a reasonable margin
        gamma = beta*rng.random()*100/3
        r0_compare(n, beta, gamma)


def test_small_r0():
    r0_compare(n=100, beta=1/100, gamma=5)


def test_small_r0_extended():
    r0_compare(n=1000, beta=1/100, gamma=10)
    for i in range(3):
        n = int(rng.integers(10, 1000))
        gamma = rng.random()*10
        # Forcing gamma to be bigger than beta n by a reasonable margin
        beta = gamma*rng.random()/2/n
        r0_compare(n, beta, gamma)


def test_initial_i_result_correct():
    n = 100
    for i in range(0, n, 20):
        test_setup = {'n': n,
                      'beta': 100/100,
                      'gamma': 0.1,
                      'initial_infected': i,
                      'idx': 0}
        test = Node(test_setup, rng)
        sim = Simulator(test, sim_duration=1, repeats=100)
        sim.run()
        df = test.get_df()
        first_is = df.loc[df['t'] == 0.0]['nI']
        for first_i in first_is:
            assert(first_i == i)

# Also tests that the number of repeats was what we asked for.


def test_n_consistent():
    n = 100
    repeats = 130
    test_setup = {'n': n,
                  'beta': 5/100,
                  'gamma': 1,
                  'initial_infected': 1,
                  'idx': 0}
    # sim_duration of 0 is special case meaning - go until rates fall to zero.
    test = Node(test_setup, rng)
    sim = Simulator(test, sim_duration=0, repeats=repeats)
    sim.run()
    lists = test.get_lists_dictionary()
    s_lists = lists["S"]
    i_lists = lists["I"]
    r_lists = lists["R"]

    assert (len(s_lists) == repeats)
    assert (len(i_lists) == repeats)
    assert (len(r_lists) == repeats)

    for index, s_list in enumerate(s_lists):
        i_list = i_lists[index]
        r_list = r_lists[index]
        assert (len(s_list) == len(i_list))
        assert (len(s_list) == len(r_list))
        for time, s in enumerate(s_list):
            if not(s + i_list[time] + r_list[time] == n):
                print("\nS list:", s_list, "\nI list:", i_list, "\nR list:", r_list)
                print(s, i_list[time], r_list[time])
            assert(s + i_list[time] + r_list[time] == n)


def test_sim_length():
    repeats = 27
    n = 100
    sim_duration = rng.integers(1, 20)
    test_setup = {'n': n,
                  'beta': 2/100,
                  'gamma': 1,
                  'initial_infected': 1,
                  'idx': 0}
    # sim_duration of 0 is special case meaning - go until rates fall to zero.
    test = Node(test_setup, rng)
    sim = Simulator(test, sim_duration=sim_duration, repeats=repeats)
    sim.run()
    lists = test.get_lists_dictionary()
    time_lists = lists["t"]

    assert (len(time_lists) == repeats)

    for time_list in time_lists:
        assert(max(time_list) <= sim_duration)
        # We are expecting to add a final point at the end of the simulation
        # Otherwise the graphs are different lengths for display.
        assert(time_list[-1] == sim_duration)


def test_bigger_beta_goes_faster():
    test_setup = {'n': 100,
                  'beta': 5/100,
                  'gamma': 1,
                  'initial_infected': 1,
                  'idx': 0}
    test1 = Node(test_setup, rng)
    sim = Simulator(test1, sim_duration=4, repeats=100)
    sim.run()
    dist1 = test1.get_time_to_proportion_infected_dist(0.5)

    test_setup = {'n': 100,
                  'beta': 50/100,
                  'gamma': 10,
                  'initial_infected': 1,
                  'idx': 0}
    test2 = Node(test_setup, rng)
    sim = Simulator(test2, sim_duration=4, repeats=100)
    sim.run()
    dist2 = test2.get_time_to_proportion_infected_dist(0.5)

    assert (dist1[0].mean() > dist2[0].mean())
