# Generates plots showing disease spread at different R0
from plant_disease_model.simulator import Node, Network, DeterministicNode
from plant_disease_model.simulator import plotter, get_constant_kernel
import numpy as np
from scipy.integrate import solve_ivp

test_rng = np.random.default_rng(seed=123)
local_beta = 0.01
gamma = 0.2
n = 100
seed_infected = 20
remote_beta = 0.002
test_setup = {'n': n,
              'beta': local_beta,
              'gamma': gamma,
              'initial_infected': seed_infected,
              'idx': 0}
timestep = 1

node_setups = ({'n': n,
                'beta': local_beta,
                'gamma': gamma,
                'initial_infected': seed_infected},
               {'n': n,
                'beta': local_beta,
                'gamma': gamma,
                'initial_infected': seed_infected}
               )
node_locations = {0: {'x': 0, 'y': 0},
                  1: {'x': 1, 'y': 0}}
link_setups = []
aerial_setups = {'kernel': get_constant_kernel(1).name, 'beta': remote_beta}


class DeterministicDoubleNode:
    def __init__(self, node_setups, aerial_setups):
        self.beta = node_setups[0]["beta"]
        assert self.beta == node_setups[1]["beta"]
        self.gamma = node_setups[0]["gamma"]
        assert self.gamma == node_setups[1]["gamma"]
        self.beta_remote = aerial_setups["beta"]
        self.i0 = [node_setups[0]["initial_infected"],
                   node_setups[1]["initial_infected"]]
        self.s0 = [node_setups[0]["n"] - self.i0[0],
                   node_setups[1]["n"] - self.i0[1]]
        self.r0 = [0, 0]
        self.s = self.s0.copy()
        self._i = self.i0.copy()
        self.r = self.r0.copy()
        self.cull_rate = 0

    @property
    def i(self):
        return self._i[0] + self._i[1]

    # Nest inside function to get rid of "self"
    def get_epidemic_ode_rate_based(self):
        # y is vector of SIR
        def epidemic_ode(_, y):
            # Rate equations:
            # S to I = beta * S * I
            # I to R = gamma * I
            # Calculate updated overall event rate
            S0, I0, R0, S1, I1, R1 = y
            dS0 = -(self.beta * I0 + self.beta_remote * I1) * S0
            dI0 = (self.beta * I0 + self.beta_remote * I1) * S0 - self.gamma * I0 - self.cull_rate
            dR0 = self.gamma * I0 + self.cull_rate
            dS1 = -(self.beta * I1 + self.beta_remote * I0) * S1
            dI1 = (self.beta * I1 + self.beta_remote * I0) * S1 - self.gamma * I1 - self.cull_rate
            dR1 = self.gamma * I1 + self.cull_rate

            dydt = np.array([dS0, dI0, dR0, dS1, dI1, dR1])
            return dydt
        return epidemic_ode

    def reset(self):
        self.s = self.s0.copy()
        self._i = self.i0.copy()
        self.r = self.r0.copy()

    # Simulate as an independent node
    def progress(self, duration):
        y0 = np.array([self.s[0], self._i[0], self.r[0], self.s[1], self._i[1], self.r[1]])
        ode = self.get_epidemic_ode_rate_based()
        sol = solve_ivp(ode, (0, duration), y0, t_eval=[duration])
        assert(len(sol.y[:, 0]) == 6)
        self.s[0], self._i[0], self.r[0], self.s[1], self._i[1], self.r[1] = sol.y[:, 0]

    def treat(self, cull_rate, _):
        self.cull_rate = cull_rate

    def stash_results(self, _):
        pass


def stochastic_die_out_with_deterministic_rate_single_node():
    # Numerical search for the ODE rate that gives 0 infected.
    det_node = DeterministicNode(test_setup, test_rng, rate_based_control=True)
    rate = iterate_deterministic_system(det_node)
    test = Node(test_setup, test_rng, rate_based_control=True)

    repeats = 1000
    results = np.zeros(repeats)
    for n in range(repeats):
        test.reset()
        test.treat(rate, 0)
        test.progress(timestep)
        results[n] = test.i
        test.stash_results(final_time=1)
    print(f"Out of 100 tests, {(results == 0).sum()} extinctions")
    df = test.get_df(resolution=100)
    iterations = [x for x in range(0, 29)]
    plotter.plot_df({0: df},
                    "Single node stochastic sim \n deterministic culling rate",
                    filename="single_node.png",
                    one_axis=True,
                    i_only=True,
                    display=False,
                    iterations=iterations)


def stochastic_die_out_with_deterministic_rate_double_node():
    # Numerical search for the ODE rate that gives 0 infected.
    det_net = DeterministicDoubleNode(node_setups, aerial_setups)
    rate = iterate_deterministic_system(det_net)
    test = Network(node_setups, node_locations, link_setups, aerial_setups, rate_based_control=True)

    repeats = 1000
    results = np.zeros((repeats, 2))
    for n in range(repeats):
        test.reset()
        test.treat(np.array([rate, rate], np.int32), 0)
        test.progress(timestep)
        results[n] = test.get_i_array()
        test.stash_results(final_time=1)
    print(f"Out of 100 tests on 2 nodes, {(results == 0).sum()} extinctions")
    test.reset()
    df = test.get_df()
    iterations = [x for x in range(0, 29)]
    reintroductions = plotter.get_reintroductions(df, iterations)
    df = test.get_df(resolution=100)
    plotter.plot_df(plotter.split_by_node(df),
                    "Double node stochastic sim with deterministic culling rate",
                    filename="double_node.png",
                    one_axis=True,
                    i_only=True,
                    display=False,
                    reintroductions=reintroductions,
                    iterations=iterations)


def iterate_deterministic_system(det_node):
    maxrate = 100
    minrate = 0
    converged = False
    iterations = 0
    while not converged:
        rate = (maxrate + minrate) / 2
        det_node.reset()
        det_node.treat(rate, 0)
        det_node.progress(timestep)
        # If there are still I remaining, the cull rate was too small
        if det_node.i > 0:
            minrate = rate
        else:
            maxrate = rate
        assert(maxrate > minrate)
        det_node.stash_results(1)
        converged = (maxrate - minrate) < 0.01
        iterations += 1
        assert iterations < 1000

    print(f"Found rate {rate}")
    return rate


def stochastic_die_out_comparisons():
    stochastic_die_out_with_deterministic_rate_single_node()
    stochastic_die_out_with_deterministic_rate_double_node()


if __name__ == "__main__":
    stochastic_die_out_comparisons()
