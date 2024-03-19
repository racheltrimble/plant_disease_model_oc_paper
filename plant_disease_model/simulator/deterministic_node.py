# Basic SIR model.
from plant_disease_model.simulator import Node
import numpy as np
from scipy.integrate import solve_ivp


class DeterministicNode(Node):
    def __init__(self, internal_setup, rng, location=(0, 0), rate_based_control=False):
        super().__init__(internal_setup, rng, location, rate_based_control=rate_based_control)

    # Nest inside function to get rid of "self"
    def get_epidemic_ode(self):
        # y is vector of SIR
        def epidemic_ode(_, y):
            # Rate equations:
            # S to I = beta * S * I
            # I to R = gamma * I
            # Calculate updated overall event rate
            S, I, R = y
            dS = -self.beta * S * I
            dI = self.beta * S * I - self.gamma * I
            dR = self.gamma * I

            dydt = np.array([dS, dI, dR])
            return dydt
        return epidemic_ode

    def get_epidemic_ode_rate_based(self):
        # y is vector of SIR
        def epidemic_ode(_, y):
            # Rate equations:
            # S to I = beta * S * I
            # I to R = gamma * I
            # Calculate updated overall event rate
            S, I, R = y
            dS = -self.beta * S * I
            dI = self.beta * S * I - self.gamma * I - self.cull_rate
            dR = self.gamma * I + self.cull_rate

            dydt = np.array([dS, dI, dR])
            return dydt
        return epidemic_ode

    # Simulate as an independent node
    def progress(self, duration):
        y0 = np.array([self.s, self.i, self.r])
        if self.rate_based_control:
            ode = self.get_epidemic_ode_rate_based()
        else:
            ode = self.get_epidemic_ode()
        sol = solve_ivp(ode, (0, duration), y0, t_eval=[duration])
        assert(len(sol.y[:, 0]) == 3)
        self._s, self._i, self._r = sol.y[:, 0]
        self.time += duration
        self.log_event(self.time)


if __name__ == "__main__":
    from simulator import Simulator
    from plotter import plot_df
    from pathlib import Path
    test_rng = np.random.default_rng(seed=123)
    test_setup = {'n': 100,
                  'beta': 5/100,
                  'gamma': 1,
                  'initial_infected': 1,
                  'idx': 0}
    test = DeterministicNode(test_setup, test_rng)
    sim = Simulator(test, 4, 1)
    sim.run_quantised(100)
    frame_inf = test.get_df(resolution=None)
    frame = test.get_df(resolution=100)
    command_line_test_dir = Path("../../data/command_line_test")
    if not command_line_test_dir.exists():
        command_line_test_dir.mkdir(parents=True)
    filepath = command_line_test_dir / Path("plot.png")
    plot_df({0: frame_inf, 1: frame}, "test", filename=str(filepath))

    echo = test.get_dict_def()

    print(test.get_time_to_proportion_infected_dist(50/100))
    test.stash_results()
    test.reset()
    test.progress(0.2)
#    test.inspect(5, 0.2)
