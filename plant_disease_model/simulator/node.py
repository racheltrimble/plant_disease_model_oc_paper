# Basic SIR model.
from enum import IntEnum
import numpy as np
import pandas as pd


class NodeEventType(IntEnum):
    SI = 0
    IR = 1
    SR = 2


class Node:
    def __init__(self, internal_setup, rng, location=(0, 0), logging_enabled=True, rate_based_control=False):

        self.random_gen = rng

        # Pull setup out of dictionary (acts as an interface check...)
        self._n0 = internal_setup['n']
        assert(isinstance(self._n0, int))

        self.initial_infected = internal_setup['initial_infected']
        assert (isinstance(self.initial_infected, int))

        self.idx = int(internal_setup['idx'])
        assert (isinstance(self.idx, int))

        self._x, self._y = location

        self.s0 = self._n0 - self.initial_infected
        self.i0 = self.initial_infected
        self.r0 = 0
        self.t0 = 0
        self._s = self.s0
        self._i = self.i0
        self._r = self.r0

        self.beta = internal_setup['beta']
        self.gamma = internal_setup['gamma']

        # Variables for single test logging
        self.time_list = []
        self.s_list = []
        self.i_list = []
        self.r_list = []
        self.time = 0
        self.action_time_list = []
        self.inspect_list = []
        self.cull_list = []
        self.thin_list = []

        # Initialise the overall logging
        self.time_lists = []
        self.s_lists = []
        self.i_lists = []
        self.r_lists = []
        self.action_time_lists = []
        self.inspect_lists = []
        self.cull_lists = []
        self.thin_lists = []
        self.stashed = True

        # Declare existence of rates and thresholds
        # Note - relies on run function to reset and properly set these.
        self.rates = np.array([0.0,   # S->I
                               0.0,   # I->R
                               0.0    # S->R
                               ])
        self.thresholds = []

        # Track the number of cases when we should have sent a host according to the trade rates but couldn't.
        self.bad_trades = 0

        self.rate_based_control = rate_based_control
        self.cull_rate = 0
        self.thin_rate = 0

        # Switch logging on and off.
        self.logging_enabled = logging_enabled

    def stash_results(self, final_time=None):
        if self.logging_enabled:
            # Stash any results from the last simulation
            assert (len(self.time_list) >= 1)
            if final_time is not None:
                # Add a final point at the end of the plot
                self.time_list.append(final_time)
                self.s_list.append(self._s)
                self.i_list.append(self._i)
                self.r_list.append(self._r)
            # Stash
            self.time_lists.append(self.time_list)
            self.s_lists.append(self.s_list)
            self.i_lists.append(self.i_list)
            self.r_lists.append(self.r_list)
            self.action_time_lists.append(self.action_time_list)
            self.inspect_lists.append(self.inspect_list)
            self.cull_lists.append(self.cull_list)
            self.thin_lists.append(self.thin_list)

            self.stashed = True

    # Resets state of node and of per sim logging
    # Note - doesn't remove logged results.
    def reset(self, random_i_allocation=0):
        # Check any previous results have been stashed
        assert self.time_list == [] or self.stashed or not self.logging_enabled
        self._s = self.s0 - random_i_allocation
        self._i = self.i0 + random_i_allocation
        self._r = self.r0

        self.calculate_rates()

        self.time_list = [0]
        self.s_list = [self._s]
        self.i_list = [self._i]
        self.r_list = [self._r]
        self.action_time_list = []
        self.inspect_list = []
        self.cull_list = []
        self.thin_list = []

        self.time = 0
        self.stashed = False

    # Rate equations:
    # S to I = beta * S * (I + D)
    # I to R = gamma * I
    def calculate_rates(self):
        # Prevent senseless negatives from human actions.
        if self._i == 0:
            local_cull_rate = 0
        else:
            local_cull_rate = self.cull_rate
        if self._s == 0:
            local_thin_rate = 0
        else:
            local_thin_rate = self.thin_rate
        # ['SI', 'IR', 'SR']
        self.rates[NodeEventType.SI] = self.beta * self._s * self._i
        self.rates[NodeEventType.IR] = self.gamma * self._i + local_cull_rate
        self.rates[NodeEventType.SR] = local_thin_rate
        threshold = 0
        self.thresholds = []
        for n in range(len(self.rates)):
            threshold += self.rates[n]
            self.thresholds.append(threshold)
        if self._i > 0:
            assert(self.get_total_rate() > 0) or (self.beta == 0)
        else:
            assert (self.rates >= 0).all()

    def get_total_rate(self):
        return sum(self.rates)

    def do_event(self):
        # Select which reaction to occur
        which_event = self.random_gen.random()*self.get_total_rate()

        # Update populations
        if which_event < self.thresholds[0]:
            # SI
            self._s = self._s - 1
            self._i = self._i + 1
        elif which_event < self.thresholds[1]:
            # IR
            self._r = self._r + 1
            self._i = self._i - 1
        else:
            # SR
            self._s = self._s - 1
            self._r = self._r + 1
        assert(self._i >= 0)
        assert(self._s >= 0)
        assert(self._r >= 0)
        return self._s, self._i

    # Used to represent aerial spread and seed initial random infections
    def infect_hosts(self, num_hosts):
        assert self._s >= num_hosts
        self._i += num_hosts
        self._s -= num_hosts

    def give_host_to_network(self):
        # Special case is there are no susceptible or infected hosts in this node
        if self._s + self._i == 0:
            self.bad_trades += 1
            return "NONE"

        which_host = self.random_gen.random() * (self._s + self._i)
        if which_host < self._s:
            self._s -= 1
            return "S"
        else:
            self._i -= 1
            return "I"

    def receive_host_from_network(self, host):
        if host == "S":
            self._s += 1
        elif host == "I":
            self._i += 1
        elif host == "NONE":
            pass
        else:
            assert 0

    def log_event(self, time):
        if self.logging_enabled:
            assert(time >= self.time_list[-1])
            if len(self.action_time_list) > 0:
                assert (time >= self.action_time_list[-1])
            self.time_list.append(time)
            self.s_list.append(self._s)
            self.i_list.append(self._i)
            self.r_list.append(self._r)

    # Simulate as an independent node
    # With thanks to:
    # https://lewiscoleblog.com/gillespie-algorithm
    def progress(self, duration):
        remaining_time = duration
        end_time = self.time + duration
        while 1:
            # Calculate updated overall event rate
            self.calculate_rates()
            rate = self.get_total_rate()
            # If nothing is driving further movement, stop simulating.
            # (needs special casing to avoid div0)
            if rate <= 0.0:
                break

            # Calculate time to next event
            tau = 1 / rate * np.log(1 / self.random_gen.random())
            remaining_time -= tau
            # Duration of 0 is a special case of "run until end of epidemic"
            if duration != 0 and remaining_time <= 0:
                self.time = end_time
                break

            self.time = self.time + tau
            self.do_event()
            # Using internal time variable at this level so the logging
            # can be triggered externally.
            self.log_event(self.time)

    def inspect(self, num_to_inspect, time):
        assert(isinstance(num_to_inspect, int))
        assert(num_to_inspect >= 0)
        inspection_pop = self._s + self._i

        # Any requests to inspect extra hosts are lost.
        num_to_inspect = min(inspection_pop, num_to_inspect)
        inspected = self.random_gen.choice(inspection_pop, size=num_to_inspect, replace=False)
        found = np.count_nonzero(inspected < self._i)
        if len(self.action_time_list) > 0:
            assert (time >= self.action_time_list[-1])
        assert (time >= self.time_list[-1])
        assert(found <= self._i)
        self.action_time_list.append(time)
        self.inspect_list.append(num_to_inspect)
        self.cull_list.append(np.NAN)
        self.thin_list.append(np.nan)
        return found

    def treat(self, num_to_treat, time):
        if self.rate_based_control:
            self.cull_rate = num_to_treat
        else:
            num_to_treat = int(num_to_treat)
            if num_to_treat > self._i:
                num_to_treat = self._i
            self._i -= num_to_treat
            self._r += num_to_treat
        self.action_time_list.append(time)
        self.inspect_list.append(np.nan)
        self.cull_list.append(num_to_treat)
        self.thin_list.append(np.nan)

    def thin(self, num_to_thin, time):
        if self.rate_based_control:
            self.thin_rate = num_to_thin
        else:
            num_to_thin = int(num_to_thin)
            if num_to_thin > self._s:
                num_to_thin = self._s
            self._s -= num_to_thin
            self._r += num_to_thin
        self.action_time_list.append(time)
        self.inspect_list.append(np.nan)
        self.cull_list.append(np.NAN)
        self.thin_list.append(num_to_thin)

    @property
    def s(self):
        return self._s

    @property
    def i(self):
        return self._i

    @property
    def r(self):
        return self._r

    @property
    def x(self):
        return self._x

    @property
    def y(self):
        return self._y

    @property
    def n0(self):
        return self._n0

    # Returns the time taken to get to a certain proportion of hosts
    # infected or removed.
    # Note this is done relative to the initial population size.
    def get_time_to_proportion_infected_dist(self, proportion):
        self.check_logging_enabled()
        dist = []
        count_disqualified = 0
        for index, i_trace in enumerate(self.i_lists):
            r_arr = np.array(self.r_lists[index])
            i_arr = np.array(i_trace)
            t_arr = np.array(self.time_lists[index])
            bad_arr = r_arr + i_arr
            qualifying = bad_arr > proportion*self._n0
            # If there are no bad times then don't want to contribute to
            # distribution
            if np.sum(qualifying):
                index = np.argmax(qualifying)
                dist.append(t_arr[index])
            else:
                count_disqualified += 1
        return np.array(dist), count_disqualified

    def check_logging_enabled(self):
        if not self.logging_enabled:
            print("Attempt to access logging data when logging not enabled")
            assert 0

    # Collates the final number of susceptible in each test into a list
    def get_final_susceptible(self):
        self.check_logging_enabled()
        out = []
        for test_result in self.s_lists:
            out.append(test_result[-1])
        return out

    def get_lists_dictionary(self):
        self.check_logging_enabled()
        out = {"S": self.s_lists,
               "I": self.i_lists,
               "R": self.r_lists,
               "t": self.time_lists}
        return out

    def get_df(self, resolution=None):
        self.check_logging_enabled()
        out = pd.DataFrame()
        if resolution is None:
            for iteration, time_list in enumerate(self.time_lists):
                d = {'t': time_list,
                     'nS': self.s_lists[iteration],
                     'nI': self.i_lists[iteration],
                     'nR': self.r_lists[iteration],
                     'test_iteration': len(time_list)*[iteration],
                     'idx': len(time_list)*[self.idx]
                     }
                new_data = pd.DataFrame.from_dict(d, orient='index').transpose()
                out = pd.concat([out, new_data])
        else:
            # Data is quantised by spreading out the data points between 0 and the max time across all tests.
            max_ts = [max(item) for item in self.time_lists]
            t_max = max(max_ts)
            time_list = np.linspace(0, t_max, resolution)
            for iteration, old_t_list in enumerate(self.time_lists):
                d = {'t': time_list,
                     'nS': self.quantise(self.s_lists[iteration], time_list, old_t_list),
                     'nI': self.quantise(self.i_lists[iteration], time_list, old_t_list),
                     'nR': self.quantise(self.r_lists[iteration], time_list, old_t_list),
                     'test_iteration': len(time_list)*[iteration],
                     'idx': len(time_list)*[self.idx]
                     }
                new_data = pd.DataFrame.from_dict(d, orient='index').transpose()
                out = pd.concat([out, new_data])
        return out

    # Actions should be natively time quantised so just pulling out into df.
    def get_action_df(self):
        self.check_logging_enabled()
        out = pd.DataFrame()
        for iteration, action_time_list in enumerate(self.action_time_lists):
            d = {'t': action_time_list,
                 'inspect': self.inspect_lists[iteration],
                 'cull': self.cull_lists[iteration],
                 'thin': self.thin_lists[iteration],
                 'test_iteration': len(action_time_list)*[iteration],
                 'idx': len(action_time_list)*[self.idx]
                 }
            new_data = pd.DataFrame.from_dict(d, orient='index').transpose()
            out = pd.concat([out, new_data])
        return out

    def get_dict_def(self):
        out = {'idx': self.idx,
               'n0': self._n0,
               'initial_infected': self.initial_infected,
               'x': self._x,
               'y': self._y,
               'beta': self.beta,
               'gamma': self.gamma
               }
        return out

    @staticmethod
    def quantise(host_list, t_list, old_t_list):
        old_index = 0
        quantised = np.zeros(len(t_list))
        for index, new_time in enumerate(t_list):
            # The new time list will go up to the largest time across all repeats so
            # if we get to the end of the data, we keep repeating the last value.
            if old_index == len(old_t_list):
                quantised[index] = host_list[old_index - 1]
            else:
                while 1:
                    if old_t_list[old_index] == new_time:
                        quantised[index] = host_list[old_index]
                        break
                    elif old_t_list[old_index] > new_time:
                        quantised[index] = host_list[old_index - 1]
                        break
                    else:
                        old_index += 1
                        if old_index == len(old_t_list):
                            quantised[index] = host_list[old_index - 1]
                            break
        return quantised


if __name__ == "__main__":
    from simulator import Simulator
    from plotter import plot_df
    test_rng = np.random.default_rng(seed=123)
    test_setup = {'n': 100,
                  'beta': 5/100,
                  'gamma': 1,
                  'initial_infected': 1,
                  'idx': 0}
    test = Node(test_setup, test_rng)
    sim = Simulator(test, 4, 100)
    sim.run()
    frame_inf = test.get_df(resolution=None)
    frame = test.get_df(resolution=100)
    plot_df({0: frame_inf, 1: frame}, "test", filename="../../data/command_line_test/plot")

    echo = test.get_dict_def()

    print(test.get_time_to_proportion_infected_dist(50/100))
    test.stash_results()
    test.reset()
    test.progress(0.2)
    test.inspect(5, 0.2)
