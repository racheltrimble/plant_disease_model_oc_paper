from enum import IntEnum
from pathlib import Path
import yaml

import numpy as np
import pandas as pd

from plant_disease_model.simulator import Node
from plant_disease_model.simulator import get_cauchy_kernel, get_kernel_from_name


class EventType(IntEnum):
    NODE = 0
    AERIAL = 1
    TRADE = 2


class Network:
    # beta and gamma for spread within the subpopulations are passed in as a list of node_setups.
    # The constant of proportionality for network spread is captured as "beta_aerial"
    def __init__(self,
                 node_setups: tuple,
                 node_locations: dict,
                 link_setups,
                 aerial_setups: dict,
                 initial_infected_random=0,
                 log_dir=None,
                 logging_enabled: bool = True,
                 rate_based_control: bool = False,
                 seed=123
                 ):
        # Debug functionality - compares iterative rate setting to calculating at each step.
        # Super slow but should notice if iterative setting is broken.
        self.rate_checking = False
        self.random_gen = np.random.default_rng(seed=seed)
        root_dir = Path(__file__).parent.parent.parent
        if log_dir is None:
            self.log_dir = root_dir / Path("data")
        else:
            self.log_dir = root_dir / Path("data") / Path(log_dir)
        # Can't create log dir here or the multithreading gets upset.
        if not self.log_dir.exists():
            print("Looking for log_dir ", str(self.log_dir), " but it doesn't exist")
            assert 0

        # These are added randomly at each reset so not done here.
        self.initial_infected_random = initial_infected_random

        # Define network nodes and connectivity
        self.nodes = []
        total_population = 0
        for index, node_setup in enumerate(node_setups):
            location_dict = node_locations[index]
            location = (location_dict['x'], location_dict['y'])
            # Overwrite node id to ensure uniqueness
            node_setup['idx'] = index
            total_population += node_setup['n']
            self.nodes.append(Node(node_setup,
                                   self.random_gen,
                                   location,
                                   logging_enabled=logging_enabled,
                                   rate_based_control=rate_based_control))

        self.rate_based_control = rate_based_control
        # Check the trade is consistent or the network will not be stable.
        self.trade_definitions = link_setups
        self.link_check()

        self.trade_rates = np.zeros(len(link_setups))
        self.trade_thresholds = np.zeros(len(link_setups))
        threshold = 0
        # Add links
        for index, link in enumerate(link_setups):
            self.trade_rates[index] = link["rate"]
            threshold += link["rate"]
            self.trade_thresholds[index] = threshold
        self.total_trade_rate = self.trade_rates.sum()

        # Spatial spread based on a simplified version of Meentemeyer 2011:
        # https://esajournals.onlinelibrary.wiley.com/doi/full/10.1890/ES10-00192.1
        # Rate of transmission between a pair of nodes is:
        # beta x I_source x S_dst x K(d)
        # with K(d) being a single Cauchy function (2 are used in Meentemeyer).
        # Part of this can be precomputed and is consistent across the sim (beta x K)
        # Calculate the static part of the aerial node-node transmission
        self.kernel_factors = np.zeros((len(self.nodes), len(self.nodes)))
        kernel = get_kernel_from_name(aerial_setups['kernel'])
        for start_index, start_node in enumerate(self.nodes):
            for end_index, end_node in enumerate(self.nodes):
                # Don't count aerial transmission to yourself.
                if start_index == end_index:
                    self.kernel_factors[start_index, end_index] = 0
                else:
                    d = self.get_node_distance(start_node, end_node)
                    kernel_factor = aerial_setups['beta'] * kernel.kernel(d)
                    self.kernel_factors[start_index, end_index] = kernel_factor

        # This is only for recording node setup and shouldn't be used in main code.
        # Deliberately forcing to string to help notice misuse.
        self.aerial_beta = str(aerial_setups['beta'])
        self.aerial_kernel_name = kernel.name

        # Initialise rates...
        # Overall
        self.event_type_thresholds = np.zeros(len(EventType))

        # Per node...
        self.node_rates = np.zeros(len(self.nodes))
        self.node_thresholds = np.zeros(len(self.nodes))

        # Aerial spread
        # S vector and I vector don't need to be class level at the moment but aiming to reduce updates
        # by only changing necessary rows / columns.
        self.s_vector = np.zeros(len(self.nodes))
        self.i_vector = np.zeros(len(self.nodes))

        # Aerial rates are calculated per source and destination but only recorded per destination node
        # as aerial spread has no effect on the source node.
        self.aerial_rates_array = np.zeros((len(self.nodes), len(self.nodes)))
        self.aerial_rates = np.zeros(len(self.nodes))
        self.aerial_thresholds = np.zeros(len(self.nodes))

        # Declare time variable
        self.time = 0
        self.last_updates = None

        # Enable logging
        self.logging_enabled = logging_enabled

        # Log number of stashes for DF length sanity check
        self.stash_count = 0

        print("Created plant disease model with ", total_population, " hosts")

    # Input sanity check.
    # Check the amount flowing into sinks is the amount flowing out of sources.
    # Check the amount flowing into each node is the amount flowing out.
    # Also provides some checking of the dictionary format
    def link_check(self):
        per_node_flows = np.zeros(len(self.nodes))
        source_sink_flows = 0

        for link in self.trade_definitions:
            if link["start_type"] == "Node":
                per_node_flows[link["start_id"]] += link["rate"]
            else:
                source_sink_flows += link["rate"]
            if link["end_type"] == "Node":
                per_node_flows[link["end_id"]] -= link["rate"]
            else:
                source_sink_flows -= link["rate"]

        assert (per_node_flows == 0).all()
        assert source_sink_flows == 0

    @staticmethod
    def get_node_distance(start_node, end_node):
        x = start_node.x - end_node.x
        y = start_node.y - end_node.y
        return np.sqrt(x ** 2 + y ** 2)

    def reset(self, initial_override=None):
        if initial_override is None:
            # Distribute random initial infections
            initial_nodes = self.random_gen.choice(len(self.nodes), self.initial_infected_random, replace=True)

            overflow = 0
            for index, node in enumerate(self.nodes):
                # Add extra infections from random distribution
                extras_per_node = np.count_nonzero(initial_nodes == index) + overflow
                # if the random Is don't fit, overflow to the next node
                if extras_per_node > node.s0:
                    overflow = extras_per_node - node.s0
                    extras_per_node = node.s0
                else:
                    overflow = 0
                node.reset(extras_per_node)
            if overflow > 0:
                print(f"Warning - random allocation failed to place {overflow} hosts")
        else:
            assert len(initial_override) == len(self.nodes)
            for index, node in enumerate(self.nodes):
                node.reset(initial_override[index])

        self.time = 0
        self.last_updates = None

    def stash_results(self, final_time=None):
        self.stash_count += 1
        if self.logging_enabled:
            for node in self.nodes:
                node.stash_results(final_time)

    def update_rates(self):
        # Don't bother updating rates if nothing has happened.
        if len(self.last_updates) == 0:
            return
        for update_index in self.last_updates:
            # Start with the per node rates
            # Update based on the last node that was changed.
            node = self.nodes[update_index]
            node.calculate_rates()
            self.node_rates[update_index] = node.get_total_rate()
            self.s_vector[update_index] = node.s
            self.i_vector[update_index] = node.i
            self.aerial_rates_array[update_index] = \
                self.kernel_factors[update_index] * self.i_vector[update_index] * self.s_vector
            self.aerial_rates_array[:, update_index] = \
                self.kernel_factors[:, update_index] * self.i_vector * self.s_vector[update_index]
            threshold = 0 if update_index == 0 else self.node_thresholds[update_index-1]
            # Then do the threshold...
            for index in range(update_index, self.get_num_nodes()):
                threshold += self.node_rates[index]
                self.node_thresholds[index] = threshold
        self.event_type_thresholds[EventType.NODE] = threshold
        self.last_updates = []

        if self.rate_checking and not self.rate_based_control:
            for index, node in enumerate(self.nodes):
                if node.i == 0:
                    assert (self.node_rates[index] == 0)
            assert (self.i_vector == self.get_i_array()).all()

        # Then add in the aerial spread events
        # aerial_rates_array_test = self.kernel_factors * np.outer(self.i_vector, self.s_vector)
        # assert (np.isclose(self.aerial_rates_array, aerial_rates_array_test).all())
        self.aerial_rates = self.aerial_rates_array.sum(axis=0)

        threshold = 0
        for dst, rate in enumerate(self.aerial_rates):
            threshold += rate
            self.aerial_thresholds[dst] = threshold

        self.event_type_thresholds[EventType.AERIAL] = self.event_type_thresholds[EventType.NODE]
        self.event_type_thresholds[EventType.AERIAL] += self.aerial_thresholds[-1]

        # Then add in the trade events
        self.event_type_thresholds[EventType.TRADE] = self.event_type_thresholds[EventType.AERIAL]
        # Use of fixed sum avoids indexing error into threshold array when there are no trade links.
        self.event_type_thresholds[EventType.TRADE] += self.total_trade_rate

        if self.get_total_rate() <= 0.0:
            assert (self.get_i_array().sum() <= 0) or (self.kernel_factors.sum() == 0)

    def initialise_rates(self):
        # Start with the per node rates
        threshold = 0
        for index, node in enumerate(self.nodes):
            node.calculate_rates()
            self.node_rates[index] = node.get_total_rate()
            threshold += self.node_rates[index]
            self.node_thresholds[index] = threshold
            self.s_vector[index] = node.s
            self.i_vector[index] = node.i
        assert (self.s_vector >= 0).all()
        assert (self.i_vector >= 0).all()
        self.event_type_thresholds[EventType.NODE] = threshold

        # Then add in the aerial spread events
        self.aerial_rates_array = self.kernel_factors * np.outer(self.i_vector, self.s_vector)
        self.aerial_rates = self.aerial_rates_array.sum(axis=0)
        threshold = 0
        for dst, rate in enumerate(self.aerial_rates):
            threshold += rate
            self.aerial_thresholds[dst] = threshold

        self.event_type_thresholds[EventType.AERIAL] = self.event_type_thresholds[EventType.NODE]
        self.event_type_thresholds[EventType.AERIAL] += self.aerial_thresholds[-1]

        # Then add in the trade events
        self.event_type_thresholds[EventType.TRADE] = self.event_type_thresholds[EventType.AERIAL]
        # Use of fixed sum avoids indexing error into threshold array when there are no trade links.
        self.event_type_thresholds[EventType.TRADE] += self.total_trade_rate

    def get_total_rate(self):
        return float(self.event_type_thresholds[-1])

    def get_epidemic_rate(self):
        return float(self.event_type_thresholds[EventType.AERIAL])

    # Nested Gillespies across all the nodes.
    def progress(self, duration):
        remaining_time = duration
        end_time = self.time + duration
        # Randomly select node for update based on total rates
        while 1:
            # Make sure all the rates are correct
            self.calculate_rates()

            rate = self.get_total_rate()

            # If nothing is driving further movement, stop simulating.
            # (needs special casing to avoid div0)
            if rate <= 0.0:
                self.time = end_time
                break

            # Calculate time to next event
            tau = 1 / rate * np.log(1 / self.random_gen.random())
            remaining_time -= tau
            # Duration of 0 is a special case of "run until end of epidemic"
            if duration != 0 and remaining_time <= 0:
                self.time = end_time
                break

            self.time = self.time + tau
            # Select which type of event to simulate
            # Generate a random number scaled to the overall rate
            which_event_type = self.random_gen.random() * rate
            # Was using argmax here but seems to be slow.
            event_type = self.event_type_thresholds > which_event_type
            if event_type[EventType.NODE]:
                self.last_updates = self.do_node_event()
                if self.rate_checking:
                    print("NODE", self.last_updates)
            elif event_type[EventType.AERIAL]:
                self.last_updates = self.do_aerial_event()
                if self.rate_checking:
                    print("Aerial", self.last_updates)
            elif event_type[EventType.TRADE]:
                self.last_updates = self.do_trade_event()
                if self.rate_checking:
                    print("Trade", self.last_updates)
            else:
                print(EventType(event_type).name)
                assert 0
            self.log_event(self.time)

        # Tidy up the rates before returning.
        self.calculate_rates()

    def calculate_rates(self):
        if self.last_updates is None:
            self.initialise_rates()
        else:
            self.update_rates()

        if self.rate_checking:
            old_node_rates = self.node_rates
            self.initialise_rates()
            assert (old_node_rates == self.node_rates).all()

    def do_node_event(self):
        which_node = self.random_gen.random() * self.node_thresholds[-1]
        selected_node = np.argmax(which_node < self.node_thresholds)
        s, i = self.nodes[selected_node].do_event()
        if self.rate_checking:
            print("New S, I:", s, i)
        return [selected_node]

    def do_aerial_event(self):
        which_node = self.random_gen.random() * self.aerial_thresholds[-1]
        selected_node = np.argmax(which_node < self.aerial_thresholds)
        self.nodes[selected_node].infect_hosts(1)
        return [selected_node]

    def do_trade_event(self):
        # Pick which links are going to be the source for this trade.
        which_trade = self.random_gen.random() * self.trade_thresholds[-1]
        selected_trade = int(np.argmax(which_trade < self.trade_thresholds))

        # Decode from the supplied trade definitions.
        trade_definition = self.trade_definitions[selected_trade]
        start_type = trade_definition["start_type"]
        start_index = trade_definition["start_id"]
        end_type = trade_definition["end_type"]
        end_index = trade_definition["end_id"]

        selected_node = []
        # Source is another node
        if start_type == 'Node':
            traded_host = self.nodes[start_index].give_host_to_network()
            selected_node.append(start_index)
        # Fixed sources
        else:
            traded_host = self.get_network_source_host()
        if end_type == 'Node':
            self.nodes[end_index].receive_host_from_network(traded_host)
            selected_node.append(end_index)
        else:
            # No need to record trades sent to sinks.
            pass
        return selected_node

    # Defines the type of hosts fed into the network from external sources
    @staticmethod
    def get_network_source_host():
        return "S"

    def inspect(self, num_to_inspect_per_node, time):
        assert(len(num_to_inspect_per_node) == len(self.nodes))
        assert (num_to_inspect_per_node.dtype == np.int32)
        found = np.zeros(len(num_to_inspect_per_node))
        for index, node in enumerate(self.nodes):
            found[index] = node.inspect(int(num_to_inspect_per_node[index]), time)
        return found

    def treat(self, num_to_treat_per_node, time):
        assert(len(num_to_treat_per_node) == len(self.nodes))
        if not self.rate_based_control:
            assert (num_to_treat_per_node.dtype == np.int32)
        if self.rate_checking:
            print("Treating nodes:", num_to_treat_per_node)
        for index, node in enumerate(self.nodes):
            treat = num_to_treat_per_node[index]
            node.treat(treat, time)
            # Will need to change the rates for these nodes for the epidemic sim.
            if treat > 0:
                if self.last_updates is None:
                    self.initialise_rates()
                    self.last_updates = [index]
                else:
                    self.last_updates.append(index)
        self.log_event(time)

    def thin(self, num_to_thin_per_node, time):
        assert(len(num_to_thin_per_node) == len(self.nodes))
        if not self.rate_based_control:
            assert (num_to_thin_per_node.dtype == np.int32)
        if self.rate_checking:
            print("Thinning nodes:", num_to_thin_per_node)
        for num_to_thin, node_enum in zip(num_to_thin_per_node, enumerate(self.nodes)):
            index, node = node_enum
            node.thin(num_to_thin, time)
            # Will need to change the rates for these nodes for the epidemic sim.
            if num_to_thin > 0:
                if self.last_updates is None:
                    self.initialise_rates()
                    self.last_updates = [index]
                else:
                    self.last_updates.append(index)
        self.log_event(time)

    def log_event(self, time):
        if self.logging_enabled:
            # Adding to all per node graphs for now.
            for node in self.nodes:
                node.log_event(time)

    def get_num_nodes(self):
        return len(self.nodes)

    def get_n0s(self):
        out = np.zeros((len(self.nodes)))
        for index, node in enumerate(self.nodes):
            out[index] = node.n0
        return out

    def get_s_array(self):
        out = np.zeros((len(self.nodes)))
        for index, node in enumerate(self.nodes):
            out[index] = node.s
        return out

    def get_i_array(self):
        out = np.zeros((len(self.nodes)))
        for index, node in enumerate(self.nodes):
            out[index] = node.i
        return out

    def get_r_array(self):
        out = np.zeros((len(self.nodes)))
        for index, node in enumerate(self.nodes):
            out[index] = node.r
        return out

    def get_total_r(self):
        out = 0
        for node in self.nodes:
            out += node.r
        return out

    def get_total_i(self):
        out = 0
        for node in self.nodes:
            out += node.i
        return out

    def get_total_s(self):
        out = 0
        for node in self.nodes:
            out += node.s
        return out

    def get_df(self, resolution=None):
        if not self.logging_enabled:
            print("Attempting to retrieve data from a network with logging disabled")
            assert 0
        df = pd.DataFrame()
        length = None
        for node in self.nodes:
            new_data = node.get_df(resolution=resolution)
            if length is None:
                length = len(new_data)
            else:
                assert length == len(new_data)
            df = pd.concat([df, new_data])
        # +2 accounts for zero vs one indexed
        assert(self.stash_count == df['test_iteration'].max() + 1)
        return df

    def get_action_df(self):
        if not self.logging_enabled:
            print("Attempting to retrieve data from a network with logging disabled")
            assert 0

        df = pd.DataFrame()
        length = None
        for node in self.nodes:
            new_data = node.get_action_df()
            if length is None:
                length = len(new_data)
            else:
                assert length == len(new_data)
            df = pd.concat([df, new_data])
        return df

    def save_df(self, resolution=None):
        df = self.get_df(resolution=resolution)
        csv_name = "net_data_res_" + str(resolution) + ".csv"
        filename_csv = self.log_dir / Path(csv_name)
        df.to_csv(filename_csv)

        action_df = self.get_action_df()
        action_csv_name = "action_data.csv"
        filename_action = self.log_dir / Path(action_csv_name)
        action_df.to_csv(filename_action)

        yaml_name = "definition.yaml"
        filename = self.log_dir / Path(yaml_name)
        self.dump_def_to_yaml(filename)

        return filename_csv, filename_action

    def get_dict_def(self):
        dict_def = {}
        node_def = []
        for node in self.nodes:
            node_def.append(node.get_dict_def())

        dict_def['node_definitions'] = node_def
        dict_def['trade_definitions'] = list(self.trade_definitions)
        dict_def['aerial_beta'] = self.aerial_beta
        dict_def['aerial_kernel'] = self.aerial_kernel_name
        return dict_def

    def dump_def_to_yaml(self, filename):
        dict_def = self.get_dict_def()

        with open(filename, "w") as file:
            yaml.dump(dict_def, file)

    # Routes through to the same function in the node...
    def get_time_to_proportion_infected_dist(self, node, proportion):
        return self.nodes[node].get_time_to_proportion_infected_dist(proportion)

    def render(self):
        pass
