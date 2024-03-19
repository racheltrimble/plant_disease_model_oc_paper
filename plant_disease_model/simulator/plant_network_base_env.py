from dataclasses import dataclass
from pathlib import Path

import gym
import numpy as np
import pandas as pd

from plant_disease_model.simulator import Network


@dataclass
class PlantNetworkBaseEnvParams:
    env_type: str = "continuous"
    stack_depth: int = 4

    timestep: float = 1.0
    training: bool = False
    randomise_initial_training: bool = False
    fixed_horizon: float = None


class PlantNetworkBase(gym.Env):
    def __init__(self,
                 setup,  # ... for the epidemic model
                 log_dir=None,
                 seed=123,
                 env_params: PlantNetworkBaseEnvParams = None
                 ):
        if env_params is None:
            env_params = PlantNetworkBaseEnvParams()
        self.timestep = env_params.timestep
        self.training = env_params.training

        # By default, we also want to limit the logging behaviour during training as this detailed
        # data is never used and can take up a lot of memory.
        self.logging_enabled = not self.training

        # Combined parameters which allow initial state of env to be fully randomised during training.
        # Note that this is applied on reset of the model to try to separate out the epidemic and env
        # functionality.
        self.randomise_initial_training = env_params.randomise_initial_training
        if not isinstance(self.training, bool):
            print(self.training)
        self.randomise_initial_training &= self.training

        # Keep a separate "clock" here to label actions for logging.
        self.time = 0

        # Optional parameter for initial setup randomisation. This is more limited and intended to
        # be used during eval as well as training.
        initial_infected_random = setup.get('initial_infected_random', 0)
        rate_based_control = setup.get("rate_based_control", False)

        self.fixed_horizon = env_params.fixed_horizon

        self.net = Network(setup['node_setups'],
                           setup['node_locations'],
                           setup['link_setups'],
                           setup['aerial_setups'],
                           initial_infected_random=initial_infected_random,
                           log_dir=log_dir,
                           logging_enabled=self.logging_enabled,
                           rate_based_control=rate_based_control,
                           seed=seed)

        self.num_nodes = self.net.get_num_nodes()

        self.reward_list = []
        self.reward_lists = []
        self.observation_list = []
        self.observation_lists = []
        self.action_list = []
        self.action_lists = []
        self.time_list = []
        self.time_lists = []

        # This is used to normalise actions and observations across the program so stash it here...
        self.n0s = self.net.get_n0s()

        self.first_run = True


    def get_done(self):
        if self.fixed_horizon is None:
            done = self.net.get_epidemic_rate() <= 0
            if done:
                assert(self.net.get_i_array().sum() == 0)
        else:
            epsilon_time = 0.005
            done = (self.time + epsilon_time >= self.fixed_horizon)
        return done

    def reset(self):
        # Later versions need the following line to seed self.np_random
        # super().reset(seed=seed)

        if not self.first_run:
            self.stash_results(final_time=self.fixed_horizon)
        self.first_run = False
        self.time = 0
        # Note - resetting does not affect random seed. Want subsequent runs to be
        # different so this is ok.
        if self.randomise_initial_training:
            initial_override = self.net.random_gen.integers(self.n0s)
            self.net.reset(initial_override=initial_override)
        else:
            self.net.reset()

    def stash_results(self, final_time):
        if self.logging_enabled:
            self.stash_custom()
            self.net.stash_results(final_time=final_time)
            self.reward_lists.append(self.reward_list)
            self.reward_list = []
            self.observation_lists.append(self.observation_list)
            self.observation_list = []
            self.action_lists.append(self.action_list)
            self.action_list = []
            self.time_lists.append(self.time_list)
            self.time_list = []

    def render(self, mode="human"):
        if mode == "human":
            self.net.render()

    def log(self, action, reward, observation):
        if self.logging_enabled:
            self.log_custom()
            self.reward_list.append(reward)
            self.time_list.append(self.time)
            self.observation_list.append(observation)
            self.action_list.append(action)

    def get_df(self):
        if not self.logging_enabled:
            print("Error - DF requested from model with logging disabled")
            assert 0
        out = pd.DataFrame()
        custom_lists = self.get_custom_df_elements()
        for iteration, times_list in enumerate(self.time_lists):
            time_list = self.time_lists[iteration]
            d = {'t': time_list,
                 'reward': self.reward_lists[iteration],
                 'test_iteration': len(time_list) * [iteration]
                 }
            for label, custom_list in custom_lists:
                d[label] = custom_list[iteration]
            observations = self.observation_lists[iteration]
            actions = self.action_lists[iteration]
            for obs_index in range(self.observation_space.shape[0]):
                d['obs'+str(obs_index)] = [o[obs_index] for o in observations]
            for action_index in range(self.action_space.shape[0]):
                d['action'+str(action_index)] = [a[action_index] for a in actions]
            new_data = pd.DataFrame.from_dict(d, orient='index').transpose()
            out = pd.concat([out, new_data])
        return out

    def save_df(self):
        df = self.get_df()
        csv_name = "env_data.csv"
        filename_csv = self.net.log_dir / Path(csv_name)
        df.to_csv(filename_csv)
        return filename_csv

    def step(self, action):
        raise NotImplementedError

    # Subclasses may want to add columns to the main datafile.
    # Can feed them in as a tuple of label + list of lists.
    def get_custom_df_elements(self):
        return []

    def log_custom(self):
        pass

    def stash_custom(self):
        pass

    @staticmethod
    def get_scaled_array(input_array, scaling_array):
        return np.divide(input_array, scaling_array, out=np.zeros_like(input_array), where=scaling_array != 0)
