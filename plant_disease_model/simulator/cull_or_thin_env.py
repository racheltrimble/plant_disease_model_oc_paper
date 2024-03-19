from dataclasses import dataclass
from plant_disease_model.simulator.plant_network_base_env import PlantNetworkBase, PlantNetworkBaseEnvParams
import numpy as np
import gym
from stable_baselines3.common.monitor import Monitor


def make_ct_env(i, sim_setup, log_dir, env_params, training):
    seed = 123 + i
    env_params.training = training

    if env_params.env_format == "OC":
        def _init():
            print("Creating cull or thin gym env for OC")
            env = CullOrThinEnvOCFormat(sim_setup, env_params=env_params, seed=seed, log_dir=log_dir)
            env.seed(seed)
            env = Monitor(env, log_dir, allow_early_resets=True)
            return env
    else:
        print("Unrecognised cull or thin env format requested")
        assert 0

    return _init


@dataclass
class CullOrThinEnvParams(PlantNetworkBaseEnvParams):
    integral_reward: bool = True
    cull_cost: float = 10.0
    thin_cost: float = 10.0
    final_sum_to_infinity: float = None
    env_format: str = "OC"
    max_rate_multiplier_per_node: float = 3.0


# Gym wrapper for meta-population model
class CullOrThinEnv(PlantNetworkBase):

    def __init__(self,
                 setup,  # ... for the epidemic model
                 log_dir=None,
                 seed=123,
                 env_params: CullOrThinEnvParams = None
                 ):
        super().__init__(setup, log_dir, seed, env_params)

        if env_params is None:
            env_params = CullOrThinEnvParams()
        self.dead_count = 0
        self.annual_budget = 100
        self.integral_reward = env_params.integral_reward
        self.final_sum_to_infinity = env_params.final_sum_to_infinity
        # Final sum only makes sense in integral context
        assert self.integral_reward or self.final_sum_to_infinity is None
        self.cull_cost = env_params.cull_cost
        self.thin_cost = env_params.thin_cost
        self.observation_space = self.get_observation_space()
        self.action_space = self.get_action_space()

    def reset(self, seed=None, options=None):
        super().reset()
        self.dead_count = 0
        # Initial observation is always zero as no inspection action has happened.
        observation = self.translate_obs(self.net.get_s_array(), self.net.get_i_array())
        # Initial action is a dummy for logging
        action = np.zeros(self.action_space.shape)

        self.log(action=action, reward=0, observation=observation)
        return observation

    def step(self, action):
        assert self.action_space.contains(action)
        assert(isinstance(action, np.ndarray))
        # treat, progress epidemic, report state

        # Inspect
        cull, thin = self.translate_action(action)

        # Treat
        # No budget constraints on treatment...
        self.net.treat(np.array(cull), self.time)
        self.net.thin(np.array(thin), self.time)
        self.net.progress(self.timestep)
        self.time += self.timestep
        assert(self.time >= self.net.nodes[0].time)

        done = self.get_done()

        reward = self.get_reward(done)

        observation = self.translate_obs(self.net.get_s_array(), self.net.get_i_array())
        self.log(action, reward, observation)
        info = {}

        return observation, reward, done, info

    def get_observation_space(self):
        raise NotImplementedError

    def get_action_space(self):
        raise NotImplementedError

    def translate_obs(self, s_array, i_array):
        raise NotImplementedError

    def translate_action(self, action):
        raise NotImplementedError

    def get_reward(self, done):
        raise NotImplementedError


class CullOrThinEnvOCFormat(CullOrThinEnv):
    def __init__(self,
                 setup,  # ... for the epidemic model
                 log_dir=None,
                 seed=123,
                 env_params: CullOrThinEnvParams = None
                 ):
        if env_params is None:
            env_params = CullOrThinEnvParams()
        max_hosts = 0
        for node in setup["node_setups"]:
            if node['n'] > max_hosts:
                max_hosts = node['n']
        assert max_hosts > 0
        self.max_hosts = max_hosts

        self.max_rate_multiplier_per_node = env_params.max_rate_multiplier_per_node

        # Interface assumes rate based control so catching issues on instantiation.
        assert (setup.get("rate_based_control", False))

        self._check_overspend = True

        super().__init__(setup, log_dir, seed, env_params)

    def get_observation_space(self):
        # Return the number of S and I for each node.
        num_values_per_node = 2

        return gym.spaces.Box(low=-0,
                              high=self.max_hosts,
                              shape=(self.num_nodes * num_values_per_node,),
                              dtype='int16')

    # Action for OC is directly generated as rate of hosts.
    # No need for rounding or conversion to -1 to 1.
    def translate_action(self, action):
        cull = action[0::2]
        thin = action[1::2]
        overspend = cull.sum()*self.cull_cost + thin.sum() * self.thin_cost - self.annual_budget
        # Allow some tolerance on spending.
        # Note - assumes overall spend is e.g. 100 so 1e-3 is still a tight tolerance.
        if self._check_overspend:
            assert(overspend < 1e-3)
        return cull, thin

    def disable_spend_checks(self):
        self._check_overspend = False

    def enable_spend_checks(self):
        self._check_overspend = True

    def get_action_space(self):
        # Action space is a "spend vector" with two elements per node.
        # The first is for cull, the second is for thin.
        # Elements 1 to n+1 define the proportion that should be spent at each node.
        return gym.spaces.Box(low=0.0,
                              high=self.max_rate_multiplier_per_node * self.max_hosts,
                              shape=(self.num_nodes * 2,),
                              dtype='float32')

    def get_reward(self, done):
        # Negative integral of S
        reward = - self.net.get_total_s()
        return reward

    def translate_obs(self, s_array, i_array):
        s = np.array(s_array, dtype=np.int16)
        i = np.array(i_array, dtype=np.int16)
        out = np.append(s, i)
        return out
