import dataclasses
from control_ablations.ablation_infra import CEPATargetSettings
from plant_disease_model.simulator import make_ct_env
from plant_disease_model.simulator import CullOrThinEnvParams


class PlantNetworkTargetSettings(CEPATargetSettings):

    @classmethod
    def env_params_from_dict(cls, env_type, env_dict):
        if env_type == "CullOrThin":
            env_params = CullOrThinEnvParams(**env_dict)
        else:
            assert 0
        return env_params

    @classmethod
    def sim_setup_from_dict(cls, sim_dict):
        temp_setup = sim_dict
        setup = temp_setup
        node_setups = []
        old_nodes = temp_setup['node_setups']
        for idx in range(len(old_nodes)):
            node_setups.append(old_nodes[idx])
        setup['node_setups'] = node_setups
        return setup

    @classmethod
    def controller_from_dict(cls, controller_dict):
        return controller_dict

    @classmethod
    def make_env_from_string(cls, make_env_str):
        return cls.string_to_function(make_env_str)

    def env_params_to_dict(self):
        if isinstance(self.env_params, CullOrThinEnvParams):
            env_type = "CullOrThin"
        else:
            print("Unsupported env type")
            assert 0
        return env_type, dataclasses.asdict(self.env_params)

    def sim_setup_to_dict(self):
        # Intermediate depth copy. This is ok as only changing at first level anyway.
        edit_setup = {key: value for key, value in self.sim_setup.items()}
        # Convert node setups to a dictionary
        new_node_setups = dict((index, data) for index, data in enumerate(self.sim_setup['node_setups']))
        edit_setup['node_setups'] = new_node_setups
        return edit_setup

    def controller_to_dict(self):
        return self.controller

    def make_env_to_string(self):
        return self.function_to_string(self.make_env)

    @staticmethod
    def string_to_function(f_name):
        if f_name == "make_ct_env":
            return make_ct_env
        else:
            print("Unknown make_env passed to TrialSettings")
            assert 0

    @staticmethod
    def function_to_string(function):
        return function.__name__
