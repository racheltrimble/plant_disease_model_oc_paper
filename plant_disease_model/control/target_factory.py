import sys
from plant_disease_model.control.plant_disease_target \
    import PlantNetworkFixedControlTarget
from plant_disease_model.control.sensitivity_based_control_target import SensitivityBasedControlTarget, \
    PrioritiseByITarget, PrioritiseBySTarget
from plant_disease_model.control.random_subpop_control import RandomSubPopControl
from plant_disease_model.control.plant_disease_oc_target \
    import PlantNetworkOCTarget, PlantNetworkMPCTarget
from control_ablations.ablation_infra import TrialSettings
from plant_disease_model.control import PlantNetworkTargetSettings
from control_ablations.ablation_infra.trial_settings import CEPATrialSettings
from plant_disease_model import local_control_ablations_config


# This is a static factory
class PlantDiseaseTargetFactory:
    def __init__(self):
        pass

    def get_target_from_file(self, filename):
        trial_settings = TrialSettings.from_file(filename, PlantNetworkTargetSettings)
        return self.get_target_from_trial_settings(trial_settings)

    @staticmethod
    def get_target_from_trial_settings(trial_settings: CEPATrialSettings):
        target_type = trial_settings.get_target_type()
        if target_type == "fixed_control":
            target = PlantNetworkFixedControlTarget(trial_settings)
        elif target_type == "casadi_oc":
            target = PlantNetworkOCTarget(trial_settings)
        elif target_type == "casadi_mpc":
            target = PlantNetworkMPCTarget(trial_settings)
        elif target_type == "sensitivity_control":
            target = SensitivityBasedControlTarget(trial_settings)
        elif target_type == "prioritise_by_s":
            target = PrioritiseBySTarget(trial_settings)
        elif target_type == "prioritise_by_i":
            target = PrioritiseByITarget(trial_settings)
        elif target_type == "random_subpop":
            target = RandomSubPopControl(trial_settings)
        else:
            print("Invalid target type requested from factory")
            assert 0
        return target


if __name__ == "__main__":
    factory = PlantDiseaseTargetFactory()
    t = factory.get_target_from_file(sys.argv[1])
    t.run()
