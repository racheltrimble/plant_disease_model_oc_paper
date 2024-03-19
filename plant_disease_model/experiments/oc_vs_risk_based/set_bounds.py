import sys
from plant_disease_model.control import PlantDiseaseTargetFactory
from plant_disease_model.analysis import PDAblationStudy
from plant_disease_model.experiments.oc_vs_risk_based.build_grid import build_cull_or_thin_grid
from plant_disease_model.experiments import simple_controls, equal_per_host_control


def cull_or_thin_2_by_2_nodes():
    return build_cull_or_thin_grid(x_nodes=2,
                                   y_nodes=2,
                                   cost_scaling=1.0,
                                   frequency_dependent_betas=False)


def with_optimal_control(target_settings):
    target_settings.controller["type"] = "casadi_oc"
    target_settings.controller["settings"] = {"final_reward_only": False
                                              }
    target_settings.run_limits = {"tuning_valid": True,
                                  "training_valid": True,
                                  "multiple_evals": False}
    target_settings.set_display_name_addendum("OC proportional")
    return target_settings


def with_optimal_control_direct_rate(target_settings):
    target_settings.controller["type"] = "casadi_oc"
    target_settings.controller["settings"] = {"direct_rates": True,
                                              "final_reward_only": False
                                              }
    target_settings.run_limits = {"tuning_valid": True,
                                  "training_valid": True,
                                  "multiple_evals": False}
    target_settings.set_display_name_addendum("OC absolute")
    return target_settings


class OCBoundSettingStudy(PDAblationStudy):
    def __init__(self, baseline_mod=None):
        ablation_list = [
            with_optimal_control,
            with_optimal_control_direct_rate,
        ]

        reward_unchanged_list = []

        # Equivalent env and comparable reward...
        ablation_list_same_interface = [
            with_optimal_control,
            with_optimal_control_direct_rate,
            equal_per_host_control
        ]

        super().__init__(PlantDiseaseTargetFactory(),
                         cull_or_thin_2_by_2_nodes,
                         ablation_list + simple_controls,
                         reward_unchanged_list,
                         ablation_list_same_interface,
                         baseline_mod=baseline_mod,
                         passthrough_label="equal_per_host_control")


if __name__ == "__main__":
    ts = OCBoundSettingStudy()
    ts.run_from_command_line(sys.argv)
    # ts.run(for_cluster=True)
