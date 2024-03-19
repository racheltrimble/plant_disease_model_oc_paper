import sys
from plant_disease_model.control import PlantDiseaseTargetFactory
from plant_disease_model.analysis import PDAblationStudy
from plant_disease_model.experiments.oc_vs_risk_based.build_grid import build_cull_or_thin_grid
from plant_disease_model.experiments import simple_controls, equal_per_host_control


def cull_or_thin_4_by_4_nodes():
    return build_cull_or_thin_grid(x_nodes=4,
                                   y_nodes=4,
                                   cost_scaling=3.0,
                                   frequency_dependent_betas=False)


def with_mpc_direct_rate_control_horizon1(target_settings):
    target_settings.controller["type"] = "casadi_mpc"
    target_settings.controller["settings"] = {"direct_rates": True,
                                              "final_reward_only": False,
                                              "control_horizon": 1
                                              }
    target_settings.run_limits = {"tuning_valid": True,
                                  "training_valid": True,
                                  "multiple_evals": False}
    target_settings.set_display_name_addendum("MPC absolute horizon 1")
    return target_settings


def random_subpop_control(target_settings):
    target_settings.controller["type"] = "random_subpop"
    target_settings.controller["settings"]["rate_correction"] = False

    target_settings.set_display_name_addendum("random subpop")
    return target_settings


def random_subpop_one_shot(target_settings):
    target_settings.controller["type"] = "random_subpop"
    target_settings.set_display_name_addendum("random subpop")
    target_settings.controller["settings"]["recalculate_sensitivities"] = False

    target_settings.set_display_name_addendum("random subpop one shot")
    return target_settings


def prioritise_i_control(target_settings):
    target_settings.controller["type"] = "prioritise_by_i"
    target_settings.controller["settings"]["rate_correction"] = False

    target_settings.set_display_name_addendum("prioritise I")
    return target_settings


def prioritise_s_control(target_settings):
    target_settings.controller["type"] = "prioritise_by_s"
    target_settings.controller["settings"]["rate_correction"] = False

    target_settings.set_display_name_addendum("prioritise S")
    return target_settings


class StrenuousGridStudy(PDAblationStudy):
    def __init__(self, baseline_mod=None):
        ablation_list = [
            with_mpc_direct_rate_control_horizon1,
            random_subpop_control,
            random_subpop_one_shot,
            prioritise_i_control,
            prioritise_s_control
        ]

        reward_unchanged_list = []

        # Equivalent env and comparable reward...
        ablation_list_same_interface = [
            with_mpc_direct_rate_control_horizon1,
            random_subpop_control,
            random_subpop_one_shot,
            prioritise_i_control,
            prioritise_s_control,
            equal_per_host_control
        ]

        super().__init__(PlantDiseaseTargetFactory(),
                         cull_or_thin_4_by_4_nodes,
                         ablation_list + simple_controls,
                         reward_unchanged_list,
                         ablation_list_same_interface,
                         baseline_mod=baseline_mod,
                         passthrough_label="equal_per_host_control")


if __name__ == "__main__":
    ts = StrenuousGridStudy()
    ts.run_from_command_line(sys.argv)
    # ts.run(for_cluster=True)
