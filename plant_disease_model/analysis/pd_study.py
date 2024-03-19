from control_ablations.ablation_infra import AblationStudyWithAnalysis
from plant_disease_model.analysis.plot_blocks import TrainedPerformancePlotBlock
from plant_disease_model.analysis.plot_blocks import ModelIllustrationPlotBlock
from plant_disease_model.analysis.plot_blocks import EradicationTimePlotBlock
from plant_disease_model.analysis.plot_blocks import SimsWithNInfectedNodesPlotBlock
from plant_disease_model.analysis.plot_blocks import InfectedCulledDiedPlotBlock
from plant_disease_model.analysis.plot_blocks import GitTagPlotBlock


class PDAblationStudy(AblationStudyWithAnalysis):

    def __init__(self,
                 target_factory,
                 baseline,
                 ablation_list,
                 reward_unchanged_list,
                 ablation_list_same_interface,
                 per_agent_task_duration_in_hours=1,
                 iterations_for_intermediate_animations=None,
                 run_untrained=False,
                 baseline_mod=None,
                 passthrough_label="baseline"
                 ):

        common_properties = {"compare_key_learning_curve_points": {"points_list": [0.5, 1.0]}}

        animation_settings = {"show_control_in_animations": True,
                              "run_untrained": run_untrained,
                              "iterations_for_intermediate_animations": iterations_for_intermediate_animations}
        relevant_analysis = {"compare_key_learning_points": (reward_unchanged_list, {}),
                             "compare_trained_performance": (ablation_list_same_interface, {}),
                             "plot_training_data": (ablation_list, {}),
                             "plot_training_wallclock_times": (ablation_list, {}),
                             "generate_animations": (ablation_list, animation_settings),
                             "model_illustration": ([], {}),
                             "infected_culled_died": (ablation_list, {}),
                             "compare_eradication_times": (ablation_list_same_interface, {}),
                             "sims_with_n_infected_nodes": (ablation_list_same_interface, {})
                             }
        plotter_block_class_list = [
                                    TrainedPerformancePlotBlock,
                                    ModelIllustrationPlotBlock,
                                    EradicationTimePlotBlock,
                                    SimsWithNInfectedNodesPlotBlock,
                                    GitTagPlotBlock,
                                    InfectedCulledDiedPlotBlock
                                    ]

        super().__init__(target_factory,
                         baseline,
                         baseline_mod,
                         ablation_list,
                         common_properties,
                         relevant_analysis,
                         self.apply_overrides,
                         plotter_block_class_list,
                         passthrough_label=passthrough_label)
        self.per_agent_task_duration_in_hours = per_agent_task_duration_in_hours

    @staticmethod
    def apply_overrides(raw_settings, ablation, analysis_type):
        if ablation.__name__ in ["no_control"] and analysis_type == "iterations_for_intermediate_animations":
            settings = raw_settings.copy()
            settings["show_control_in_animations"] = False
        else:
            settings = raw_settings
        return settings

    def get_per_agent_task_duration_in_hours(self):
        return self.per_agent_task_duration_in_hours
