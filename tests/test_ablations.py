from plant_disease_model.experiments import TinyCullOrThin
from plant_disease_model.experiments import OCBoundSettingStudy

from plant_disease_model import local_control_ablations_config

mini_ablation_config = {"repeats": 1,
                        "tar_data": False,
                        "eval_run_settings": {"example_plot_repeats": 2},
                        "perf_comparison_repeats": 10}

analysis_only = {"run_training": False,
                 "generate_examples": False,
                 "run_plotting": False}


def test_tiny_cull_or_thin():
    ts = TinyCullOrThin()
    ablation_config = mini_ablation_config
    ablation_config["repeats"] = 2
    ts.run(**ablation_config)


def test_tiny_study_split_eval():
    ts = TinyCullOrThin()
    ts.run(repeats=1, run_via_command_line=True, split_eval_runs_into_groups_of=10)


def test_oc():
    two_mini_ablation_config = {"repeats": 2,
                                "tar_data": False,
                                "eval_run_settings": {"example_plot_repeats": 2},
                                "perf_comparison_repeats": 10}

    oc = OCBoundSettingStudy()
    oc.run(**two_mini_ablation_config)
