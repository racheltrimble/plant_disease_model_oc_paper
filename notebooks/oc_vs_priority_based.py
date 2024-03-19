from notebooks.study_in_parts import combine_data_from_study_in_parts
from plant_disease_model.experiments.oc_vs_risk_based import (ControllableGridStudy, StrenuousGridStudy,
                                                              IntermediateGridStudy)
from plant_disease_model.experiments.oc_vs_risk_based import MPCScaleStudy
from plant_disease_model.experiments.oc_vs_risk_based import OCBoundSettingStudy
from plant_disease_model.control import ControllerPlotter
from plant_disease_model.simulator import plotter
from plant_disease_model.simulator.plotter import get_plot_df_legend
from plant_disease_model import local_control_ablations_config


from reporting_utils import *
import numpy as np
import pandas as pd

here = Path(os.path.abspath(''))
data_stash = here / Path('data_stash')
report = here / Path('report')
tiffs = here / Path('tiffs')
bound_setting_data_path = data_stash / Path("bound_setting") / Path("data")
mpc1_path = data_stash / Path("mpc1") / Path("data")
mpc3_path = data_stash / Path("mpc3") / Path("data")
controllable_grid_path_raw = data_stash / Path("4x4_controllable") / Path("data")
controllable_grid_path = data_stash / Path("4x4_controllable") / Path("reconstructed_data")
intermediate_grid_path_raw = data_stash / Path("4x4_intermediate") / Path("data")
intermediate_grid_path = data_stash / Path("4x4_intermediate") / Path("reconstructed_data")
strenuous_grid_path_raw = data_stash / Path("4x4_strenuous") / Path("data")
strenuous_grid_path = data_stash / Path("4x4_strenuous") / Path("reconstructed_data")

OC_bounds_colour_scheme = {"OC proportional": 0,
                           "OC absolute": 7,
                           "equal per host control": 8}

MPC_colour_scheme = {"MPC absolute": 0,
                     "MPC absolute horizon 1": 1,
                     "random subpop": 2,
                     "random subpop rate corrected": 3,
                     "random subpop one shot": 4,
                     "prioritise I": 5,
                     "prioritise S": 6,
                     "OC absolute": 7,
                     "equal per host control": 8,
                     "no control": 9}


def get_data_for_oc_bounds():
    s = OCBoundSettingStudy()
    lots_of_evals = {"eval_run_settings": {"example_plot_repeats": 1000},
                     "generate_tarball": False}
    analyser_settings = {"plot_training_data": False,
                         "compare_key_learning_curve_points": False,
                         "compare_trained_performance": True,
                         "colour_scheme": OC_bounds_colour_scheme,
                         "performance_plot_title": "Violin plot of costs for different controllers",
                         "axis_rename": {"Rewards": "Costs"},
                         "plot_type": "violin"}
    s.run(**lots_of_evals, analyser_settings=analyser_settings)
    move_data_to_stash(bound_setting_data_path)


def get_data_for_mpc_controllable():
    s = MPCScaleStudy(cost_scaling=1.0)
    s.run(generate_tarball=False)
    move_data_to_stash(mpc1_path)


def get_data_for_mpc_strenuous():
    s = MPCScaleStudy(cost_scaling=3.0)
    s.run(generate_tarball=False)
    move_data_to_stash(mpc3_path)


def get_data_for_controllable_in_parts():
    s = ControllableGridStudy()
    s.run(split_eval_runs_into_groups_of=10,
          run_via_command_line=True,
          generate_examples=True,
          run_plotting=False,
          run_analysis=False,
          generate_tarball=False)
    move_data_to_stash(controllable_grid_path_raw)


def get_data_for_intermediate_in_parts():
    s = IntermediateGridStudy()
    s.run(split_eval_runs_into_groups_of=10,
          run_via_command_line=True,
          generate_examples=True,
          run_plotting=False,
          run_analysis=False,
          generate_tarball=False)
    move_data_to_stash(intermediate_grid_path_raw)


def get_data_for_strenuous_in_parts():
    s = StrenuousGridStudy()

    s.run(split_eval_runs_into_groups_of=10,
          run_via_command_line=True,
          generate_examples=True,
          run_plotting=False,
          run_analysis=False,
          generate_tarball=False)

    move_data_to_stash(strenuous_grid_path_raw)


def combine_controllable_data():
    combine_data_from_study_in_parts(controllable_grid_path_raw)
    move_data_to_stash(controllable_grid_path)


def combine_intermediate_data():
    combine_data_from_study_in_parts(intermediate_grid_path_raw)
    move_data_to_stash(intermediate_grid_path)


def combine_strenuous_data():
    combine_data_from_study_in_parts(strenuous_grid_path_raw)
    move_data_to_stash(strenuous_grid_path)

###################


def figure_1(as_png=False):
    # Get the layout plot
    s = setup_existing_data(MPCScaleStudy, mpc1_path)
    settings = {"run_training": False,
                "generate_examples": False,
                "run_plotting": True,
                "run_analysis": True,
                "generate_tarball": False}
    analyser_settings = {"plot_training_data": False,
                         "compare_key_learning_curve_points": False,
                         "compare_trained_performance": False,
                         "sims_with_n_infected_nodes": False,
                         "plot_training_wallclock_times": False,
                         "generate_animations": False,
                         "compare_eradication_times": False,
                         "infected_culled_died": False,
                         "model_illustration_time_display_resolution": 0}
    plotting_run_settings = {"targets_to_plot": ['cull_or_thin_2_by_2_nodes_no_control'],
                             "plot_actions": False}

    s.run(**settings,
          analyser_settings=analyser_settings,
          plotting_run_settings=plotting_run_settings)

    legend_path = report / "figure_1_eg_legend.png"
    get_plot_df_legend(legend_path, i_only=True, with_actions=False, display=False)

    file_list = [
        (analysis_root / Path("cull_or_thin_2_by_2_nodes_no_control") / Path("eval_plot0.png"),
         "no_control_eval.png"),
        (analysis_root / Path("cull_or_thin_2_by_2_nodes") / Path("network_layout.png"),
         "network_layout.png"),
    ]
    move_to_report_dump(file_list, "figure_1", report)

    # Combine layout and eval plot into single figure.
    top_image = report / Path("figure_1_network_layout.png")
    bottom_image = report / Path("figure_1_no_control_eval.png")

    tile_subplots_vertically_to_tiff([[top_image], [bottom_image], [legend_path]],
                                     tiffs,
                                     "figure_1",
                                     new_subfigure=[[True], [True], [False]],
                                     scales=[[1.0], [0.6], [0.6]],
                                     as_png=as_png)

##############


def figure_2(as_png=False):
    s = setup_existing_data(OCBoundSettingStudy, bound_setting_data_path)

    analyser_settings = {"plot_training_data": False,
                         "compare_key_learning_curve_points": False,
                         "compare_trained_performance": True,
                         "sims_with_n_infected_nodes": False,
                         "plot_training_wallclock_times": False,
                         "generate_animations": False,
                         "compare_eradication_times": True,
                         "infected_culled_died": False,
                         "model_illustration": False,
                         "colour_scheme": OC_bounds_colour_scheme,
                         "performance_plot_title": "Violin plot of rewards for different controllers",
                         "plot_type": "violin",
                         "negate_reward_axis": True}

    s.run(**analysis_only,
          analyser_settings=analyser_settings)
    file_list = [
                 (analysis_root / Path("cull_or_thin_2_by_2_nodes") / Path("final_model_rewards_hist.png"),
                  "rewards_hist.png"),
                 (analysis_root / Path("cull_or_thin_2_by_2_nodes") / Path("final_model_eradication.png"),
                  "eradication.png")
                ]
    move_to_report_dump(file_list, "bound_setting", report)

    tile_subplots_vertically_to_tiff([[report / Path("bound_setting_rewards_hist.png"),
                                       report / Path("bound_setting_eradication.png")]],
                                     tiffs,
                                     "figure_2",
                                     new_subfigure=[[True, True]],
                                     scales=[[1.0, 1.0]],
                                     as_png=as_png)

####################


def oc_stochastic_plots(t_max=None, file_postscript=''):
    s = setup_existing_data(OCBoundSettingStudy,
                            bound_setting_data_path)
    plotting_only = {"run_training": False,
                     "run_analysis": False,
                     "generate_examples": False,
                     "run_plotting": True,
                     "generate_tarball": False}
    plotting_run_settings = {"targets_to_plot": ['cull_or_thin_2_by_2_nodes_with_optimal_control',
                                                 "cull_or_thin_2_by_2_nodes_with_optimal_control_direct_rate"],
                             "max_display_iterations": 199,
                             "t_max": t_max}

    s.run(**plotting_only, plotting_run_settings=plotting_run_settings)
    file_list = [
                 (analysis_root / Path("cull_or_thin_2_by_2_nodes_with_optimal_control") / Path("eval_plot0.png"),
                  f"oc_proportional{file_postscript}.png"),
                 (analysis_root / Path("cull_or_thin_2_by_2_nodes_with_optimal_control_direct_rate")
                  / Path("eval_plot0.png"),
                  f"oc_direct{file_postscript}.png")
    ]
    move_to_report_dump(file_list, "stochastic_eg_plots", report)


def oc_predictions_plots(t_max=None, file_postscript=''):
    s = OCBoundSettingStudy()
    training_only = {"run_training": True,
                     "run_analysis": False,
                     "generate_examples": False,
                     "run_plotting": False,
                     "generate_tarball": False}
    s.run(**training_only)
    for test_name, display_name in s.analyser.analysis_spec.test_name_display_name_pairs:
        if test_name in ["cull_or_thin_2_by_2_nodes_with_optimal_control",
                         "cull_or_thin_2_by_2_nodes_with_optimal_control_direct_rate"]:
            this_test_path = data_root / Path(test_name) / Path("0")
            state_df_path = this_test_path / Path("state_df.csv")
            action_df_path = this_test_path / Path("action_df.csv")
            state_df = pd.read_csv(state_df_path)
            action_df = pd.read_csv(action_df_path)
            df_split = plotter.split_by_node(state_df)
            action_df_split = plotter.split_by_node(action_df)
            filename = report / f"deterministic_trace_{test_name}{file_postscript}.png"
            title = "Deterministic Prediction Plot - " + display_name
            plotter.plot_df(df_split,
                            title.title(),
                            str(filename),
                            one_axis=True,
                            action_df_split=action_df_split,
                            display=False,
                            i_only=True,
                            t_max=t_max)


def figure_3(as_png=False):
    oc_stochastic_plots()
    oc_predictions_plots()
    oc_stochastic_plots(t_max=2, file_postscript='_zoomed')
    oc_predictions_plots(t_max=2, file_postscript='_zoomed')
    legend_path = report / "figure_3_eg_legend.png"
    get_plot_df_legend(legend_path,
                       i_only=True,
                       with_actions=True,
                       with_inspect=False,
                       with_thin=True,
                       display=False)
    # # Combine into single figure.
    images = [
              [report / Path("deterministic_trace_cull_or_thin_2_by_2_nodes_with_optimal_control_direct_rate.png")],
              [report / Path("stochastic_eg_plots_oc_direct.png")],
              [report / Path("deterministic_trace_cull_or_thin_2_by_2_nodes_with_optimal_control_direct_rate_zoomed.png")],
              [report / Path("stochastic_eg_plots_oc_direct_zoomed.png")],
              [report / Path("deterministic_trace_cull_or_thin_2_by_2_nodes_with_optimal_control.png")],
              [legend_path]
              ]

    tile_subplots_vertically_to_tiff(images,
                                     tiffs,
                                     "figure_3",
                                     new_subfigure=[[True]]*5 + [[False]],
                                     scales=[[1.0]]*6,
                                     as_png=as_png)

##################


def state_after_first_cycle():
    s = setup_existing_data(OCBoundSettingStudy,
                            bound_setting_data_path)
    analysis_spec = s.analyser.analysis_spec
    for test_name, display_name in analysis_spec.test_name_display_name_pairs:
        c_plotter = ControllerPlotter(test_name, display_name, 0)
        df = c_plotter.get_df()
        about_1_timestep = df['t'][5]
        assert np.isclose(about_1_timestep, 1, atol=0.02)
        t1 = df.loc[df['t'] == about_1_timestep]
        node_extinction_rate = []
        num_nodes = 4
        for node in range(num_nodes):
            node_data = t1.loc[t1['idx'] == node]
            node_extinct_iterations = len(node_data[node_data['nI'] == 0])
            node_extinction_rate += [node_extinct_iterations / len(node_data)]
        extinction_groups = t1.groupby('test_iteration')
        # extinction groups where all nodes are extinct. Divide by number of nodes corrects for
        # the fact that each node has a separate entry in the df.
        overall_extinctions = extinction_groups.filter(lambda x: (x['nI'] == 0).all()) / num_nodes
        extinctions = len(overall_extinctions)
        print(f"{display_name} extinction rates: {node_extinction_rate}, overall: {extinctions} / {len(node_data)}")

####################


def figure_4(as_png=False):

    s = setup_existing_data(MPCScaleStudy, mpc1_path)

    analyser_settings = {"plot_training_data": False,
                         "compare_key_learning_curve_points": False,
                         "compare_trained_performance": True,
                         "sims_with_n_infected_nodes": False,
                         "plot_training_wallclock_times": False,
                         "generate_animations": False,
                         "compare_eradication_times": True,
                         "model_illustration": False,
                         "colour_scheme": MPC_colour_scheme,
                         "performance_plot_title": "Violin plot of rewards for different controllers",
                         "plot_type": "violin",
                         "negate_reward_axis": True,
                         "eradication_data_axis_limits": (0, 20),
                         "infected_culled_died": True,
                         "infected_culled_died_run_combined_plots": False,
                         "infected_culled_died_specific_trajectories": [
                             ("cull_or_thin_2_by_2_nodes_with_mpc_direct_rate_control_horizon1", 70),
                             ("cull_or_thin_2_by_2_nodes_prioritise_i_control", 30)]}

    s.run(**analysis_only,
          analyser_settings=analyser_settings)

    file_list = [
                 (analysis_root / Path("cull_or_thin_2_by_2_nodes") / Path("final_model_rewards_hist.png"),
                  "rewards_hist.png"),
                 (analysis_root / Path("cull_or_thin_2_by_2_nodes") / Path("final_model_eradication.png"),
                  "eradication_times.png"),
                 (analysis_root / Path("cull_or_thin_2_by_2_nodes") /
                  Path("infected_culled_died_cull_or_thin_2_by_2_nodes_prioritise_i_control_30.png"),
                  "prioritise_i_example_infected_culled_died.png"),
                 (analysis_root / Path("cull_or_thin_2_by_2_nodes") /
                  Path("infected_culled_died_cull_or_thin_2_by_2_nodes_with_mpc_direct_rate_control_horizon1_70.png"),
                  "mpc_example_infected_culled_died.png"),
                 (analysis_root / Path("cull_or_thin_2_by_2_nodes") /
                  Path("infected_culled_died_legend.png"),
                  "infected_culled_died_legend.png"),
    ]
    move_to_report_dump(file_list, "mpc_1", report)
    merge_list = [[report / Path("mpc_1_rewards_hist.png"),
                   report / Path("mpc_1_eradication_times.png")],
                  [report / Path("mpc_1_mpc_example_infected_culled_died.png"),
                   report / Path("mpc_1_prioritise_i_example_infected_culled_died.png")],
                  [report / Path("mpc_1_infected_culled_died_legend.png")]]

    tile_subplots_vertically_to_tiff(merge_list,
                                     tiffs,
                                     "figure_4",
                                     new_subfigure=[[True]*2]*2 + [[False]],
                                     scales=[[0.5]*2]*2 + [[0.5]],
                                     as_png=as_png)


def figure_5(as_png=False):
    s = setup_existing_data(MPCScaleStudy, mpc3_path)

    analyser_settings = {"plot_training_data": False,
                         "compare_key_learning_curve_points": False,
                         "compare_trained_performance": True,
                         "sims_with_n_infected_nodes": False,
                         "plot_training_wallclock_times": False,
                         "generate_animations": False,
                         "compare_eradication_times": True,
                         "model_illustration": False,
                         "colour_scheme": MPC_colour_scheme,
                         "performance_plot_title": "Violin plot of rewards for different controllers",
                         "plot_type": "violin",
                         "negate_reward_axis": True,
                         "infected_culled_died": True,
                         "infected_culled_died_run_combined_plots": False,
                         "infected_culled_died_specific_trajectories": [
                             ("cull_or_thin_2_by_2_nodes_with_mpc_direct_rate_control_horizon1", 70),
                             ("cull_or_thin_2_by_2_nodes_prioritise_i_control", 30)],
                         "reward_data_axis_limits": (0, 4000),
                         "eradication_data_axis_limits": (0, 20)}

    s.run(**analysis_only,
          analyser_settings=analyser_settings)

    file_list = [
                 (analysis_root / Path("cull_or_thin_2_by_2_nodes") / Path("final_model_rewards_hist.png"),
                  "rewards_hist.png"),
                 (analysis_root / Path("cull_or_thin_2_by_2_nodes") / Path("final_model_eradication.png"),
                  "eradication_times.png"),
                 (analysis_root / Path("cull_or_thin_2_by_2_nodes") /
                  Path("infected_culled_died_cull_or_thin_2_by_2_nodes_prioritise_i_control_30.png"),
                  "prioritise_i_example_infected_culled_died.png"),
                 (analysis_root / Path("cull_or_thin_2_by_2_nodes") /
                  Path("infected_culled_died_cull_or_thin_2_by_2_nodes_with_mpc_direct_rate_control_horizon1_70.png"),
                  "mpc_example_infected_culled_died.png"),
                 (analysis_root / Path("cull_or_thin_2_by_2_nodes") /
                  Path("infected_culled_died_legend.png"),
                  "infected_culled_died_legend.png"),
    ]
    move_to_report_dump(file_list, "mpc_3", report)

    merge_list = [[report / Path("mpc_3_rewards_hist.png"),
                   report / Path("mpc_3_eradication_times.png")],
                  [report / Path("mpc_3_mpc_example_infected_culled_died.png"),
                   report / Path("mpc_3_prioritise_i_example_infected_culled_died.png")],
                  [report / Path("mpc_3_infected_culled_died_legend.png")]]

    tile_subplots_vertically_to_tiff(merge_list,
                                     tiffs,
                                     "figure_5",
                                     new_subfigure=[[True]*2]*2 + [[False]],
                                     scales=[[0.5]*2]*2 + [[0.5]],
                                     as_png=as_png)

####################


def figure_6(as_png=False):
    gen_grid_study_plots(ControllableGridStudy, controllable_grid_path, "controllable")
    gen_grid_study_plots(IntermediateGridStudy, intermediate_grid_path, "intermediate")
    gen_grid_study_plots(StrenuousGridStudy, strenuous_grid_path, "strenuous")

    merge_list = [[report / Path("large_controllable_grid_rewards_hist.png"),
                   report / Path("large_controllable_grid_eradication.png")],
                  [report / Path("large_intermediate_grid_rewards_hist.png"),
                   report / Path("large_intermediate_grid_eradication.png")],
                  [report / Path("large_strenuous_grid_rewards_hist.png"),
                   report / Path("large_strenuous_grid_eradication.png")]]

    tile_subplots_vertically_to_tiff(merge_list,
                                     tiffs,
                                     "figure_6",
                                     new_subfigure=[[True]*2]*3,
                                     scales=[[1.0]*2]*3,
                                     as_png=as_png,
                                     row_separation_text=["Controllable 4x4 System",
                                                          "Intermediate 4x4 System",
                                                          "Challenging 4x4 System"])


def gen_grid_study_plots(study, path, file_prefix):
    s = setup_existing_data(study, path)

    analyser_settings = {"plot_training_data": False,
                         "compare_key_learning_curve_points": False,
                         "compare_trained_performance": True,
                         "sims_with_n_infected_nodes": False,
                         "plot_training_wallclock_times": False,
                         "generate_animations": False,
                         "compare_eradication_times": True,
                         "infected_culled_died": False,
                         "model_illustration": False,
                         "colour_scheme": MPC_colour_scheme,
                         "performance_plot_title": "Violin plot of rewards for different controllers",
                         "plot_type": "violin",
                         "negate_reward_axis": True,
                         "reward_data_axis_limits": (3_000, 40_000),
                         "eradication_data_axis_limits": (0, 20),
                         "display_n_in_category_labels_reward": False}

    s.run(**analysis_only,
          analyser_settings=analyser_settings)

    file_list = [
                 (analysis_root / Path("cull_or_thin_4_by_4_nodes") / Path("final_model_rewards_hist.png"),
                  "rewards_hist.png"),
                 (analysis_root / Path("cull_or_thin_4_by_4_nodes") / Path("final_model_eradication.png"),
                  "eradication.png"),
    ]
    move_to_report_dump(file_list, f"large_{file_prefix}_grid", report)


def generate_data():
    get_data_for_oc_bounds()
    get_data_for_mpc_controllable()
    get_data_for_mpc_strenuous()
    get_data_for_controllable_in_parts()
    combine_controllable_data()
    get_data_for_intermediate_in_parts()
    combine_intermediate_data()
    get_data_for_strenuous_in_parts()
    combine_strenuous_data()


def generate_report(as_png=False):
    figure_1(as_png)
    figure_2(as_png)
    figure_3(as_png)
    figure_4(as_png)
    figure_5(as_png)
    figure_6(as_png)


if __name__ == "__main__":
    generate_data()
    generate_report(as_png=True)
