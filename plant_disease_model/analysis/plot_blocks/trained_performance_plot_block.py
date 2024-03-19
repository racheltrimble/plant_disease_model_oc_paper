import numpy as np
from matplotlib import pyplot as plt
from pathlib import Path

from control_ablations.ablation_infra import RunSettings, CEPATrialSettings
from plant_disease_model.analysis.plot_blocks.comparison_plot_block import ComparisonPlotBlock
from control_ablations.ablation_infra import AnalysisIO
from plant_disease_model.control \
    import ControllerPlotter, PlantDiseaseTargetFactory, PlantNetworkTargetSettings
from plant_disease_model.simulator import get_iteration_equals


# Base class which works out which models already have data and which need to be run.
# This is determined based on whether the env parametrisation is the same between the baseline and each ablation.
class TrainedComparisonPlotBlock(ComparisonPlotBlock):

    def __init__(self,
                 analysis_spec,
                 io: AnalysisIO,
                 analyser_settings):
        super().__init__(analyser_settings)
        self.io = io
        self.analysis_spec = analysis_spec
        self.force_rerun = analyser_settings.get("force_rerun", False)
        self.perf_comparison_repeats = analyser_settings["perf_comparison_repeats"]

    def plot(self):
        raise NotImplementedError

    @staticmethod
    def get_trigger():
        raise NotImplementedError

    def get_trained_data(self):
        # For each of the trained models for each of the ablations, get the reward against the default setup over
        # multiple repeats
        logdir_root_list = []
        for test_name, display_name in self.analysis_spec.get_relevant_name_pairs(self.get_trigger()):
            baseline_settings = self.analysis_spec.get_passthrough_target_settings()
            ablation_settings = self.analysis_spec.get_target_settings(test_name)

            # If the env is the same between the baseline and the ablation, can just use the
            # existing data.
            # Also need to make sure that the evaluation was actually run
            controller_plotter = ControllerPlotter(test_name,
                                                   display_name,
                                                   0,
                                                   logdir_root=None)

            if self.check_env_aligned(baseline_settings, ablation_settings) and \
                    controller_plotter.has_eval_data():
                logdir_root = None
            else:
                logdir_root = self.get_reward_comparison_data_folder()
                self.generate_comparison(test_name, display_name, ablation_settings, baseline_settings)
            logdir_root_list += [logdir_root]
        dir_test_display = zip(logdir_root_list,
                               self.analysis_spec.get_relevant_name_pairs(self.get_trigger()))
        return dir_test_display


    @staticmethod
    def check_env_aligned(baseline_settings, ablation_settings):
        # Remove the idx field from the node setups as this is not relevant to the comparison
        for idx, setup in enumerate(baseline_settings.sim_setup["node_setups"]):
            setup.pop('idx', None)
            baseline_settings.sim_setup["node_setups"][idx] = setup
        return ((baseline_settings.env_params == ablation_settings.env_params) and
                (baseline_settings.sim_setup == ablation_settings.sim_setup) and
                (baseline_settings.make_env == ablation_settings.make_env))

    def generate_comparison(self, test_name, display_name, ablation_settings, baseline_settings):
        perf_comparison_repeats = self.perf_comparison_repeats
        a_controller = ablation_settings.controller
        target_factory = PlantDiseaseTargetFactory()
        # Only rerun the comparison if required or if forced.
        if not self.reward_comparison_exists(test_name) or self.force_rerun:
            # test_name is the ablated model but the env setup is the baseline
            # This runs the models with the same reward scheme.
            # Return format is direct from stable_baselines3: Returns ([float], [int])
            # first list containing per-episode rewards
            rs = RunSettings()
            rs.add_eval(iterations=0,
                        return_episode_rewards=True,
                        repeats=perf_comparison_repeats,
                        logdir_root=self.get_reward_comparison_data_folder())
            target_settings = PlantNetworkTargetSettings(baseline_settings.env_params,
                                                         baseline_settings.sim_setup,
                                                         a_controller,
                                                         baseline_settings.make_env,
                                                         display_name)
            target_settings.set_display_name_addendum(ablation_settings.display_name_addendum)
            ts = CEPATrialSettings(test_name, rs, target_settings)

            target = target_factory.get_target_from_trial_settings(ts)

            target.run()

    def get_reward_comparison_data_folder(self):
        return self.io.get_data_dir() / Path(f"{self.io.summary_name}_post") / Path("reward_comparison_data")


    def reward_comparison_exists(self, test_name):
        folder_to_check = self.get_reward_comparison_data_folder()
        if not folder_to_check.exists():
            return False
        subdirectories = [x for x in folder_to_check.iterdir() if x.is_dir() and x.parts[-1] == test_name]
        return len(subdirectories) > 0


class TrainedPerformancePlotBlock(TrainedComparisonPlotBlock):
    def __init__(self,
                 analysis_spec,
                 io: AnalysisIO,
                 analyser_settings):
        super().__init__(analysis_spec, io, analyser_settings)
        self.explain_good_performance = analyser_settings.get("explain_good_performance", None)
        self.title = analyser_settings.get("performance_plot_title", "Boxplot of rewards for different controllers")
        self.axis_rename = analyser_settings.get("axis_rename", {})
        self.plot_type = analyser_settings.get("plot_type", "boxplot")
        self.negate_reward_axis = analyser_settings.get("negate_reward_axis", False)
        self.reward_data_axis_limits = analyser_settings.get("reward_data_axis_limits", None)
        self.generate_trained_performance_tables = analyser_settings.get("generate_trained_performance_tables", False)
        default_display_n_in_category_labels = self.reward_data_axis_limits is not None
        self.display_n_in_category_labels_reward = analyser_settings.get("display_n_in_category_labels_reward",
                                                                         default_display_n_in_category_labels)

    @staticmethod
    def get_trigger():
        return "compare_trained_performance"

    # Takes in the trained models from a set of ablation tests and runs them against
    # a common env (which also defines the definitive reward scheme).
    def plot(self):
        if len(self.analysis_spec.get_test_name_list_relevant_to_analysis("compare_trained_performance")) == 0:
            "No trained performance data is valid for comparison (different interfaces)"
            return

        rewards_dict = self.get_trained_performance()

        # Histogram:
        filename = self.get_final_model_hist_filename()
        fig, ax = plt.subplots()
        if "Rewards" in self.axis_rename:
            reward_label = self.axis_rename["Rewards"]
        else:
            reward_label = "Rewards"

        # If all data points are shown, just use N in the title. Otherwise, plots labelled individually.
        add_n_to_title = self.reward_data_axis_limits is None
        self.swarmplot_from_dict(ax,
                                 reward_label,
                                 "Ablation Name",
                                 rewards_dict,
                                 self.title,
                                 self.plot_type,
                                 axis_rename=self.axis_rename,
                                 data_axis_limits=self.reward_data_axis_limits,
                                 add_n_to_title=add_n_to_title)
        fig.savefig(str(filename), dpi=fig.dpi)
        plt.close(fig)

        if self.explain_good_performance is not None:
            for display_name in rewards_dict:
                rewards = rewards_dict[display_name]
                for index, r in enumerate(rewards):
                    if r > self.explain_good_performance:
                        test_name = self.analysis_spec.get_test_name_from_display_name(display_name)
                        plotter = ControllerPlotter(test_name, display_name, 0)
                        plot_filter = get_iteration_equals(index)
                        print("Plotting good performance plot for ", test_name, " index ", index)
                        plotter.plot_eval(display=False, plot_filter=plot_filter)

        if self.generate_trained_performance_tables:
            # These two options are currently incompatible. Would need to regen data without N values in labels.
            assert(self.reward_data_axis_limits is None)
            comparison_list = self.analysis_spec.get_relevant_comparison("compare_trained_performance")
            baseline_test_name = self.analysis_spec.get_passthrough_test_name()
            full_data, table_data, _ = self.pairwise_comparison(baseline_test_name, comparison_list, rewards_dict)
            filename = self.get_reward_comparison_full_latex_filename()
            self.write_latex_table(full_data, filename)
            filename = self.get_reward_comparison_latex_filename()
            self.write_latex_table(table_data, filename)

    def get_reward_comparison_full_latex_filename(self):
        return self.io.get_analysis_summary_dir() / Path("reward_comparison_full.tex")

    def get_reward_comparison_latex_filename(self):
        return self.io.get_analysis_summary_dir() / Path("reward_comparison_latex.tex")

    def get_final_model_hist_filename(self):
        return self.io.get_analysis_summary_dir() / Path("final_model_rewards_hist.png")

    def get_trained_performance(self):
        rewards_dict = {}
        dir_test_display = self.get_trained_data()

        for logdir_root, test_display in dir_test_display:
            test_name, display_name = test_display
            plotter = ControllerPlotter(test_name,
                                        display_name,
                                        0,
                                        logdir_root=logdir_root)
            new_rewards = plotter.read_reward_file(logdir_root=logdir_root)
            if self.negate_reward_axis:
                new_rewards = -np.array(new_rewards)
                new_rewards = list(new_rewards)

            axis_label = display_name
            if self.display_n_in_category_labels_reward:
                rewards_array = np.array(new_rewards)
                valid_data_array = np.logical_and(self.reward_data_axis_limits[0] < rewards_array,
                                                  rewards_array < self.reward_data_axis_limits[1])
                number_of_samples = valid_data_array.sum()
                axis_label += f"\nN = {number_of_samples}"
            rewards_dict[axis_label] = new_rewards
        return rewards_dict
