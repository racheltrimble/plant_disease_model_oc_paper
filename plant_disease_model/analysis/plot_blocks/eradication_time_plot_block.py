from matplotlib import pyplot as plt
from pathlib import Path
import numpy as np

from control_ablations.ablation_infra.analysis_io import AnalysisIO
from plant_disease_model.analysis.plot_blocks.trained_performance_plot_block import TrainedPerformancePlotBlock
from plant_disease_model.control import ControllerPlotter


class EradicationTimePlotBlock(TrainedPerformancePlotBlock):
    def __init__(self,
                 analysis_spec,
                 io: AnalysisIO,
                 analyser_settings):
        super().__init__(analysis_spec, io, analyser_settings)
        self.plot_type = analyser_settings.get("plot_type", "boxplot")
        self.title = analyser_settings.get("eradication_plot_title", "Boxplot of eradication times for different controllers")
        self.plot_type = analyser_settings.get("plot_type", "boxplot")
        self.eradication_data_axis_limits = analyser_settings.get("eradication_data_axis_limits", None)

    @staticmethod
    def get_trigger():
        return "compare_eradication_times"

    # Takes in the trained models from a set of ablation tests and runs them against
    # a common env (which also defines the definitive reward scheme).
    def plot(self):
        if len(self.analysis_spec.get_test_name_list_relevant_to_analysis("compare_eradication_times")) == 0:
            "No trained performance data is valid for comparison (different interfaces)"
            return

        eradication_time_dict = self.get_trained_eradication_times()

        # Histogram:
        filename = self.get_final_model_eradication_filename()
        fig, ax = plt.subplots()
        self.swarmplot_from_dict(ax,
                                 "Eradication Time / cycles",
                                 "Ablation Name",
                                 eradication_time_dict,
                                 self.title,
                                 self.plot_type,
                                 add_n_to_title=False,
                                 drop_nas=False,
                                 data_axis_limits=self.eradication_data_axis_limits)
        fig.savefig(str(filename), dpi=fig.dpi)
        plt.close(fig)

    def get_final_model_eradication_filename(self):
        return self.io.get_analysis_summary_dir() / Path("final_model_eradication.png")

    def get_trained_eradication_times(self, append_n_to_display_name=True):
        time_dict = {}
        dir_test_display = self.get_trained_data()

        for logdir_root, test_display in dir_test_display:
            test_name, display_name = test_display
            plotter = ControllerPlotter(test_name,
                                        display_name,
                                        0,
                                        logdir_root=logdir_root)
            new_times = plotter.get_eradication_times()
            axis_label = display_name
            if append_n_to_display_name:
                axis_label += f"\nN = {len(new_times)}"

            if len(new_times) == 0:
                # This forces the category to be shown on plots even when empty.
                time_dict[axis_label] = [np.nan]
            else:
                time_dict[axis_label] = new_times

        return time_dict
