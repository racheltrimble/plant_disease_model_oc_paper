import numpy as np
from matplotlib import pyplot as plt
from pathlib import Path

from control_ablations.ablation_infra import AnalysisIO
from plant_disease_model.analysis.plot_blocks.trained_performance_plot_block \
    import TrainedComparisonPlotBlock
from plant_disease_model.control import ControllerPlotter


class SimsWithNInfectedNodesPlotBlock(TrainedComparisonPlotBlock):
    def __init__(self,
                 analysis_spec,
                 io: AnalysisIO,
                 analyser_settings):
        super().__init__(analysis_spec, io, analyser_settings)
        self.title = analyser_settings.get("sims_with_n_infected_nodes_plot_title", "Number of sims with N infected nodes \n vs time (cumulative)")
        self.plot_type = analyser_settings.get("plot_type", "boxplot")
        # Handles is a list of two lists - first is the line objects, second is the labels
        self.handles = [[], []]

    @staticmethod
    def get_trigger():
        return "sims_with_n_infected_nodes"

    # Takes in the trained models from a set of ablation tests and runs them against
    # a common env (which also defines the definitive reward scheme).
    def plot(self):
        if len(self.analysis_spec.get_test_name_list_relevant_to_analysis("sims_with_n_infected_nodes")) == 0:
            "No trained performance data is valid for comparison (different interfaces)"
            return

        dir_test_display = self.get_trained_data()

        # Generates a separate plot for each controller.
        for logdir_root, test_display in dir_test_display:
            test_name, display_name = test_display

            # Returns a list of times and a list of lists (n_infected_nodes_array).
            # Each list in n_infected_nodes_array represents the number of nodes at that time point containing
            # N infected hosts.
            plotter = ControllerPlotter(test_name,
                                        display_name,
                                        0,
                                        logdir_root=logdir_root)
            # Returns a list of times and a list of lists (n_infected_nodes_array).
            # Each list in n_infected_nodes_array represents the number of nodes at that time point containing
            # N infected hosts.
            times, n_infected_nodes_array = plotter.get_sims_with_n_infected_nodes()

            # Histogram:
            filename = self.get_sims_with_n_infected_nodes_filename(test_name)
            fig, ax = plt.subplots()
            ax.set_title(display_name)
            ax.set_xlabel("Time / timesteps")
            ax.set_ylabel("Number of simulations")
            cumulative = np.zeros_like(n_infected_nodes_array[0])
            for idx, a in enumerate(n_infected_nodes_array):
                cumulative += a
                ax.plot(times, cumulative, label=f"less than or equal to {idx}")
            new_handles = ax.get_legend_handles_labels()
            self.handles[0] = new_handles[0]
            self.handles[1] = new_handles[1]
            fig.savefig(str(filename), dpi=fig.dpi)
            plt.close(fig)
        self.generate_legend()

    def get_sims_with_n_infected_nodes_filename(self, test_name):
        return self.io.get_analysis_summary_dir() / Path(f"sims_with_n_infected_nodes_{test_name}.png")

    def generate_legend(self):
        fig, ax = plt.subplots(figsize=[3.5, 2.5])
        ax.legend(handles=self.handles[0], labels=self.handles[1])
        ax.set_title(self.title, fontsize="medium")
        plt.axis("off")
        plt.tight_layout()

        fig.savefig(self.get_sims_with_n_infected_legend_filename(), dpi=fig.dpi)

    def get_sims_with_n_infected_legend_filename(self):
        return self.io.get_analysis_summary_dir() / Path(f"sims_with_n_infected_nodes_legend.png")
