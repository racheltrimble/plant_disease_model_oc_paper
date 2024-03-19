import pandas as pd
from matplotlib import pyplot as plt
import numpy as np

from plant_disease_model.analysis.plot_blocks.trained_performance_plot_block \
    import TrainedComparisonPlotBlock
from plant_disease_model.control import ControllerPlotter


class InfectedCulledDiedPlotBlock(TrainedComparisonPlotBlock):
    def __init__(self,
                 analysis_spec,
                 io,
                 analyser_settings):
        super().__init__(analysis_spec, io, analyser_settings)
        self.specific_trajectories = analyser_settings.get("infected_culled_died_specific_trajectories", None)
        self.run_combined_plots = analyser_settings.get("infected_culled_died_run_combined_plots", True)

    @staticmethod
    def get_trigger():
        return "infected_culled_died"

    # Takes in the trained models from a set of ablation tests and runs them against
    # a common env (which also defines the definitive reward scheme).
    def plot(self):
        dir_test_display = self.get_trained_data()
        directory_lookup = {}

        # Generates a separate plot for each controller.
        if self.run_combined_plots:
            for logdir_root, test_display in dir_test_display:
                test_name, display_name = test_display
                directory_lookup[test_name] = logdir_root

                # Returns a list of times and a list of lists (n_infected_nodes_array).
                # Each list in n_infected_nodes_array represents the number of nodes at that time point containing
                # N infected hosts.
                plotter = ControllerPlotter(test_name,
                                            display_name,
                                            0,
                                            logdir_root=logdir_root)
                action_df = plotter.get_action_df()
                net_df = plotter.get_df()

                filename = self.infected_culled_died_filename(test_name)
                ax, fig = self.plot_setup(display_name)
                total_infected, total_culled, total_removed, \
                    wasted_df, times, action_times = self.extract_data_for_plots(net_df, action_df)
                for test_iteration in total_infected["test_iteration"].unique():
                    self.plot_iteration(test_iteration,
                                        ax,
                                        total_infected,
                                        total_culled,
                                        total_removed,
                                        wasted_df,
                                        times,
                                        action_times)

                fig.savefig(str(filename), dpi=fig.dpi)
                plt.close(fig)
        else:
            for logdir_root, test_display in dir_test_display:
                test_name, display_name = test_display
                directory_lookup[test_name] = logdir_root

        if self.specific_trajectories is not None:
            for test_name, iteration in self.specific_trajectories:
                filename = self.infected_culled_died_specific_filename(test_name, iteration)
                display_name = self.analysis_spec.get_display_name_from_test_name(test_name)
                ax, fig = self.plot_setup(display_name)
                logdir_root = directory_lookup[test_name]
                plotter = ControllerPlotter(test_name,
                                            test_name,
                                            0,
                                            logdir_root=logdir_root)
                action_df = plotter.get_action_df()
                net_df = plotter.get_df()
                total_infected, total_culled, total_removed, \
                    wasted_df, times, action_times = self.extract_data_for_plots(net_df, action_df)
                self.plot_iteration(iteration,
                                    ax,
                                    total_infected,
                                    total_culled,
                                    total_removed,
                                    wasted_df,
                                    times,
                                    action_times,
                                    alpha=1.0)

                fig.savefig(str(filename), dpi=fig.dpi)
                plt.close(fig)
        self.create_legend(ax)

    @staticmethod
    def plot_setup(display_name):
        fig, ax = plt.subplots()
        ax.set_title(display_name)
        ax.set_xlabel("Time / timesteps")
        ax.set_ylabel("Hosts")
        return ax, fig

    @staticmethod
    def extract_data_for_plots(net_df, action_df):
        times = net_df["t"].unique()
        times.sort()
        action_times = action_df["t"].unique()
        action_times.sort()
        total_infected = net_df.groupby(["t", "test_iteration"]).agg({'nI': ['sum']}).reset_index()
        total_culled = action_df.groupby(["t", "test_iteration"]).agg({'cull': ['sum']}).reset_index()
        total_removed = net_df.groupby(["t", "test_iteration"]).agg({'nR': ['sum']}).reset_index()
        time_step = len(times) // len(action_times)

        # Create a wasted culling vector for each node and each iteration.
        # This is aligned to the time resolution of the net_df and then amalgamated for plotting.
        wasted_df = pd.DataFrame()
        for node in net_df["idx"].unique():
            node_culled = action_df[action_df["idx"] == node]
            node_infected = net_df[net_df["idx"] == node]
            for iteration in node_infected["test_iteration"].unique():
                iteration_culled = node_culled.loc[(node_culled["test_iteration"] == iteration) & ~(node_culled["cull"].isna())].sort_values("t")['cull']
                iteration_infected = node_infected.loc[node_infected["test_iteration"] == iteration].sort_values("t")['nI']
                aligned_culled = iteration_culled.values.repeat(time_step)
                wasted_culling_timeslot = iteration_infected == 0
                wasted_culling = wasted_culling_timeslot * aligned_culled
                wasted_culling = pd.DataFrame({"idx": [node]*len(wasted_culling),
                                               "wasted_culling": wasted_culling,
                                               "t": times,
                                               "test_iteration": [iteration]*len(wasted_culling)})
                wasted_df = pd.concat([wasted_df, wasted_culling])
        # Sum over nodes to get the total wasted culling.
        wasted_df = wasted_df.groupby(["t", "test_iteration"]).agg({'wasted_culling': ['sum']}).reset_index()
        return total_infected, total_culled, total_removed, wasted_df, times, action_times

    @staticmethod
    def plot_iteration(test_iteration, ax, total_infected, total_culled, total_removed, wasted_df, times, action_times, alpha=0.1):
        iteration_infected = total_infected.loc[total_infected["test_iteration"] == test_iteration].sort_values("t")[
            'nI']
        iteration_culled = total_culled.loc[total_culled["test_iteration"] == test_iteration].sort_values("t")['cull']
        wasted_culling = wasted_df.loc[wasted_df["test_iteration"] == test_iteration].sort_values("t")['wasted_culling']
        print("Warning - using fixed value for gamma to calculate expected deaths from disease.")
        expected_dead_from_disease = iteration_infected * 0.2
        removed_on_net_time = total_removed.loc[total_removed["test_iteration"] == test_iteration].sort_values("t").diff()['nR']
        time_step = len(times) // len(action_times)
        iteration_removed = []
        iteration_expected_dead = []
        iteration_wasted = []
        # Calculate the removed as a sum and the expected dead / wasted as an average.
        for sample in range(len(action_times)):
            iteration_removed += [removed_on_net_time[sample*time_step:sample*time_step+time_step].sum()]
            iteration_expected_dead += [expected_dead_from_disease[sample*time_step:sample*time_step+time_step].sum() / time_step]
            iteration_wasted += [wasted_culling[sample*time_step:sample*time_step+time_step].sum() / time_step]

        # Get time resolution of removed and culled to be the same. This only works if the time resolution of the
        # action_df and net_df are even multiples.
        iteration_expected_removed = iteration_culled.to_numpy() + np.array(iteration_expected_dead) - np.array(iteration_wasted)

        ax.plot(times, iteration_infected, label="Infected", marker='', color='red', linewidth=0.6, alpha=alpha)
        ax.plot(action_times, iteration_culled, label="Culled", marker='', color='blue', linewidth=0.6, alpha=alpha)
        ax.plot(action_times, iteration_removed, label="Removed", marker='', color='black', linewidth=0.6, alpha=alpha)
        ax.plot(action_times, iteration_expected_removed, label="Expected Removed", marker='', color='grey', linewidth=0.6, alpha=alpha)

    def create_legend(self, ax_of_plot):
        new_handles = ax_of_plot.get_legend_handles_labels()
        handles = new_handles[0]
        labels = new_handles[1]

        fig, ax = plt.subplots(figsize=[3.5, 2.5])
        ax.legend(handles=handles, labels=labels)
        plt.axis("off")
        plt.tight_layout()

        fig.savefig(self.infected_culled_died_legend_filename(), dpi=fig.dpi)

    def infected_culled_died_filename(self, test_name):
        return self.io.get_analysis_summary_dir() / f"infected_culled_died_{test_name}.png"

    def infected_culled_died_legend_filename(self):
        return self.io.get_analysis_summary_dir() / f"infected_culled_died_legend.png"

    def infected_culled_died_specific_filename(self, test_name, iteration):
        return self.io.get_analysis_summary_dir() / f"infected_culled_died_{test_name}_{iteration}.png"
