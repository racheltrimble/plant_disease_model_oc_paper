from pathlib import Path
import pandas as pd
from plant_disease_model.simulator import plotter
from control_ablations.ablation_infra import PerIterationIO


# Deals with per iteration plotting i.e. performance of individual trained agents.
class ControllerPlotter:
    def __init__(self, test_name, display_name, iteration, logdir_root=None):
        self.test_name = test_name
        self.display_name = display_name
        self.iteration = iteration
        self.io = PerIterationIO(test_name)
        self.io.set_iteration(iteration)
        self.logdir_root = logdir_root

    def has_eval_data(self):
        filename = self.get_df_file_path()
        return filename.exists()

    def get_df_file_path(self):
        return self.io.get_eval_dir(logdir_root=self.logdir_root) / Path("net_data_res_100.csv")

    def get_action_file_path(self):
        return self.io.get_eval_dir(logdir_root=self.logdir_root) / Path("action_data.csv")

    def get_env_df_path(self):
        return self.io.get_eval_dir(logdir_root=self.logdir_root) / Path("env_data.csv")

    def get_eval_plot_filename(self):
        return self.io.get_analysis_dir() / Path("eval_plot" + str(self.iteration) + ".png")

    def get_env_plot_filename(self):
        return self.io.get_analysis_dir() / Path("env_plot" + str(self.iteration) + ".png")

    def get_baseline_plot_filename(self):
        return self.io.get_analysis_dir() / Path("baseline_plot.png")

    def get_eval_plot_filename_for_intermediate(self, training_point, eval_num):
        return self.io.get_analysis_dir() / Path(f"eval_plot{self.iteration}_{training_point}_{eval_num}.png")

    def example_data_exists(self):
        return self.get_df_file_path().exists()

    def read_reward_file(self, logdir_root=None):
        return self.io.read_reward_file(logdir_root=logdir_root)

    def get_action_df(self):
        return pd.read_csv(self.get_action_file_path())

    def get_df(self):
        return pd.read_csv(self.get_df_file_path())

    def get_eradication_times(self):
        return plotter.get_eradication_times(self.get_df())

    def get_sims_with_n_infected_nodes(self):
        return plotter.get_sims_with_n_infected_nodes(self.get_df())

    def plot_eval(self,
                  display=True,
                  plot_filter=None,
                  i_only=True,
                  max_iterations=None,
                  plot_actions=True,
                  t_max=None):
        self.plot_main(plot_filter,
                       max_iterations,
                       display,
                       i_only,
                       plot_actions,
                       t_max)
        self.plot_env(display)

    def plot_main(self,
                  plot_filter,
                  max_iterations,
                  display,
                  i_only,
                  plot_actions,
                  t_max=None):
        df = self.get_df()
        df_split = plotter.split_by_node(df)
        action_df = self.get_action_df()
        action_df_split = plotter.split_by_node(action_df)
        if plot_filter is not None:
            # Got through each node and filter for cases where the initial infected is in that node.
            matching_split = {}
            matching_action_split = {}
            for index in df_split:
                node_df = df_split[index]
                node_action_df = action_df_split[index]
                plotter_out = plotter.split_test_iterations_by_criterion(node_df,
                                                                         node_action_df,
                                                                         plot_filter.filter_func)
                matching, not_matching, action_matching, action_not_matching = plotter_out
                matching_split[index] = matching
                matching_action_split[index] = action_matching
            # Overwrite with the filtered values
            df_split = matching_split
            action_df_split = matching_action_split

        filename = self.get_eval_plot_filename()
        title = "Evaluation plot - " + self.display_name
        if max_iterations is not None:
            iterations = list(range(max_iterations+1))
        else:
            iterations = None

        # Not efficient way to do this but simple
        if not plot_actions:
            action_df_split = None

        plotter.plot_df(df_split,
                        title.title(),
                        str(filename),
                        one_axis=True,
                        action_df_split=action_df_split,
                        display=display,
                        i_only=i_only,
                        iterations=iterations,
                        t_max=t_max)

    def plot_env(self, display):
        env_df = pd.read_csv(self.get_env_df_path())
        filename = self.get_env_plot_filename()
        plotter.plot_env(env_df,
                         "Env level",
                         str(filename),
                         display=display)
