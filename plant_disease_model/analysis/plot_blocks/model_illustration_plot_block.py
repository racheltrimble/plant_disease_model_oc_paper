from pathlib import Path

from control_ablations.ablation_infra import AnalysisSpec, AnalysisIO
from plant_disease_model.analysis.plot_blocks.trained_performance_plot_block import \
    TrainedPerformancePlotBlock
from plant_disease_model.simulator import draw_xy_visual_set_across_time
from plant_disease_model.simulator import Network
from plant_disease_model.control import ControllerPlotter

class ModelIllustrationPlotBlock(TrainedPerformancePlotBlock):
    def __init__(self,
                 analysis_spec: AnalysisSpec,
                 io: AnalysisIO,
                 analyser_settings):
        super().__init__(analysis_spec, io, analyser_settings)
        self.io = io
        self.analysis_spec = analysis_spec
        self.test_name_for_illustration = analysis_spec.get_passthrough_test_name()
        self.time_display_resolution = analyser_settings.get("model_illustration_time_display_resolution", 2)

    def plot(self):
        baseline_settings = self.analysis_spec.get_passthrough_target_settings()

        setup = baseline_settings.sim_setup
        net = Network(setup["node_setups"],
                      setup['node_locations'],
                      setup['link_setups'],
                      setup['aerial_setups'],
                      initial_infected_random=setup['initial_infected_random'])
        controller_plotter = ControllerPlotter(self.test_name_for_illustration,
                                               "",
                                               0,
                                               logdir_root=None)
        df = controller_plotter.get_df()
        net_dict_def = net.get_dict_def()

        draw_xy_visual_set_across_time(df,
                                       net_dict_def['node_definitions'],
                                       subplot_size_factor=400,
                                       subplot_aspect_ratio=1.0,
                                       num_plots=None,
                                       target_times=[0, 2, 5],
                                       subplot_type="pie",
                                       filename=self.get_network_layout_picture_filename(),
                                       time_display_resolution=self.time_display_resolution)


    def get_network_layout_picture_filename(self):
        return self.io.get_analysis_summary_dir() / Path("network_layout.png")


    @staticmethod
    def get_trigger():
        return "model_illustration"
