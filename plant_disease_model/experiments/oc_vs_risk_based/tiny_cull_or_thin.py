import sys
from plant_disease_model.control import PlantDiseaseTargetFactory
from plant_disease_model.analysis import PDAblationStudy
from plant_disease_model.experiments.oc_vs_risk_based.build_grid import build_cull_or_thin_grid
from plant_disease_model.experiments import simple_controls, baseline


def cull_or_thin_2_nodes():
    return build_cull_or_thin_grid(x_nodes=2, y_nodes=1)


class TinyCullOrThin(PDAblationStudy):
    def __init__(self, baseline_mod=None):
        ablation_list = [
            baseline
        ]

        reward_unchanged_list = []

        # Equivalent env and comparable reward...
        ablation_list_same_interface = []

        super().__init__(PlantDiseaseTargetFactory(),
                         cull_or_thin_2_nodes,
                         ablation_list + simple_controls,
                         reward_unchanged_list,
                         ablation_list_same_interface,
                         baseline_mod=baseline_mod)


if __name__ == "__main__":
    ts = TinyCullOrThin()
    ts.run_from_command_line(sys.argv)
    # ts.run(for_cluster=True)
