import pandas as pd
import numpy as np

from control_ablations.ablation_infra import RunSettings, CEPATrialSettings

from plant_disease_model.simulator import plotter
from plant_disease_model.control import (ControllerPlotter,
                                         PlantNetworkFixedControlTarget,
                                         SensitivityBasedControlTarget)
from plant_disease_model.experiments.oc_vs_risk_based import build_cull_or_thin_grid


def test_no_control_target():
    target_settings = build_cull_or_thin_grid(x_nodes=1, y_nodes=2)
    # Not strictly necessary to do this but tidier!
    target_settings.controller["type"] = "fixed_control"
    target_settings.controller["settings"] = {"action": "equal"}
    target_settings.set_display_name_addendum("test")
    run_settings = RunSettings()
    run_settings.add_eval(iterations=0)
    run_settings.add_plotting(iterations=0)
    ts = CEPATrialSettings("test_no_control", run_settings, target_settings)
    fixed = PlantNetworkFixedControlTarget(trial_settings=ts)
    fixed.run()


def test_per_host_scaled_control_target():
    target_settings = build_cull_or_thin_grid(x_nodes=1, y_nodes=2)
    target_settings.controller["type"] = "fixed_control"
    target_settings.controller["settings"] = {"action": "equal_per_host"}

    # Change first node population to be 200 rather than 100.
    target_settings.sim_setup["node_setups"][0]["n"] = 200
    target_settings.set_display_name_addendum("test")
    run_settings = RunSettings()
    run_settings.add_eval(iterations=0)
    ts = CEPATrialSettings("test_equal_per_host", run_settings, target_settings)
    fixed = PlantNetworkFixedControlTarget(trial_settings=ts)
    fixed.run()
    cp = ControllerPlotter(ts.test_name, ts.test_name, 0)
    cp.plot_eval(0)
    action_df = pd.read_csv(cp.get_action_file_path())
    action_df_split = plotter.split_by_node(action_df)
    # Environment rounds down the inspection before logging if there aren't enough hosts so can only
    # guarantee for initial inspections.
    actions_series_0 = action_df_split[0].loc[action_df_split[0]['t'] == 0]['inspect'].values
    actions_series_1 = action_df_split[1].loc[action_df_split[1]['t'] == 0]['inspect'].values
    assert(np.logical_or((actions_series_0 == 33), np.isnan(actions_series_0)).all())
    assert(np.logical_or((actions_series_1 == 16), np.isnan(actions_series_1)).all())


def test_per_host_scaled_control_target_oc():
    target_settings = build_cull_or_thin_grid(x_nodes=2, y_nodes=2)
    target_settings.env_params.env_format = "OC"
    target_settings.env_params.cull_cost = 1
    target_settings.env_params.thin_cost = 1
    target_settings.sim_setup["rate_based_control"] = True
    target_settings.controller["type"] = "fixed_control"
    target_settings.controller["settings"] = {"action": "equal_per_host"}

    # HArd code the node populations to be 100, 200, 300, 400.
    target_settings.sim_setup["node_setups"][0]["n"] = 100
    target_settings.sim_setup["node_setups"][1]["n"] = 200
    target_settings.sim_setup["node_setups"][2]["n"] = 300
    target_settings.sim_setup["node_setups"][3]["n"] = 400
    target_settings.set_display_name_addendum("test")
    run_settings = RunSettings()
    run_settings.add_eval(iterations=0)
    ts = CEPATrialSettings("test_equal_per_host", run_settings, target_settings)
    fixed = PlantNetworkFixedControlTarget(trial_settings=ts)
    fixed.run()
    cp = ControllerPlotter(ts.test_name, ts.test_name, 0)
    cp.plot_eval(0)
    action_df = pd.read_csv(cp.get_action_file_path())
    action_df_split = plotter.split_by_node(action_df)
    # Environment rounds down the inspection before logging if there aren't enough hosts so can only
    # guarantee for initial inspections.
    actions_series_0 = action_df_split[0].loc[action_df_split[0]['t'] == 0]['cull'].values
    actions_series_1 = action_df_split[1].loc[action_df_split[1]['t'] == 0]['cull'].values
    actions_series_2 = action_df_split[2].loc[action_df_split[2]['t'] == 0]['cull'].values
    actions_series_3 = action_df_split[3].loc[action_df_split[3]['t'] == 0]['cull'].values

    assert(np.logical_or((actions_series_0 == 5), np.isnan(actions_series_0)).all())
    assert(np.logical_or((actions_series_1 == 10), np.isnan(actions_series_1)).all())
    assert(np.logical_or((actions_series_2 == 15), np.isnan(actions_series_2)).all())
    assert(np.logical_or((actions_series_3 == 20), np.isnan(actions_series_3)).all())

    actions_series_0 = action_df_split[0].loc[action_df_split[0]['t'] == 0]['thin'].values
    actions_series_1 = action_df_split[1].loc[action_df_split[1]['t'] == 0]['thin'].values
    actions_series_2 = action_df_split[2].loc[action_df_split[2]['t'] == 0]['thin'].values
    actions_series_3 = action_df_split[3].loc[action_df_split[3]['t'] == 0]['thin'].values

    assert(np.logical_or((actions_series_0 == 5), np.isnan(actions_series_0)).all())
    assert(np.logical_or((actions_series_1 == 10), np.isnan(actions_series_1)).all())
    assert(np.logical_or((actions_series_2 == 15), np.isnan(actions_series_2)).all())
    assert(np.logical_or((actions_series_3 == 20), np.isnan(actions_series_3)).all())


def test_per_host_scaled_control_target_cull_only():
    target_settings = build_cull_or_thin_grid(x_nodes=2, y_nodes=2)
    target_settings.env_params.env_format = "OC"
    target_settings.env_params.cull_cost = 1
    target_settings.env_params.thin_cost = 1
    target_settings.sim_setup["rate_based_control"] = True
    target_settings.controller["type"] = "fixed_control"
    target_settings.controller["settings"] = {"action": "equal_per_host_cull"}

    # HArd code the node populations to be 100, 200, 300, 400.
    target_settings.sim_setup["node_setups"][0]["n"] = 100
    target_settings.sim_setup["node_setups"][1]["n"] = 200
    target_settings.sim_setup["node_setups"][2]["n"] = 300
    target_settings.sim_setup["node_setups"][3]["n"] = 400
    target_settings.set_display_name_addendum("test")
    run_settings = RunSettings()
    run_settings.add_eval(iterations=0)
    ts = CEPATrialSettings("test_equal_cull_per_host", run_settings, target_settings)
    fixed = PlantNetworkFixedControlTarget(trial_settings=ts)
    fixed.run()
    cp = ControllerPlotter(ts.test_name, ts.test_name, 0)
    cp.plot_eval(0)
    action_df = pd.read_csv(cp.get_action_file_path())
    action_df_split = plotter.split_by_node(action_df)
    # Environment rounds down the inspection before logging if there aren't enough hosts so can only
    # guarantee for initial inspections.
    actions_series_0 = action_df_split[0].loc[action_df_split[0]['t'] == 0]['cull'].values
    actions_series_1 = action_df_split[1].loc[action_df_split[1]['t'] == 0]['cull'].values
    actions_series_2 = action_df_split[2].loc[action_df_split[2]['t'] == 0]['cull'].values
    actions_series_3 = action_df_split[3].loc[action_df_split[3]['t'] == 0]['cull'].values

    assert(np.logical_or((actions_series_0 == 10), np.isnan(actions_series_0)).all())
    assert(np.logical_or((actions_series_1 == 20), np.isnan(actions_series_1)).all())
    assert(np.logical_or((actions_series_2 == 30), np.isnan(actions_series_2)).all())
    assert(np.logical_or((actions_series_3 == 40), np.isnan(actions_series_3)).all())

    actions_series_0 = action_df_split[0].loc[action_df_split[0]['t'] == 0]['thin'].values
    actions_series_1 = action_df_split[1].loc[action_df_split[1]['t'] == 0]['thin'].values
    actions_series_2 = action_df_split[2].loc[action_df_split[2]['t'] == 0]['thin'].values
    actions_series_3 = action_df_split[3].loc[action_df_split[3]['t'] == 0]['thin'].values

    assert(np.logical_or((actions_series_0 == 0), np.isnan(actions_series_0)).all())
    assert(np.logical_or((actions_series_1 == 0), np.isnan(actions_series_1)).all())
    assert(np.logical_or((actions_series_2 == 0), np.isnan(actions_series_2)).all())
    assert(np.logical_or((actions_series_3 == 0), np.isnan(actions_series_3)).all())


def test_per_pop_target_cull_only():
    target_settings = build_cull_or_thin_grid(x_nodes=2, y_nodes=2)
    target_settings.env_params.env_format = "OC"
    target_settings.env_params.cull_cost = 1
    target_settings.env_params.thin_cost = 1
    target_settings.sim_setup["rate_based_control"] = True
    target_settings.controller["type"] = "fixed_control"
    target_settings.controller["settings"] = {"action": "equal_cull"}

    # HArd code the node populations to be 100, 200, 300, 400.
    target_settings.sim_setup["node_setups"][0]["n"] = 100
    target_settings.sim_setup["node_setups"][1]["n"] = 200
    target_settings.sim_setup["node_setups"][2]["n"] = 300
    target_settings.sim_setup["node_setups"][3]["n"] = 400
    target_settings.set_display_name_addendum("test")
    run_settings = RunSettings()
    run_settings.add_eval(iterations=0)
    ts = CEPATrialSettings("test_equal_cull_per_host", run_settings, target_settings)
    fixed = PlantNetworkFixedControlTarget(trial_settings=ts)
    fixed.run()
    cp = ControllerPlotter(ts.test_name, ts.test_name, 0)
    cp.plot_eval(0)
    action_df = pd.read_csv(cp.get_action_file_path())
    action_df_split = plotter.split_by_node(action_df)
    # Environment rounds down the inspection before logging if there aren't enough hosts so can only
    # guarantee for initial inspections.
    actions_series_0 = action_df_split[0].loc[action_df_split[0]['t'] == 0]['cull'].values
    actions_series_1 = action_df_split[1].loc[action_df_split[1]['t'] == 0]['cull'].values
    actions_series_2 = action_df_split[2].loc[action_df_split[2]['t'] == 0]['cull'].values
    actions_series_3 = action_df_split[3].loc[action_df_split[3]['t'] == 0]['cull'].values

    assert(np.logical_or((actions_series_0 == 25), np.isnan(actions_series_0)).all())
    assert(np.logical_or((actions_series_1 == 25), np.isnan(actions_series_1)).all())
    assert(np.logical_or((actions_series_2 == 25), np.isnan(actions_series_2)).all())
    assert(np.logical_or((actions_series_3 == 25), np.isnan(actions_series_3)).all())

    actions_series_0 = action_df_split[0].loc[action_df_split[0]['t'] == 0]['thin'].values
    actions_series_1 = action_df_split[1].loc[action_df_split[1]['t'] == 0]['thin'].values
    actions_series_2 = action_df_split[2].loc[action_df_split[2]['t'] == 0]['thin'].values
    actions_series_3 = action_df_split[3].loc[action_df_split[3]['t'] == 0]['thin'].values

    assert(np.logical_or((actions_series_0 == 0), np.isnan(actions_series_0)).all())
    assert(np.logical_or((actions_series_1 == 0), np.isnan(actions_series_1)).all())
    assert(np.logical_or((actions_series_2 == 0), np.isnan(actions_series_2)).all())
    assert(np.logical_or((actions_series_3 == 0), np.isnan(actions_series_3)).all())


def test_multi_action_per_host_scaled_control_target():
    target_settings = build_cull_or_thin_grid(x_nodes=1, y_nodes=2)
    target_settings.controller["type"] = "fixed_control"
    target_settings.controller["settings"] = {"action": "equal_per_host"}

    # Change first node population to be 200 rather than 100.
    # target_settings.sim_setup["node_setups"][0]["n"] = 200
    target_settings.set_display_name_addendum("test")
    run_settings = RunSettings()
    run_settings.add_eval(iterations=0)
    ts = CEPATrialSettings("test_equal_per_host", run_settings, target_settings)
    fixed = PlantNetworkFixedControlTarget(trial_settings=ts)
    fixed.run()
    cp = ControllerPlotter(ts.test_name, ts.test_name, 0)
    cp.plot_eval(0)
    action_df = pd.read_csv(cp.get_action_file_path())
    action_df_split = plotter.split_by_node(action_df)
    # Environment rounds down the inspection before logging if there aren't enough hosts so can only
    # guarantee for initial inspections.
    actions_series_cull = action_df_split[0].loc[action_df_split[0]['t'] == 0]['cull'].values
    actions_series_thin = action_df_split[0].loc[action_df_split[0]['t'] == 0]['thin'].values
    # Strip out nans.
    actions_series_cull = actions_series_cull[~np.isnan(actions_series_cull)]
    actions_series_thin = actions_series_thin[~np.isnan(actions_series_thin)]
    # Cull can be one more than thin because the env will round up culls when there is sufficient budget.
    assert((np.logical_or((actions_series_cull == actions_series_thin),
                          (actions_series_cull == actions_series_thin + 1)).all()))


def test_sensitivity_based_control_target():
    target_settings = build_cull_or_thin_grid(x_nodes=2, y_nodes=2)
    target_settings.env_params.env_format = "OC"
    target_settings.sim_setup["rate_based_control"] = True
    target_settings.controller["type"] = "sensitivity_control"
    target_settings.controller["settings"] = {}

    target_settings.set_display_name_addendum("test")
    run_settings = RunSettings()
    run_settings.add_eval(iterations=0)
    ts = CEPATrialSettings("test_sensitivity_control", run_settings, target_settings)
    fixed = SensitivityBasedControlTarget(trial_settings=ts)
    fixed.run()
    cp = ControllerPlotter(ts.test_name, ts.test_name, 0)
    cp.plot_eval(0)
    action_df = pd.read_csv(cp.get_action_file_path())
    action_df_split = plotter.split_by_node(action_df)
    # Environment rounds down the inspection before logging if there aren't enough hosts so can only
    # guarantee for initial inspections.
    culls = []
    for node in range(4):
        new_cull = action_df_split[node].loc[action_df_split[node]['t'] == 0]['cull'].values
        culls += [new_cull[~np.isnan(new_cull)]]
        new_thin = action_df_split[node].loc[action_df_split[node]['t'] == 0]['thin'].values
        new_thin = new_thin[~np.isnan(new_thin)]
        assert (new_thin == 0).all()
    # At t=0, the cull values for each node should be the same. The cull for node 0 should be
    # max and the others should be zero
    assert (culls[1] == 0).all()
    assert (culls[2] == 0).all()
    assert (culls[3] == 0).all()
    assert (culls[0] == 70).all()
