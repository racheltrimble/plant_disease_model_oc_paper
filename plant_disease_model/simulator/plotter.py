# Module to handle graphing of outputs
from math import ceil

import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import numpy as np
import pandas as pd


# Basic averaging function for trajectories on the same time downsampling.
def get_average(df_in: pd.DataFrame, filter_stochastic_threshold=None, iterations=None):
    if iterations is not None:
        df_in = df_in.loc[df_in["test_iteration"].isin(iterations)]

    # This should all be from a single node
    assert (len(pd.unique(df_in['idx'])))

    # All the trajectories should be at the same resolution
    assert is_time_quantised(df_in)

    df = filter_stochastic_fade(df_in, filter_stochastic_threshold)

    averages = df.groupby('t').mean()
    averages.reset_index(inplace=True)
    averages.drop('test_iteration', axis=1, inplace=True)

    return averages


# Collects averages across the trajectories at a given resolution
# Only required if the data has not been down-sampled in time within the
# simulation.
def get_average_at_res(df_in, res, filter_stochastic_threshold=None, iterations=None):
    if iterations is not None:
        df_in = df_in.loc[df_in["test_iteration"] in iterations]

    unique = pd.unique(df_in['idx'])
    assert (len(unique) == 1)
    idx = unique[0]

    # Must not already have been quantised (invalid).
    assert not is_time_quantised(df_in)

    # Create fixed grid for averaging
    t_max = df_in['t'].max()

    if res is None:
        steps = 100
    else:
        steps = ceil(t_max/res)

    t_average = np.linspace(0, t_max, steps + 1)

    # Set up average arrays
    s_average = np.zeros(steps + 1)
    i_average = np.zeros(steps + 1)
    r_average = np.zeros(steps + 1)

    # For some tests the stochastic die out can dominate the average and needs a much
    # larger number of tests to equilibrate. So filtering:
    df = filter_stochastic_fade(df_in, filter_stochastic_threshold)

    s_average[0] = df.loc[df['t'] == 0]['nS'].mean()
    i_average[0] = df.loc[df['t'] == 0]['nI'].mean()
    r_average[0] = df.loc[df['t'] == 0]['nR'].mean()

    # Pass over average array entries
    for i in range(1, steps + 1):
        # Pick out the largest time value for each test repeat that is smaller
        # than the current maximum time.
        before_time = df.loc[df['t'] <= t_average[i]]
        # Find the latest entry in this time window for each run
        index = before_time.groupby(['test_iteration'])['t'].transform(max) == before_time['t']
        latest_val = before_time[index]
        # Should always get some data for this...
        assert (len(latest_val) > 0)
        s_average[i] = latest_val['nS'].mean()
        i_average[i] = latest_val['nI'].mean()
        r_average[i] = latest_val['nR'].mean()
    out = pd.DataFrame({'t': t_average,
                        'nS': s_average,
                        'nI': i_average,
                        'nR': r_average,
                        'idx': [idx]*len(t_average)})
    return out


def sum_over_network(net_df):
    # Check time quantised
    assert(is_time_quantised(net_df))
    out = net_df.groupby(['t', 'test_iteration']).sum()
    out.reset_index(inplace=True)
    # Easier to assign to arbitrary value than to remove and redo all the functions.
    out = out.assign(idx=-1)
    return out


def get_subplots_from_df(axs,
                         df,
                         res=None,
                         filter_stochastic_threshold=None,
                         one_axis=False,
                         action_df=None,
                         title="Node summary",
                         i_only=True,
                         y_max=None,
                         iterations=None,
                         reintroductions=None,
                         t_max=None):
    assert(t_max is None or one_axis)
    if is_time_quantised(df):
        averages = get_average(df,
                               filter_stochastic_threshold=filter_stochastic_threshold,
                               iterations=iterations)
        averages_raw = get_average(df,
                                   iterations=iterations)
    else:
        # Plotting without time quantisation unsupported for single axis and with action reporting
        assert(not one_axis)
        assert(action_df is None)
        averages = get_average_at_res(df,
                                      res,
                                      filter_stochastic_threshold=filter_stochastic_threshold,
                                      iterations=iterations)
        averages_raw = get_average_at_res(df, res, iterations=iterations)
    if one_axis:
        plot_node_on_one_axis(axs,
                              averages,
                              df,
                              action_df,
                              title=title,
                              i_only=i_only,
                              y_max=y_max,
                              iterations=iterations,
                              reintroductions=reintroductions,
                              t_max=t_max)
    else:
        plot_averages_and_individual(axs, averages, averages_raw, df)


def plot_df(df_split,
            title,
            filename,
            filter_stochastic_threshold=None,
            one_axis=False,
            action_df_split=None,
            display=True,
            i_only=True,
            iterations=None,
            reintroductions=None,
            t_max=None):
    if action_df_split is not None:
        assert len(action_df_split) == len(df_split)

    num_plots_per_node = 1 if one_axis else 3
    vertical_height = 3 if one_axis else 8
    fig, axs = plt.subplots(num_plots_per_node, len(df_split), figsize=(len(df_split) * 2.5, vertical_height))
    fig.suptitle(title)
    for index in df_split:
        df = df_split[index]
        if reintroductions is None:
            node_reintroductions = None
        else:
            node_reintroductions = reintroductions[index]
        if action_df_split is None:
            action_df = None
        else:
            action_df = action_df_split[index]
        if len(df_split) > 1:
            if one_axis:
                ax = axs[index]
            else:
                ax = axs[:, index]
            # If only 2 nodes, keep the maximum y value the same across both plots
            if len(df_split) == 2:
                max_list = [df_split[y]['nI'].max() for y in df_split]
                if action_df_split is not None:
                    max_list += [action_df_split[y]['inspect'].max() for y in action_df_split]
                y_max = max(max_list)
            else:
                y_max = None
            get_subplots_from_df(ax,
                                 df,
                                 filter_stochastic_threshold=filter_stochastic_threshold,
                                 one_axis=one_axis,
                                 action_df=action_df,
                                 title="Subpopulation " + str(index + 1),
                                 i_only=i_only,
                                 y_max=y_max,
                                 iterations=iterations,
                                 reintroductions=node_reintroductions,
                                 t_max=t_max)
        else:
            get_subplots_from_df(axs,
                                 df_split[0],
                                 filter_stochastic_threshold=filter_stochastic_threshold,
                                 one_axis=one_axis,
                                 action_df=action_df,
                                 title="Subpopulation " + str(index + 1),
                                 i_only=i_only,
                                 iterations=iterations,
                                 reintroductions=node_reintroductions,
                                 t_max=t_max)
    plt.tight_layout()
    fig.savefig(filename, dpi=fig.dpi)
    if display:
        plt.show()
    else:
        plt.close(fig)


def get_plot_df_legend(filename,
                       i_only=True,
                       with_actions=True,
                       with_inspect=True,
                       with_thin=False,
                       display=False):
    assert i_only
    if with_actions:
        figsize = [3.5, 1.5]
    else:
        figsize = [3.5, 0.75]
    fig, ax = plt.subplots(figsize=figsize)
    # Note - alpha values of individual lines have been bulked up for the legend so they are
    # more visible.
    i_line = mlines.Line2D([], [], marker='', color='red',
                           linewidth=0.6, alpha=0.3, label="infected hosts (individual sim)")
    i_average = mlines.Line2D([], [], marker='', color='darkred',
                              linewidth=1.9, alpha=0.9, label="infected hosts (average)")

    handles = [i_line, i_average]
    if with_actions:
        assert(not (with_inspect and with_thin))
        if with_inspect:
            inspect = mlines.Line2D([], [], marker='', color='orange',
                                    linewidth=0.6, alpha=0.3, label="inspected hosts (individual sim)")
            found = mlines.Line2D([], [], marker='', color='blue',
                                  linewidth=0.6, alpha=0.3, label="found infected hosts (individual sim)")
            handles += [inspect, found]
        if with_thin:
            cull = mlines.Line2D([], [], marker='', color='blue',
                                 linewidth=0.6, alpha=1.0, label="culled hosts")
            thin = mlines.Line2D([], [], marker='', color='purple',
                                 linewidth=0.6, alpha=1.0, label="thinned hosts")
            handles += [cull, thin]


    ax.legend(handles=handles)
    plt.axis("off")
    plt.tight_layout()

    fig.savefig(filename, dpi=fig.dpi)
    if display:
        plt.show()
    else:
        plt.close(fig)


# Plots env level data for reward and bank balance on separate axes.
def plot_env(df, title, filename, display=True, bank=False):
    fig, axs = plt.subplots(3 + bank, 1, figsize=(5, 8))
    fig.suptitle(title)
    reward_subplot(axs[0], df)
    obs_subplot(axs[1], df)
    action_subplot(axs[2], df)
    if bank:
        bank_subplot(axs[3], df)
    plt.tight_layout()
    fig.savefig(filename, dpi=fig.dpi)
    if display:
        plt.show()
    else:
        plt.close(fig)


def reward_subplot(ax, df):
    test_iteration_max = int(df['test_iteration'].max())

    for i in range(test_iteration_max + 1):
        it_data = df.loc[df['test_iteration'] == i]
        ax.plot(it_data['t'], it_data['reward'], marker='', color='grey', linewidth=0.6, alpha=0.3)

    ax.set_xlabel("Time")
    ax.set_ylabel("Reward")


def bank_subplot(ax, df):
    test_iteration_max = int(df['test_iteration'].max())

    for i in range(test_iteration_max + 1):
        it_data = df.loc[df['test_iteration'] == i]
        ax.plot(it_data['t'], it_data['bank'], marker='', color='grey', linewidth=0.6, alpha=0.3)
    ax.set_xlabel("Time")
    ax.set_ylabel("Bank balance")


def obs_subplot(ax, df):
    test_iteration_max = int(df['test_iteration'].max())

    for i in range(test_iteration_max + 1):
        it_data = df.loc[df['test_iteration'] == i]
        time = it_data['t']
        obs = it_data.filter(regex='obs')
        for obs_line in obs:
            ax.plot(time, obs[obs_line], marker='', linewidth=0.6, alpha=0.3)
    ax.set_xlabel("Time")
    ax.set_ylabel("Observations")


def action_subplot(ax, df):
    test_iteration_max = int(df['test_iteration'].max())

    for i in range(test_iteration_max + 1):
        it_data = df.loc[df['test_iteration'] == i]
        time = it_data['t']
        action = it_data.filter(regex='action')
        for a_line in action:
            ax.plot(time, action[a_line], marker='', linewidth=0.6, alpha=0.3)
    ax.set_xlabel("Time")
    ax.set_ylabel("Actions")


def split_and_compare(df1,
                      df2,
                      allowable_diff=5.0,
                      check_different=False,
                      filter_stochastic_threshold=None,
                      allowable_fadeout_diff=5.0):
    case1_split = split_by_node(df1)
    case2_split = split_by_node(df2)
    assert(len(case2_split) == len(case1_split))

    for index in case1_split:
        node1 = case1_split[index]
        node2 = case2_split[index]
        compare_results(node1, node2, allowable_diff, check_different, filter_stochastic_threshold)
        compare_fadeout(node1, node2,
                        allowable_diff=allowable_fadeout_diff,
                        stochastic_threshold=filter_stochastic_threshold)


def compare_results(df1,
                    df2,
                    allowable_diff=5.0,
                    check_different=False,
                    filter_stochastic_threshold=None):
    # Check we're comparing things from the same node.
    unique1 = pd.unique(df1['idx'])
    unique2 = pd.unique(df2['idx'])
    assert (len(unique1) == 1)
    assert (len(unique2) == 1)
    assert (unique1[0] == unique2[0])

    average_df1 = get_average(df1, filter_stochastic_threshold=filter_stochastic_threshold)
    average_df2 = get_average(df2, filter_stochastic_threshold=filter_stochastic_threshold)
    sampled1 = average_df1[['nS', 'nI', 'nR']].to_numpy(dtype='float32')
    sampled2 = average_df2[['nS', 'nI', 'nR']].to_numpy(dtype='float32')

    close = np.isclose(sampled1, sampled2, rtol=0.0, atol=allowable_diff)
    if check_different:
        check_var = not close.all()
    else:
        check_var = close.all()
    if not check_var:
        # Print which indices were and were not close
        print(close)
        # Calculate the difference for the cases where differences were too large
        print(sampled1[np.logical_not(close)] - sampled2[np.logical_not(close)])
    assert check_var


# Compare proportion of die outs.
# Defining a die out as when less than 5% of the population is R at the end of sim.
def compare_fadeout(df1, df2, stochastic_threshold, allowable_diff):
    # Check we're comparing things from the same node.
    unique1 = pd.unique(df1['idx'])
    unique2 = pd.unique(df2['idx'])
    assert (len(unique1) == 1)
    assert (len(unique2) == 1)
    assert (unique1[0] == unique2[0])

    final1 = df1.loc[df1['t'] == df1['t'].max()]
    final2 = df2.loc[df2['t'] == df2['t'].max()]

    die_outs1 = final1['nR'] < stochastic_threshold
    die_outs2 = final2['nR'] < stochastic_threshold
    print("First df die outs: ", sum(die_outs1))
    print("Second df die outs", sum(die_outs2))

    result = sum(die_outs1) - sum(die_outs2)
    print("Difference: ", result)
    assert (result * np.sign(result) < allowable_diff)


def is_time_quantised(net_df):
    # All the trajectories should be at the same resolution
    test_groups = net_df.groupby('test_iteration')
    if (test_groups.size().values == test_groups.size().values[0]).all():
        out = True
        groups = pd.unique(net_df['test_iteration'])
        ref_times = test_groups.get_group(groups[0])['t'].values
        for group in groups:
            # If the data comes from a single run then all the times will be the same
            # but when multiple test runs are combined, there can be some offset.
            this_test_times = test_groups.get_group(group)['t'].values
            out &= np.isclose(ref_times, this_test_times, atol=0.05).all()
    else:
        out = False

    return out


def split_by_node(net_df):
    out = {}
    for idx in pd.unique(net_df['idx']):
        out[int(idx)] = net_df.loc[net_df['idx'] == idx]
    return out


# Expects to receive a datafile with data for a single node but multiple test iterations.
# Also takes in the corresponding action datafile
# Filters into two dfs - one contains test iterations matching the initial conditions defined by the
# supplied function. The other contains not matching test iterations.
def split_test_iterations_by_criterion(node_df, action_df, func):
    assert len(pd.unique(node_df['idx'])) == 1
    test_groups = node_df.groupby('test_iteration')
    matching = test_groups.filter(func)
    # Pull out the corresponding action data
    matching_iterations = pd.unique(matching['test_iteration'])
    not_matching = node_df.loc[~node_df['test_iteration'].isin(matching_iterations)]
    assert(len(matching) + len(not_matching) == len(node_df))
    action_matching = action_df.loc[action_df['test_iteration'].isin(matching_iterations)]
    action_not_matching = action_df.loc[~action_df['test_iteration'].isin(matching_iterations)]
    return matching, not_matching, action_matching, action_not_matching


def lists_from_df(df):
    t_average = df['t'].tolist()
    s_average = df['nS'].tolist()
    i_average = df['nI'].tolist()
    r_average = df['nR'].tolist()
    return t_average, s_average, i_average, r_average


def plot_averages_and_individual(axs, averages, averages_raw, df):
    t_average, s_average, i_average, r_average = lists_from_df(averages)
    t_average_raw, s_average_raw, i_average_raw, r_average_raw = lists_from_df(averages_raw)
    test_iteration_max = int(df['test_iteration'].max())

    for i in range(test_iteration_max + 1):
        it_data = df.loc[df['test_iteration'] == i]
        axs[0].plot(it_data['t'], it_data['nS'], marker='', color='grey', linewidth=0.6, alpha=0.3)
        axs[1].plot(it_data['t'], it_data['nI'], marker='', color='grey', linewidth=0.6, alpha=0.3)
        axs[2].plot(it_data['t'], it_data['nR'], marker='', color='grey', linewidth=0.6, alpha=0.3)

    # Plot average trajectories
    axs[0].plot(t_average, s_average, marker='', color='green', linewidth=1.9, alpha=0.9)
    axs[0].plot(t_average_raw, s_average_raw, marker='', color='blue', linewidth=1.9, alpha=0.9)
    axs[0].set_title('Number susceptible')
    axs[0].set_xlabel("Time")
    axs[0].set_ylabel("Hosts")
    axs[1].plot(t_average, i_average, marker='', color='darkred', linewidth=1.9, alpha=0.9)
    axs[1].plot(t_average_raw, i_average_raw, marker='', color='orange', linewidth=1.9, alpha=0.9)
    axs[1].set_title('Number infected')
    axs[1].set_xlabel("Time")
    axs[1].set_ylabel("Hosts")
    axs[2].plot(t_average, r_average, marker='', color='black', linewidth=1.9, alpha=0.9)
    axs[2].plot(t_average_raw, r_average_raw, marker='', color='purple', linewidth=1.9, alpha=0.9)
    axs[2].set_title('Number removed')
    axs[2].set_xlabel("Time")
    axs[2].set_ylabel("Hosts")


# Epidemiologist preference is to see just I rather than stacked S,I,R so i_only==True aligns to this.
def plot_node_on_one_axis(ax,
                          averages,
                          df,
                          action_df,
                          title="Node Summary",
                          i_only=True,
                          y_max=None,
                          iterations=None,
                          reintroductions=None,
                          t_max=None):
    if t_max is not None:
        averages = averages.loc[averages['t'] <= t_max]
        df = df.loc[df['t'] <= t_max]
        action_df = action_df.loc[action_df['t'] <= t_max]
    t_average = averages['t'].to_numpy(dtype='float32')
    s_average = averages['nS'].to_numpy(dtype='float32')
    i_average = averages['nI'].to_numpy(dtype='float32')
    r_average = averages['nR'].to_numpy(dtype='float32')

    iterations = get_iteration_list(iterations, df)

    if len(iterations) == 1:
        line_thickness = 1.5
    else:
        line_thickness = 0.6
    print("Plotting ", len(iterations), " iterations for plot title ", title)
    for i in iterations:
        it_data = df.loc[df['test_iteration'] == i]
        if i_only:
            ax.plot(it_data['t'], it_data['nI'], marker='', color='red', linewidth=0.6, alpha=0.1)
        else:
            ax.plot(it_data['t'], it_data['nS'], marker='', color='green', linewidth=0.6, alpha=0.1)
            ax.plot(it_data['t'], it_data['nS'] + it_data['nI'], marker='', color='red', linewidth=0.6, alpha=0.1)
            total = it_data['nS'] + it_data['nI'] + it_data['nR']
            ax.plot(it_data['t'], total, marker='', color='black', linewidth=0.6, alpha=0.1)
        # Plot reintroductions
        if reintroductions is not None:
            ax.plot(reintroductions,
                    np.zeros_like(reintroductions),
                    marker=".",
                    color='green',
                    linestyle='')

        if action_df is not None:
            action_data = action_df.loc[action_df['test_iteration'] == i]
            inspect_data = action_data.loc[np.logical_not(np.isnan(action_data['inspect']))]
            # Allow backwards compatibility
            if 'cull' in list(action_data):
                cull_data = action_data.loc[np.logical_not(np.isnan(action_data['cull']))]
                thin_data = action_data.loc[np.logical_not(np.isnan(action_data['thin']))]
                cull_key = "cull"
            else:
                cull_data = action_data.loc[np.logical_not(np.isnan(action_data['found']))]
                #
                thin_data = pd.DataFrame(columns=["t", "thin"])
                cull_key = "found"
            inspect_marker = ''
            cull_marker = ''
            thin_marker = ''
            alpha = 0.5
            if len(cull_data) == 1:
                cull_marker = 'o'
            if len(thin_data) == 1:
                thin_marker = 'o'
            if len(inspect_data) == 1:
                inspect_marker = 'o'
            ax.step(inspect_data['t'], inspect_data['inspect'], marker=inspect_marker, color='orange', linewidth=line_thickness, alpha=alpha, where='post')
            ax.step(cull_data['t'], cull_data[cull_key], marker=cull_marker, color='blue', linewidth=line_thickness, alpha=alpha, where='post')
            ax.step(thin_data['t'], thin_data['thin'], marker=thin_marker, color='purple', linewidth=line_thickness, alpha=alpha, where='post')

    # Plot average trajectories
    if len(i_average) == 1:
        marker = 'o'
    else:
        marker = ''

    if i_only:
        ax.plot(t_average, i_average, marker=marker, color='darkred', linewidth=1.9, alpha=0.9, label="S+I")
    else:
        ax.plot(t_average, s_average, marker=marker, color='green', linewidth=1.9, alpha=0.9, label="S")
        ax.plot(t_average, s_average + i_average, marker=marker, color='darkred', linewidth=1.9, alpha=0.9, label="S+I")
        ax.plot(t_average, s_average + i_average + r_average,
                marker='',
                color='black',
                linewidth=1.9,
                alpha=0.9,
                label="N")

    # None leaves the limit unchanged so can pass through without if check.
    ax.set_ylim(top=y_max)
    ax.set_title(title)
    ax.set_xlabel("Time")
    ax.set_ylabel("Hosts")


def get_iteration_list(iterations, df):
    test_iteration_max = int(df['test_iteration'].max())
    if iterations is None:
        iterations = list(range(test_iteration_max + 1))
    else:
        assert (np.array(iterations) < test_iteration_max).all()
    return iterations


# Reintroduction test needs a raw df otherwise it can miss cases where reintroduction is quickly
# reeradicated.
def get_reintroductions(df, iterations=None):
    assert not is_time_quantised(df)
    iterations = get_iteration_list(iterations, df)
    max_node = int(df['idx'].max())
    reintroduction_times = [[] for _ in range(max_node+1)]
    for i in iterations:
        it_data = df.loc[df['test_iteration'] == i]
        # Using https://stackoverflow.com/questions/48710783/pandas-find-and-index-rows-that-match-row-sequence-pattern
        # The pattern of a reintroduction is an infected of zero followed by a 1.
        # Note - this is not precise as it is using time quantised data and so will miss some
        # introductions that are quickly reeradicated or cases where the disease spreads in the timestep.
        reintroduction_pattern = np.array([0, 1])
        reintroduction = it_data['nI'].rolling(window=2, min_periods=2) \
            .apply(lambda x: (x == reintroduction_pattern).all()) \
            .mask(lambda x: x == 0) \
            .bfill(limit=len(reintroduction_pattern) - 1) \
            .fillna(0) \
            .astype(bool)
        for idx, node_list in enumerate(reintroduction_times):
            node_list += (it_data.loc[reintroduction & (it_data['idx'] == idx)]['t'].tolist())
    return reintroduction_times


def filter_stochastic_fade(df_in, threshold):
    # For some tests the stochastic die out can dominate the average and needs a much
    # larger number of tests to equilibrate. So filtering:
    if threshold is None:
        df = df_in
    else:
        df = df_in.groupby(df_in['test_iteration']+1000*df_in['idx']).filter(
            lambda x: ((x['nR'] > threshold).sum() > 0)
        )
        df.reset_index(drop=True, inplace=True)
    return df


# Expects a df containing the data for the whole network.
# Returns the time that disease was eradicated from the entire network for each individual test run.
def get_eradication_times(df):
    assert(is_time_quantised(df))
    # Split df by test iteration
    test_iteration_max = int(df['test_iteration'].max())

    # For each test iteration, find the first time that I=0 when summed across all nodes
    time_list = []
    # Default to NaN if disease is not eradicated
    # Setting to be the same across all iterations.
    for i in range(test_iteration_max + 1):
        it_data = df.loc[df['test_iteration'] == i]
        times_for_this_test = it_data['t'].unique()
        # Sort times into ascending order
        times_for_this_test.sort()
        new_time = np.NAN
        for t in times_for_this_test:
            if it_data.loc[it_data['t'] == t]['nI'].sum() == 0:
                new_time = t
                break
        if not np.isnan(new_time):
            time_list.append(new_time)

    # Return a list of these times and a count of iterations that were never eradicated.
    return time_list


# Returns a list of times and a list of lists (n_infected_nodes_list).
# Each list in n_infected_nodes_list represents the number of nodes at that time point containing
# N infected hosts.
def get_sims_with_n_infected_nodes(df):
    assert(is_time_quantised(df))
    # For each time point, find the number of infected nodes (nodes with nI > 0)
    # Time quantised so should be the same across nodes and iterations.
    times = df['t'].unique()
    num_nodes = None
    n_infected_nodes_list = []
    for t in times:
        data_for_this_time = df.loc[df['t'] == t]
        # For each iteration, check if one or more nodes are infected.
        iterations = data_for_this_time['test_iteration'].unique()
        nodes = data_for_this_time['idx'].unique()
        if num_nodes is None:
            num_nodes = len(nodes)
        # Count for each possible number of nodes that could be infected (0 to num_nodes) -> num_nodes + 1 entries
        per_timestep = np.zeros(num_nodes + 1)
        for i in iterations:
            data_for_this_iteration = data_for_this_time.loc[data_for_this_time['test_iteration'] == i]
            # Expecting the same number of nodes at each timestep and iteration.
            nodes = data_for_this_iteration['idx'].unique()
            assert(num_nodes == len(nodes))
            # Check which nodes are infected
            by_nodes = data_for_this_iteration.groupby('idx')['nI'].sum() > 0
            assert(len(by_nodes) == num_nodes)
            # Count infected nodes and add to list
            per_timestep[by_nodes.sum()] += 1
        n_infected_nodes_list.append(per_timestep)
    n_infected_nodes_array = np.array(n_infected_nodes_list).transpose()

    return times, n_infected_nodes_array
