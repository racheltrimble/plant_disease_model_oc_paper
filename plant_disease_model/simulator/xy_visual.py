import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from pathlib import Path
import yaml


# Takes in a datafile and associated network and creates a plot with
# the disease status at a particular time.
def draw_xy_visual_set_across_time(df,
                                   net_dict_list,
                                   subplot_size_factor=0.35,
                                   subplot_aspect_ratio=1.8,
                                   num_plots=3,
                                   time_display_resolution=2,
                                   target_times=None,
                                   subplot_type="bar",
                                   filename=None):
    # Specify either a number of plots spaced evenly in time or specific times.
    # Times can only be specified approximately as resolution is limited by logging frequency
    assert (num_plots is None) or (target_times is None)
    assert not ((num_plots is None) and (target_times is None))

    times = np.sort(df['t'].unique())

    # If target times are specified, find the closest time in the data to each target time.
    if num_plots is None:
        selected_t = []
        for time in target_times:
            selected_t.append(times[np.argmin(np.abs(times - time))])
        num_plots = len(selected_t)
    else:
        indices = np.linspace(0, len(times)-1, num_plots).round(0).astype(np.int16)
        selected_t = times[indices]

    assert num_plots > 0

    fig, ax_set = plt.subplots(1, num_plots, figsize=[10, 4])
    if num_plots == 1:
        ax_set = [ax_set]
    for index, ax in enumerate(ax_set):
        time = selected_t[index]
        handles, labels = draw_xy_sub_plots(ax,
                                            df,
                                            net_dict_list,
                                            time,
                                            subplot_size_factor=subplot_size_factor,
                                            subplot_aspect_ratio=subplot_aspect_ratio,
                                            subplot_type=subplot_type)
        ax.set_title("State at time " + "{:.{p}f}".format(time, p=time_display_resolution))

    fig.supxlabel("x position")
    fig.supylabel("y position")
    if subplot_type == "pie":
        fig.legend(handles[1],
                   labels[1],
                   loc='lower right',
                   labelspacing=7.5,
                   handletextpad=5,
                   borderpad=3.5,
                   bbox_to_anchor=(1.25, 0.0))

        handles = handles[0]
        labels = labels[0]

    fig.legend(handles,
               labels,
               loc='lower right',
               handletextpad=1.5,
               borderpad=0.75,
               bbox_to_anchor=(1.25, 0.8))

    fig.suptitle("Node states at different time points")
    plt.tight_layout()
    if filename is None:
        plt.show()
    else:
        fig.savefig(filename, bbox_inches='tight')


# Credit: https://stackoverflow.com/questions/56337732/how-to-plot-scatter-pie-chart-using-matplotlib
def draw_pie(dist,
             xpos,
             ypos,
             size,
             colours,
             ax):

    # for incremental pie slices
    cumsum = np.cumsum(dist)
    cumsum = cumsum/ cumsum[-1]
    pie = [0] + cumsum.tolist()

    for r1, r2, colour in zip(pie[:-1], pie[1:], colours):
        if r1 != r2:
            angles = np.linspace(2 * np.pi * r1, 2 * np.pi * r2)
            x = [0] + np.cos(angles).tolist()
            y = [0] + np.sin(angles).tolist()

            xy = np.column_stack([x, y])

            ax.scatter([xpos], [ypos], marker=xy, s=size, c=colour)


# Take the SIR status of a particular node and draw a bar onto the supplied axis showing the split between S,
# I and R. Based on https://matplotlib.org/stable/gallery/lines_bars_and_markers/horizontal_barchart_distribution
# .html#sphx-glr-gallery-lines-bars-and-markers-horizontal-barchart-distribution-py
def draw_bar_for_one_node(ax, s, i, r, colours, legend=False):
    category_names = ["S", "I", "R"]
    data = np.array([s, i, r])
    data_cum = data.cumsum()

    ax.yaxis.set_visible(False)
    ax.set_xlim(0, np.sum(data))

    for i, (colname, color) in enumerate(zip(category_names, colours)):
        width = data[i]
        start = data_cum[i] - width
    if legend:
        ax.legend(ncol=len(category_names), bbox_to_anchor=(0, 1),
                  loc='lower left', fontsize='small')


def find_min_node_distance(net_dict_list):
    answer = 1_000_000
    nonsense = True
    for idx1, node1 in enumerate(net_dict_list):
        for idx2, node2 in enumerate(net_dict_list):
            if idx2 > idx1:
                r = get_node_distance(node1, node2)
                if r < answer:
                    answer = r
                    nonsense = False
    assert not nonsense
    return answer


def get_node_distance(node1, node2):
    x = node1['x'] - node2['x']
    y = node1['y'] - node2['y']
    r = np.sqrt(x**2 + y**2)
    return r


def get_x_y_pop_ranges(net_dict_list):
    net_x_max = 0
    net_y_max = 0
    net_pop_max = 0
    net_x_min = 1_000_000
    net_y_min = 1_000_000
    for node_info in net_dict_list:
        if node_info['x'] > net_x_max:
            net_x_max = node_info['x']
        if node_info['y'] > net_y_max:
            net_y_max = node_info['y']
        if node_info['x'] < net_x_min:
            net_x_min = node_info['x']
        if node_info['y'] < net_y_min:
            net_y_min = node_info['y']
        if node_info['n0'] > net_pop_max:
            net_pop_max = node_info['n0']
    return net_x_min, net_x_max, net_y_min, net_y_max, net_pop_max


# Draws the system state with bar charts for each node at the given time.
def draw_xy_sub_plots(ax,
                      df,
                      net_dict_list,
                      time,
                      subplot_size_factor=0.35,
                      subplot_aspect_ratio=1.8,
                      subplot_type="bar"):
    net_x_min, net_x_max, net_y_min, net_y_max, net_pop_max = get_x_y_pop_ranges(net_dict_list)

    min_node_distance = find_min_node_distance(net_dict_list)
    biggest_plot = np.sqrt(net_pop_max)
    subplot_scale = min_node_distance / biggest_plot * subplot_size_factor
    colours = ['green', 'red', 'black']

    if subplot_type == "bar":
        x_border = max(min_node_distance, 0.1)
        y_border = max(min_node_distance, 0.1)
    elif subplot_type == "pie":
        x_border = 0.5
        y_border = 0.5
    else:
        assert 0

    plot_x_min = net_x_min - x_border
    plot_x_max = net_x_max + x_border
    plot_y_min = net_y_min - y_border
    plot_y_max = net_y_max + y_border

    ax.set_xlim((plot_x_min, plot_x_max))
    ax.set_ylim((plot_y_min, plot_y_max))

    # Check every node has data for the specified time
    per_node_i = []
    for index, node_info in enumerate(net_dict_list):
        data_for_this_node = df.loc[(df['idx'] == node_info['idx']) & (df['t'] == time)]
        # Should be averaging over a number of runs here.
        assert (len(data_for_this_node) > 1)
        s = data_for_this_node['nS'].mean()
        i = data_for_this_node['nI'].mean()
        r = data_for_this_node['nR'].mean()
        node_size = s + i + r
        label_offset = np.sqrt(node_size) / 40
        size = subplot_scale * node_size
        per_node_i += [i]
        if subplot_type == "bar":
            x = (node_info['x'] + x_border) / (net_x_max - net_x_min + 2 * x_border)
            y = (node_info['y'] + y_border) / (net_y_max - net_y_min + 2 * y_border)
            inset_height = size
            inset_width = size * subplot_aspect_ratio
            x0 = x - inset_width
            y0 = y - inset_height
            ax_inset = ax.inset_axes([x0, y0, inset_width, inset_height])
            ax_inset.set_title(f"Node {node_info['idx']}")
            draw_bar_for_one_node(ax_inset, s, i, r, colours)
            handles, labels = ax_inset.get_legend_handles_labels()
        elif subplot_type == "pie":
            x = node_info['x']
            y = node_info['y']
            draw_pie([s, i, r], x, y, size, colours, ax)
            handles0 = [get_pie_marker(50, subplot_scale, colours[0], angle=np.pi/6),
                        get_pie_marker(50, subplot_scale, colours[1], angle=np.pi/6),
                        get_pie_marker(50, subplot_scale, colours[2], angle=np.pi/6)]
            labels0 = ['S', 'I', 'R']
            labels1 = ['50', '100', '200']
            handles1 = [get_pie_marker(50, subplot_scale, colours[0]),
                        get_pie_marker(100, subplot_scale, colours[0]),
                        get_pie_marker(200, subplot_scale, colours[0])]
            handles = [handles0, handles1]
            labels = [labels0, labels1]

            # Numerical label for node
            ax.text(x - label_offset, y + label_offset, str(index + 1))
        else:
            print(f"Unexpected subplot type: {subplot_type}")
            assert 0

    add_connectivity_lines(ax, net_dict_list, per_node_i)

    plt.rc('grid', linestyle=(0, (25, 25)), color='oldlace')
    ax.grid(True)
    ax.set_xticklabels([])
    ax.set_yticklabels([])

    return handles, labels


def get_pie_marker(size, subplot_scale, colour, angle=2*np.pi):
    angles = np.linspace(2 * np.pi * 0, angle)
    x = [0] + np.cos(angles).tolist()
    y = [0] + np.sin(angles).tolist()

    xy = np.column_stack([x, y])
    marker = plt.scatter([0], [0], marker=xy,
                         s=size * subplot_scale,
                         c=colour)
    # Don't actually want to show this on plot.
    marker.remove()
    return marker


def add_connectivity_lines(ax, net_dict_list, per_node_i):
    for start_index, start_node_info in enumerate(net_dict_list):
        for end_index, end_node_info in enumerate(net_dict_list):
            if end_index != start_index:
                x = [start_node_info['x'], end_node_info['x']]
                y = [start_node_info['y'], end_node_info['y']]
                strength = 1/(get_node_distance(start_node_info, end_node_info))**3
                total_i = per_node_i[start_index] + per_node_i[end_index]
                line_scaling = 16
                ax.plot(x, y, color='maroon', linewidth=strength*total_i/line_scaling, alpha=0.4)
