from control_ablations.ablation_infra import BasePlotBlock
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from seaborn import swarmplot, stripplot, boxplot, violinplot, color_palette
import textwrap


# This is not very Pythonic but want to wrap these mainly static functions with the associated classes.
# They are used with different types of data within the different class (and so take the data as
# arguments) but they rely on particular data formats which are not sensible interfaces for anything
# else.
class ComparisonPlotBlock(BasePlotBlock):

    def __init__(self,
                 analyser_settings):
        self.colour_scheme = analyser_settings.get("colour_scheme", None)
        super().__init__()

    def plot(self):
        raise NotImplementedError

    @staticmethod
    def get_trigger():
        raise NotImplementedError

    def swarmplot_from_dict(self,
                            ax,
                            x,
                            y,
                            source_dict,
                            title,
                            plot_type="swarm",
                            axis_rename=None,
                            add_n_to_title=True,
                            drop_nas=True,
                            data_axis_limits=None):
        if axis_rename is None:
            axis_rename = {}
        medium_size = 10
        # Dictionary of lists of rollout rewards (index is test name) to dataframe
        # Lists can be uneven so going via Series.
        df = pd.DataFrame(dict([(k, pd.Series(v, dtype=np.float64)) for k, v in source_dict.items()]))
        # Dataframe to long form. Ignoring any nans from the uneven lists.
        df = df.melt(value_vars=df.columns.values.tolist(), var_name=y, value_name=x)
        if drop_nas:
            df.dropna(inplace=True)
        df.rename(columns=axis_rename)
        if plot_type == "swarm":
            sp = swarmplot(df, ax=ax, x=x, y=y, hue=y, legend=False, zorder=.5)
            sp.set(ylabel=None)
            plt.rc('ytick', labelsize=medium_size)  # fontsize of the tick labels
        elif plot_type == "strip":
            sp = stripplot(df, ax=ax, x=x, y=y, hue=y, legend=False)
            sp.set(ylabel=None)
        elif plot_type == "boxplot":
            bp = boxplot(df, ax=ax, x=x, y=y)
            bp.set(ylabel=None)
            # Takes the length of the first item in the dictionary.
            num_samples = len(list(source_dict.values())[0])
            title += f" (N={num_samples})"
        elif plot_type == "violin":
            order = df[y].unique()
            palette = self.get_palette(order)

            def max_width_format(label, _):
                # Don't mess with existing formatting
                if "\n" in label:
                    return label
                else:
                    return textwrap.fill(label, 15)

            max_width_formatter = plt.FuncFormatter(max_width_format)
            vp = violinplot(df,
                            ax=ax,
                            x=x,
                            y=y,
                            hue=y,
                            order=order,
                            hue_order=order,
                            density_norm="width",
                            bw_adjust=0.5,
                            linewidth=0,
                            formatter=max_width_formatter,
                            inner_kws=dict(box_width=4, whis_width=1, color="k"),
                            legend=False,
                            palette=palette)
            vp.set(ylabel=None)
            ax.yaxis.set_tick_params(which='both', labelsize=10.0)
            if data_axis_limits is not None:
                vp.set_xlim(data_axis_limits[0], data_axis_limits[1])
            # Takes the length of the first item in the dictionary.
            if add_n_to_title:
                num_samples = len(list(source_dict.values())[0])
                title += f" (N={num_samples})"
        else:
            print("Unrecognised plot type passed to swarmplot generation")
            assert 0

        ax.set_title(title)
        plt.tight_layout()

    def get_palette(self, y_labels):
        default_palette = color_palette()
        if self.colour_scheme is None:
            palette = default_palette
        else:
            palette = []
            for label in y_labels:
                # Get the palette index for this label.
                # Split out any sample size labelling and strip whitespace.
                index = self.colour_scheme[label.split('\n')[0].strip()]
                palette += [default_palette[index]]
        return palette

    @staticmethod
    def histogram_from_dict(ax, x, source_dict, title):
        # Calculate bins for histograms (see ref for other hist in file).
        all_data = [item for sublist in source_dict.values() for item in sublist]
        bins = np.histogram(all_data, bins=20)[1]  # get the bin edges
        for key in source_dict:
            ax.hist(source_dict[key], label=key, alpha=0.8, bins=bins)
        ax.set_xlabel(x)
        ax.set_title(title)
        ax.legend()

    @staticmethod
    def bootstrap_diff(null, test_data):
        bootstrap_sample = len(test_data)
        rng = np.random.default_rng()
        boots = []
        for x in range(1000):
            n = rng.choice(null, bootstrap_sample)
            h = rng.choice(test_data, bootstrap_sample)
            boots.append(np.mean(h - n))

        boots = np.array(boots)
        mean = np.mean(boots)
        range_pair = np.percentile(boots, 2.5), np.percentile(boots, 97.5)

        return mean, range_pair

    @staticmethod
    def filter_outliers(data, print_label, allowed_range=None):
        new_data = []
        if allowed_range is None:
            for index, point in enumerate(data):
                excluded = np.delete(np.array(data), index)
                mean = np.mean(excluded)
                sd = np.std(excluded)
                valid = point > (mean - 6 * sd)
                valid = valid and point < (mean + 6 * sd)
                if valid:
                    new_data.append(point)
                else:
                    print(print_label, index)
        else:
            min_allowed = allowed_range[0]
            max_allowed = allowed_range[1]
            for point in data:
                if min_allowed < point < max_allowed:
                    new_data.append(point)
        return new_data

    @staticmethod
    def write_latex_table(table_data, filename):
        bootstrap_diff_df = pd.DataFrame(table_data,
                                         columns=["Learning proportion",
                                                  "Variant",
                                                  "Mean difference",
                                                  "Confidence Interval"])

        with open(filename, 'w') as f:
            f.write(bootstrap_diff_df.style.hide(axis="index")
                    .format(escape="latex")
                    .to_latex(column_format="c|c|c|c"))

    def pairwise_comparison(self, baseline_test_name, comparison_list, data):
        new_full = []
        new_table = []
        new_raw = []
        for test_name1, display_name1, test_name2, display_name2 in comparison_list:
            mean, percentiles = self.bootstrap_diff(data[display_name1],
                                                    data[display_name2])
            percentiles_string = "(" + str(round(percentiles[0], 1)) + ", " \
                                 + str(round(percentiles[1], 1)) + ")"
            new_row = {"Variant": display_name2,
                       "Mean difference": str(round(float(mean), 1)),
                       "Confidence Interval": percentiles_string}
            new_raw_row = {"Variant": display_name2,
                           "Mean difference": mean,
                           "Confidence Interval": percentiles}
            new_full.append(new_row)
            if baseline_test_name == test_name1:
                new_table.append(new_row)
                new_raw.append(new_raw_row)
        return new_full, new_table, new_raw
