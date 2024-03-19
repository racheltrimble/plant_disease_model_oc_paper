import shutil
from pathlib import Path

import pandas as pd

from notebooks.reporting_utils import data_root, clear_analysis_dir, clean_data_folder


def combine_data_from_study_in_parts(data_path):
    clear_analysis_dir()
    clean_data_folder()

    data_folder = Path(data_path)

    # Copy over only the splits data
    subdirs = [x for x in data_folder.iterdir() if x.is_dir()]
    for test_dir in subdirs:
        test_name = test_dir.parts[-1]
        if test_name.startswith("splits"):
            destination = data_root / test_name
            shutil.copytree(test_dir, destination)

    # Merge evaluation data into "standard" format.
    for n in range(10):
        splits_dir = data_root / Path(f"splits{n}")
        # Copy the first set of data as the structure template to the data root directory.
        if n == 0:
            for subdir in splits_dir.iterdir():
                if subdir.is_dir():
                    shutil.copytree(subdir, data_root / Path(subdir.name))
        else:
            # Go through each of the data directories as the source
            # Ignore any hidden directories that apple seeems to add for the stashed data...
            for source in [x for x in splits_dir.iterdir() if not x.name.startswith('.')]:
                source_eval = source / Path("0") / Path("eval")
                # Find the corresponding destination directory
                dest_eval = data_root / Path(source.name) / Path("0") / Path("eval")
                # Ignore monitor.csv (not used) and definition.yaml (already copied)
                # Merge action_data, env_data, net_data, monitor.csv and rewards.csv
                merge_monitor_csv(source_eval / Path("monitor.csv"), dest_eval / Path("monitor.csv"))
                merge_rewards_csv(source_eval / Path("rewards.csv"), dest_eval / Path("rewards.csv"))
                filepairs = [(source_eval / Path("action_data.csv"), dest_eval / Path("action_data.csv")),
                             (source_eval / Path("env_data.csv"), dest_eval / Path("env_data.csv")),
                             (source_eval / Path("net_data_res_100.csv"), dest_eval / Path("net_data_res_100.csv"))]
                merge_with_iterations(filepairs, iteration_offset=n*10)


# Rewards file is a single line with all data
def merge_rewards_csv(sourcefile, destfile):
    # get the new data:
    with open(sourcefile) as infile:
        in_lines = [x for x in infile.readlines()]

    assert len(in_lines) == 1
    new_data = in_lines[0]
    # Read in the current line from the output file and append
    with open(destfile, 'r') as outfile:
        file_lines = [''.join([x.strip(), ",", new_data]) for x in outfile.read().splitlines()]
    assert len(file_lines) == 1

    # Write back to the output file
    with open(destfile, 'w') as f:
        f.writelines(file_lines)


def merge_with_iterations(filepairs, iteration_offset):
    for sourcefile, destfile in filepairs:
        # Read both files in as dataframes
        source_df = pd.read_csv(sourcefile, index_col=0)
        dest_df = pd.read_csv(destfile, index_col=0)

        # Offset the iteration number from the source file
        source_df['test_iteration'] = source_df['test_iteration'] + iteration_offset

        # Append the dataframes
        combined_df = pd.concat([dest_df, source_df], ignore_index=True)

        # Write back to the destination file
        combined_df.to_csv(destfile)


# Merging the monitor file is straightforward (just summary per iteration so no updates required)
def merge_monitor_csv(sourcefile, destfile):
    with open(destfile, 'a') as outfile:
        with open(sourcefile) as infile:
            start = True
            for line in infile:
                if not start:
                    outfile.write(line)
                start = False
