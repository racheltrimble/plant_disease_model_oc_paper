import math
import os
import shutil
from pathlib import Path

from PIL import Image, ImageFont, ImageDraw

root = Path(os.path.dirname(os.path.abspath('')))
analysis_root = root / Path("analysis")
analysis_only = {"run_training": False,
                 "generate_examples": False,
                 "run_plotting": False,
                 "generate_tarball": False}
examples_only = {"run_training": False,
                 "run_analysis": False,
                 "generate_examples": True,
                 "run_plotting": False,
                 "generate_tarball": False}

data_root = root / Path("data")


def copy_data_from(folder):
    clean_data_folder()

    subdirs = [x for x in folder.iterdir() if x.is_dir()]
    for test_name in subdirs:
        destination = data_root / test_name.parts[-1]
        shutil.copytree(test_name, destination)


def move_data_to_stash(stash_path):
    data_folders = [f for f in data_root.iterdir() if f.is_dir()]
    for data_folder in data_folders:
        subdir = data_folder.parts[-1]
        shutil.move(data_folder, stash_path / subdir)


def clean_data_folder():
    subdirs = [x for x in data_root.iterdir() if x.is_dir()]
    for test_name in subdirs:
        shutil.rmtree(test_name)


def clear_analysis_dir():
    subdirs = [x for x in analysis_root.iterdir() if x.is_dir()]
    for test_name in subdirs:
        shutil.rmtree(test_name)


def move_to_report_dump(file_list, dump_sub_folder, report):
    dump_path = report
    if not dump_path.exists():
        dump_path.mkdir()
    for f, destination_name in file_list:
        destination = dump_path / Path(dump_sub_folder + "_" + destination_name)
        shutil.move(str(f), str(destination))


def setup_existing_data(study_class, data_path):
    clear_analysis_dir()
    data_folder = Path(data_path)
    copy_data_from(data_folder)
    return study_class()


def compare_performance_for_existing_study(study_class,
                                           data_path,
                                           oc_style=False,
                                           colour_scheme=None):
    s = setup_existing_data(study_class, data_path)

    analyser_settings = {"plot_training_data": False,
                         "compare_key_learning_curve_points": False,
                         "compare_trained_performance": True,
                         "colour_scheme": colour_scheme}

    if oc_style:
        analyser_settings["performance_plot_title"] = "Violin plot of rewards for different controllers"
        analyser_settings["plot_type"] = "violin"
        analyser_settings["negate_reward_axis"] = True

    s.run(**analysis_only,
          analyser_settings=analyser_settings)


def eradication_times(data_path, study_class, tag, report):

    s = setup_existing_data(study_class, data_path)

    analyser_settings = {"plot_training_data": False,
                         "compare_key_learning_curve_points": False,
                         "compare_trained_performance": False,
                         "compare_eradication_times": True,
                         "eradication_plot_title": "Violin plot of eradication times different controllers",
                         "plot_type": "violin"}
    s.run(**analysis_only,
          analyser_settings=analyser_settings)

    file_list = [
                 (analysis_root / Path("cull_or_thin_2_by_2_nodes") / Path("final_model_eradication.png"),
                  "eradication_times.png"),
                ]
    move_to_report_dump(file_list, tag, report)


def sims_with_n_infected_nodes(study_class, data_path, tag, report):

    s = setup_existing_data(study_class, data_path)

    analyser_settings = {"plot_training_data": False,
                         "compare_key_learning_curve_points": False,
                         "compare_trained_performance": False,
                         "compare_eradication_times": False,
                         "sims_with_n_infected_nodes": True
                         }
    s.run(**analysis_only,
          analyser_settings=analyser_settings)
    root_for_files = analysis_root / Path("cull_or_thin_2_by_2_nodes")
    file_list_inputs = root_for_files.glob("sims_with_n_*.png")
    file_list = [(f, f.name) for f in file_list_inputs]
    move_to_report_dump(file_list, tag, report)


def shunt_to_stash(stash_path):
    # Move data and analysis folders to the main stash.
    folder = stash_path / Path("data")
    data_folders = [f for f in data_root.iterdir() if f.is_dir()]
    for data_folder in data_folders:
        subdir = data_folder.parts[-1]
        shutil.move(data_folder, folder / subdir)

    folder = stash_path / Path("analysis")
    analysis_folders = [f for f in analysis_root.iterdir() if f.is_dir()]
    for analysis_folder in analysis_folders:
        subdir = analysis_folder.parts[-1]
        shutil.move(analysis_folder, folder / subdir)


def tile_subplots_vertically_to_tiff(image_file_list,
                                     report_path,
                                     figure_name,
                                     new_subfigure,
                                     scales,
                                     subfigure_label_offset=(0, 0),
                                     as_png=False,
                                     row_separation_text=None):
    if row_separation_text is None:
        vspace = 30
    else:
        vspace = 100
    hspace = 30
    # Find max width and sum of heights
    max_horizontal = 0
    total_vertical = 0
    vertical_positions = []
    horizontal_points_2d = []
    scaled_width_2d = []
    scaled_height_2d = []
    overall_widths = []
    for idx, horizontal_stripe in enumerate(image_file_list):
        if new_subfigure[idx]:
            total_vertical += vspace
        vertical_positions += [total_vertical]
        overall_width, scaled_widths, scaled_heights, horizontal_points = get_horizontal_stripe_dims(horizontal_stripe,
                                                                                                     scales[idx],
                                                                                                     hspace)
        scaled_width_2d += [scaled_widths]
        scaled_height_2d += [scaled_heights]
        overall_widths += [overall_width]
        max_horizontal = max(max_horizontal, overall_width)
        total_vertical += max(scaled_heights)
        horizontal_points_2d += [horizontal_points]
    total_vertical += vspace
    im = Image.new('RGB', (max_horizontal, total_vertical), color="white")
    letter_index = ord('a')

    for idx, horizontal_stripe in enumerate(image_file_list):
        if row_separation_text is None:
            header_text = None
        else:
            header_text = row_separation_text[idx]

        letter_index = draw_horizontal_stripe(im,
                                              horizontal_stripe,
                                              scaled_width_2d[idx],
                                              scaled_height_2d[idx],
                                              vertical_positions[idx],
                                              new_subfigure[idx],
                                              horizontal_points_2d[idx],
                                              (max_horizontal - overall_widths[idx])/2,
                                              letter_index,
                                              subfigure_label_offset,
                                              header_text)
    if as_png:
        suffix = ".png"
    else:
        suffix = ".tif"
    file_name = report_path / Path(figure_name + suffix)
    im.save(file_name)

    # Reload to check integrity
    im_check = Image.open(file_name)
    im_check.verify()


def draw_horizontal_stripe(im,
                           horizontal_stripe,
                           scaled_widths,
                           scaled_heights,
                           v_position,
                           new_subfigures,
                           horizontal_points,
                           h_border,
                           letter_index,
                           subfigure_label_offset,
                           header_text):
    font_path = "/Library/Fonts/Arial Unicode.ttf"
    font = ImageFont.truetype(font_path, 32)
    draw_handle = ImageDraw.Draw(im)

    for idx, image_file in enumerate(horizontal_stripe):
        new_im = Image.open(image_file)
        new_im = new_im.resize((scaled_widths[idx], scaled_heights[idx]))
        h_position = math.ceil(h_border+horizontal_points[idx])
        im.paste(new_im, (h_position, v_position))
        if new_subfigures[idx]:
            letter = chr(letter_index)
            label_text = letter + ")"
            letter_index += 1
            draw_handle.text((h_position + subfigure_label_offset[0], v_position + subfigure_label_offset[1]),
                             label_text,
                             "black",
                             font=font)
    if header_text is not None:
        font48 = ImageFont.truetype(font_path, 48)
        overall_width, _ = im.size
        _, _, w, _ = draw_handle.textbbox((0, 0), header_text, font=font48)
        # Can place header text above "top" of stripe because there is v space added.
        draw_handle.text(((overall_width - w) / 2, v_position-70), header_text, "black", font=font48)

    return letter_index


def get_horizontal_stripe_dims(horizontal_stripe_files, horizontal_stripe_scales, horizontal_pad):
    total_width = 0
    scaled_heights = []
    scaled_widths = []
    horizontal_points = []
    for idx, image_file in enumerate(horizontal_stripe_files):
        total_width += horizontal_pad
        horizontal_points += [total_width]
        width, height = Image.open(image_file).size
        scaled_width = int(width * horizontal_stripe_scales[idx])
        scaled_height = int(height * horizontal_stripe_scales[idx])
        scaled_heights += [scaled_height]
        scaled_widths += [scaled_width]
        total_width += scaled_width
    total_width += horizontal_pad
    return total_width, scaled_widths, scaled_heights, horizontal_points


def pad_image_to_matching_size(file_to_pad, file_to_match_size, output_file):
    horizontal, vertical = Image.open(file_to_match_size).size
    im = Image.new('RGB', (horizontal, vertical), color="white")
    content = Image.open(file_to_pad)
    content_horizontal, content_vertical = content.size
    h_position = math.ceil((horizontal - content_horizontal)/2)
    v_position = math.ceil((vertical - content_vertical) / 2)

    im.paste(content, (h_position, v_position))
    im.save(output_file)
