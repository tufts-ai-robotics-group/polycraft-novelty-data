from pathlib import Path
import shutil

import pandas as pd

import polycraft_nov_data.novelcraft_const as nc_const


if __name__ == "__main__":
    targets_df = pd.read_csv(nc_const.dataset_root / "targets.csv")
    nov_to_ex_frame = {}
    for _, row in targets_df.iterrows():
        raw_path, nov_percent = row
        nov_type, episode, frame = raw_path.split("/")
        # ignore dropped bedrock class
        if nov_type == "item_bedrock":
            continue
        # add frame if it contains large number of novel pixels
        if nov_percent >= .05 and nov_percent < .08 and nov_type not in nov_to_ex_frame:
            nov_to_ex_frame[nov_type] = raw_path
    if len(nov_to_ex_frame) != 53:
        raise Exception(f"Number of examples {len(nov_to_ex_frame)} not expected value 53")
    # create output folder and variables
    out_folder = Path("novel_examples")
    out_folder.mkdir(exist_ok=True)
    latex_headers = []
    latex_paths = []
    for i, (nov_type, raw_path) in enumerate(nov_to_ex_frame.items()):
        # convert nov_type to novelty label
        if "item" in nov_type:
            item_parts = nov_type.split("_")[1:]
            nov_label = "Item:" + "".join([item_part.capitalize() for item_part in item_parts])
        elif "fence" == nov_type:
            nov_label = "Gameplay:Fence"
        else:
            nov_label = "Gameplay:Tree"
        # copy image to output folder
        ex_img = nc_const.DATASET_ROOT / (raw_path + ".png")
        out_path = out_folder / f"{nov_type}.png"
        shutil.copy(ex_img, out_path)
        # add to LaTex output
        latex_headers.append(nov_label)
        latex_paths.append(str(out_path).replace("\\", "/"))
    # hack to make tree right after fence
    latex_headers.insert(1, latex_headers.pop(-1))
    latex_paths.insert(1, latex_paths.pop(-1))
    # output rows for latex table
    col_count = 5
    i = 0
    latex_output = ""
    while i < len(latex_headers):
        row_header_line = ""
        row_path_line = ""
        for _ in range(col_count):
            ender = " &\n" if i % col_count != col_count - 1 else " \n"
            row_header_line += latex_headers[i] + ender
            row_path_line += "\\includegraphics[width=\\WWW]{figures/" + \
                latex_paths[i] + "}" + ender
            i += 1
            if i >= len(latex_headers):
                break
        row_header_line += "\\\\\n"
        row_path_line += "\\\\\n"
        latex_output += row_header_line + row_path_line
    print(latex_output)
