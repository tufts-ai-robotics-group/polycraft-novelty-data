import pandas as pd

import polycraft_nov_data.novelcraft_const as novelcraft_const


if __name__ == "__main__":
    targets_df = pd.read_csv(novelcraft_const.dataset_root / "targets.csv")
    valid_eps = []
    cur_ep = 0
    cur_frames = 0
    for _, row in targets_df.iterrows():
        raw_path, nov_percent = row
        nov_type, episode, frame = raw_path.split("/")
        if episode != cur_ep:
            cur_ep = episode
            cur_frames = 0
        if nov_percent >= novelcraft_const.NOV_THRESH and int(frame) in [1, 11, 21, 41]:
            cur_frames += 1
            if cur_frames == 3 and int(frame) >= 41:
                valid_eps += [f"{nov_type}/{episode}"]
    print(valid_eps)
