import pandas as pd

import polycraft_nov_data.data_const as data_const


class SplitStats:
    def __init__(self) -> None:
        self.num_ep = 0
        self.frames_raw = 0
        self.frames_filtered = 0


if __name__ == "__main__":
    # compute stats for novel data
    targets_df = pd.read_csv(data_const.dataset_root / "targets.csv")
    type_to_episodes = {}
    type_to_frames_raw = {}
    type_to_frames_filtered = {}
    for _, row in targets_df.iterrows():
        raw_path, nov_percent = row
        nov_type, episode, frame = raw_path.split("/")
        # always add 1 to raw frame count
        type_to_frames_raw[nov_type] = type_to_frames_raw.get(nov_type, 0) + 1
        # add unseen episode to episode list
        if episode not in type_to_episodes.get(nov_type, []):
            type_to_episodes[nov_type] = type_to_episodes.get(nov_type, []) + [episode]
        if nov_percent >= data_const.NOV_THRESH:
            # add 1 to filtered frame count if usable
            type_to_frames_filtered[nov_type] = type_to_frames_filtered.get(nov_type, 0) + 1
    # combine classes into splits
    valid_stats = SplitStats()
    for valid_class in data_const.NOVEL_VALID_CLASSES:
        valid_stats.num_ep += len(type_to_episodes.get(valid_class, []))
        valid_stats.frames_raw += type_to_frames_raw.get(valid_class, 0)
        valid_stats.frames_filtered += type_to_frames_filtered.get(valid_class, 0)
    test_stats = SplitStats()
    for test_class in data_const.NOVEL_TEST_CLASSES:
        test_stats.num_ep += len(type_to_episodes.get(test_class, []))
        test_stats.frames_raw += type_to_frames_raw.get(test_class, 0)
        test_stats.frames_filtered += type_to_frames_filtered.get(test_class, 0)
    # compute stats for normal data
    splits_df = pd.read_csv(data_const.dataset_root / "splits.csv")
    for _, row in splits_df.iterrows():
        raw_ep, split, num_frames = row
        nov_type, episode = raw_ep.split("/")
        nov_split = nov_type + "_" + split
        # collect frame count from normal data, with no filtering
        if nov_type == "normal":
            type_to_frames_raw[nov_split] = type_to_frames_raw.get(nov_split, 0) + num_frames
            type_to_frames_filtered[nov_split] = type_to_frames_filtered.get(nov_split, 0) + \
                num_frames
            type_to_episodes[nov_split] = type_to_episodes.get(nov_split, []) + [episode]
        # collect filtered frame count for other normal classes
        else:
            type_to_frames_filtered[nov_split] = type_to_frames_filtered.get(nov_split, 0) + \
                num_frames
            type_to_episodes[nov_split] = type_to_episodes.get(nov_split, []) + [episode]
    # collect raw frame count for normal classes
    for nov_type in data_const.NORMAL_CLASSES:
        if nov_type == "normal":
            continue
        for split in ["train", "valid", "test"]:
            nov_split = nov_type + "_" + split
            for ep in type_to_episodes.get(nov_split, []):
                ep_dir = data_const.DATASET_ROOT / (nov_type + "/" + ep)
                # count number of PNGs for raw count
                num_frames = len(list(ep_dir.glob("*.png")))
                type_to_frames_raw[nov_split] = type_to_frames_raw.get(nov_split, 0) + num_frames
    # normal split stats
    norm_train_stats = SplitStats()
    norm_valid_stats = SplitStats()
    norm_test_stats = SplitStats()
    norm_stats_list = [norm_train_stats, norm_valid_stats, norm_test_stats]
    for nov_type in data_const.NORMAL_CLASSES:
        for split, cur_stats in zip(["train", "valid", "test"], norm_stats_list):
            nov_split = nov_type + "_" + split
            cur_stats.num_ep += len(type_to_episodes.get(nov_split, []))
            cur_stats.frames_raw += type_to_frames_raw.get(nov_split, 0)
            cur_stats.frames_filtered += type_to_frames_filtered.get(nov_split, 0)
    # print results
    labels = ["norm_train", "norm_valid", "norm_test", "novel_valid", "novel_test"]
    stats = [norm_train_stats, norm_valid_stats, norm_test_stats, valid_stats, test_stats]
    for i in range(len(labels)):
        print(labels[i])
        print("Num. Ep:")
        print(stats[i].num_ep)
        print("Frames Raw:")
        print(stats[i].frames_raw)
        print("Frames Filtered:")
        print(stats[i].frames_filtered)
        print()
