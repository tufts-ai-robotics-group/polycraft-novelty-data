import pandas as pd

import polycraft_nov_data.novelcraft_const as nc_const


if __name__ == "__main__":
    targets_df = pd.read_csv(nc_const.dataset_root / "targets.csv")
    type_to_episodes = {}
    type_to_frames = {}
    for _, row in targets_df.iterrows():
        raw_path, nov_percent = row
        nov_type, episode, frame = raw_path.split("/")
        if nov_percent >= nc_const.NOV_THRESH:
            # add 1 to frame count if usable
            type_to_frames[nov_type] = type_to_frames.get(nov_type, 0) + 1
            # add unseen episode to episode list
            if episode not in type_to_episodes.get(nov_type, []):
                type_to_episodes[nov_type] = type_to_episodes.get(nov_type, []) + [episode]

    summary_df = pd.DataFrame(columns=["nov_type", "num_episodes", "num_frames"])
    for nov_type, episodes in type_to_episodes.items():
        summary_df = pd.concat([
            summary_df,
            pd.DataFrame({
                "nov_type": nov_type,
                "num_episodes": len(episodes),
                "num_frames": type_to_frames[nov_type],
            }, [summary_df.shape[0]])
        ])
    summary_df.to_csv(nc_const.dataset_root / "summary.csv", index=False)
