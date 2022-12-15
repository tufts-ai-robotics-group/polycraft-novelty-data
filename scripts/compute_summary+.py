import polycraft_nov_data.novelcraft_const as nc_const


if __name__ == "__main__":
    norm_root = nc_const.DATASET_ROOT / "normal"
    num_episodes = 0
    for norm_dir in norm_root.iterdir():
        if norm_dir.is_dir() and "normal_" in norm_dir.name:
            for _ in norm_dir.iterdir():
                num_episodes += 1
    print(f"Number of episodes in NovelCraft+: {num_episodes}")
