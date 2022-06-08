import polycraft_nov_data.data_const as data_const


if __name__ == "__main__":
    class_to_lens = {}
    for dir in data_const.dataset_root.iterdir():
        if dir.is_dir():
            lens = []
            for ep in dir.iterdir():
                lens += [len(list(ep.glob("*.png")))]
            lens = sorted(lens)
            class_to_lens[dir.stem] = lens
    print(class_to_lens)
    norm_lens = class_to_lens["normal"]
    norm_percentiles = [norm_lens[len(norm_lens)//20], norm_lens[(19 * len(norm_lens))//20]]
    print(f"Normal 5%-95%:{norm_percentiles[0]}-{norm_percentiles[1]}")
    game_lens = sorted(class_to_lens["fence"] + class_to_lens["tree_easy"])
    game_percentiles = [game_lens[len(game_lens)//20], game_lens[(19 * len(game_lens))//20]]
    print(f"Gameplay Mod 5%-95%:{game_percentiles[0]}-{game_percentiles[1]}")
    item_names = [key for key in class_to_lens.keys()
                  if key not in ["normal", "fence", "tree_easy"]]
    item_lens = sorted([x for name in item_names for x in class_to_lens[name]])
    item_percentiles = [item_lens[len(item_lens)//20], item_lens[(19 * len(item_lens))//20]]
    print(f"Item Mod 5%-95%:{item_percentiles[0]}-{item_percentiles[1]}")
