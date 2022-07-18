import math
from pathlib import Path

import numpy as np
import pandas as pd

from polycraft_nov_data.dataset import NovelCraft
import polycraft_nov_data.novelcraft_const as nc_const


def split_error(cur_split, goal_split):
    return sum([abs(1 - cur / goal) for cur, goal in zip(cur_split, goal_split)])


if __name__ == "__main__":
    # nested dict with keys target, ep and value image count
    target_to_eps = {}
    split_enum = nc_const.SplitEnum
    datasets = [
        NovelCraft(split_enum.TRAIN.value),
        NovelCraft(split_enum.VALID.value),
        NovelCraft(split_enum.TEST.value)
    ]
    for dataset in datasets:
        for img_path_str, target_idx in dataset.samples:
            img_path = Path(img_path_str)
            target = nc_const.ALL_IDX_TO_CLASS[target_idx]
            ep_num = img_path.parent.stem
            eps_to_count = target_to_eps.get(target, {})
            eps_to_count[ep_num] = eps_to_count.get(ep_num, 0) + 1
            target_to_eps[target] = eps_to_count

    rng = np.random.default_rng(seed=42)
    split_df = pd.DataFrame(columns=["episode", "split", "num_frames"])
    for target in nc_const.NORMAL_CLASSES:
        eps_to_count = target_to_eps[target]
        ep_nums = np.array(list(eps_to_count.keys()))
        ep_counts = np.array(list(eps_to_count.values()))
        target_count = np.sum(ep_counts)
        # get desired split lengths with the first non-zero entry resolving rounding errors
        goal_split_lens = [math.ceil(target_count * percent)
                           for percent in nc_const.NORMAL_SPLIT]
        first_non_zero = 0
        for j, split_length in enumerate(goal_split_lens):
            if split_length > 0:
                first_non_zero = j
                break
        goal_split_lens[first_non_zero] += target_count - sum(goal_split_lens)
        # get desired number of episodes with the first non-zero entry resolving rounding errors
        split_eps = [math.ceil(len(ep_counts) * percent)
                     for percent in nc_const.NORMAL_SPLIT]
        first_non_zero = 0
        for j, split_length in enumerate(split_eps):
            if split_length > 0:
                first_non_zero = j
                break
        split_eps[first_non_zero] += len(ep_counts) - sum(split_eps)
        # sample random splits and select the best found
        best_split = None
        best_error = 10e6
        for i in range(100):
            perm_ind = rng.permutation(len(ep_counts))
            ep_split = [perm_ind[:split_eps[0]],
                        perm_ind[split_eps[0]:split_eps[0] + split_eps[1]],
                        perm_ind[split_eps[0] + split_eps[1]:]]
            cur_split_lens = [np.sum(ep_counts[ind]) for ind in ep_split]
            cur_error = split_error(cur_split_lens, goal_split_lens)
            if cur_error < best_error:
                best_split = ep_split
                best_error = cur_error
        # report results
        print("GOAL:")
        print(goal_split_lens)
        print("GOT:")
        print([np.sum(ep_counts[ind]) for ind in best_split])
        # convert to episode numbers
        best_split_eps = [ep_nums[split] for split in best_split]
        # add to dataframe
        labels = ["train", "valid", "test"]
        for i in range(len(best_split)):
            # sort episodes for cleaner output
            sort_ind = np.argsort(best_split_eps[i])
            for j in range(len(best_split[i])):
                sort_j = sort_ind[j]
                split_df = pd.concat([
                    split_df,
                    pd.DataFrame({
                        "episode": target + "/" + best_split_eps[i][sort_j],
                        "split": labels[i],
                        "num_frames": str(ep_counts[best_split[i][sort_j]]),
                    }, [split_df.shape[0]])
                ])
    split_df.to_csv(nc_const.dataset_root / "splits.csv", index=False)
