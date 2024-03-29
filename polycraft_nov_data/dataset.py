from pathlib import Path
import shutil
from typing import Any, Callable, Dict, List, Optional, Tuple
import urllib.request

import pandas as pd
import numpy as np
from torchvision.datasets import DatasetFolder
from torchvision.datasets.folder import default_loader

import polycraft_nov_data.episode_const as ep_const
import polycraft_nov_data.novelcraft_const as nc_const


def download_datasets(refresh=False):
    """Download Polycraft datasets if not downloaded

    Args:
        refresh (bool, optional): Delete all folders to force redownload. Defaults to False.
    """
    if refresh:
        for f in nc_const.DATASET_ROOT.iterdir():
            if f.is_dir():
                shutil.rmtree(f)
    # assume data is downloaded if folder contains subfolders
    if sum((1 if f.is_dir() else 0) for f in nc_const.DATASET_ROOT.iterdir()) < 1:
        urls = [
            ep_const.DATA_URL,
            nc_const.DATA_URL,
            nc_const.NOVELCRAFT_PLUS_URL
        ]
        roots = [
            ep_const.DATASET_ROOT,
            nc_const.DATASET_ROOT,
            nc_const.DATASET_ROOT / "normal",
        ]
        for url, root in zip(urls, roots):
            # download, extract, and delete zip of the data
            zip_path = root / Path("temp.zip")
            urllib.request.urlretrieve(url, zip_path)
            shutil.unpack_archive(zip_path, root)
            zip_path.unlink()


class NovelCraft(DatasetFolder):
    def __init__(self,
                 split: str,
                 transform: Optional[Callable] = None,
                 target_transform: Optional[Callable] = None,
                 training_plus: bool = False) -> None:
        download_datasets()
        # whether to use NovelCraft+ training set
        self.training_plus = training_plus
        # make split specify novel, norm, or both
        if split not in set(item.value for item in nc_const.SplitEnum):
            raise ValueError(f"NovelCraft split '{split}' not one of following:\n" +
                             "\n".join(set(item.value for item in nc_const.SplitEnum)))
        self.split = split
        # novel percentage data
        self.id_to_percent = pd.read_csv(nc_const.DATASET_TARGETS).to_numpy()
        # split data
        self.ep_to_split = pd.read_csv(nc_const.DATASET_SPLITS).to_numpy()[:, :2]
        # init dataset
        root = nc_const.DATASET_ROOT
        super().__init__(root, default_loader, None, transform, target_transform,
                         self.is_valid_file)

    def find_classes(self, directory: str) -> Tuple[List[str], Dict[str, int]]:
        classes = []
        split_enum = nc_const.SplitEnum
        if self.split not in [split_enum.VALID_NOVEL, split_enum.TEST_NOVEL]:
            classes += nc_const.NORMAL_CLASSES
        if self.split in [split_enum.VALID, split_enum.VALID_NOVEL]:
            classes += nc_const.NOVEL_VALID_CLASSES
        if self.split in [split_enum.TEST, split_enum.TEST_NOVEL]:
            classes += nc_const.NOVEL_TEST_CLASSES
        # classes always mapped to same target index but only includes non-empty classes
        return classes, {cls: nc_const.ALL_CLASS_TO_IDX[cls] for cls in classes}

    def is_valid_file(self, path: str) -> bool:
        path = Path(path)
        split_enum = nc_const.SplitEnum
        # reject file if not png
        if path.suffix.lower() != ".png":
            return False
        # reject file if from NovelCraft+ and want only normal set
        if "normal_" in str(path):
            if not self.training_plus:
                return False
            # accept file if from NovelCraft+ and want that training set
            else:
                if self.split == split_enum.TRAIN:
                    return True
        # reject file if not above novel percentage
        cls_label = path.parts[-3]
        cur_id = "/".join(path.parts[-3:-1] + (path.stem,))
        if cls_label != "normal":
            id_ind = np.argwhere(self.id_to_percent == cur_id)
            if id_ind.shape[0] != 1:
                print("Warning found %i targets for image: %s" % (id_ind.shape[0], cur_id))
            else:
                novel_percent = self.id_to_percent[id_ind[0, 0], 1]
                if novel_percent < nc_const.NOV_THRESH:
                    return False
        # reject file if not in split based on class
        if cls_label in nc_const.NOVEL_VALID_CLASSES and \
                self.split not in [split_enum.VALID, split_enum.VALID_NOVEL]:
            return False
        if cls_label in nc_const.NOVEL_TEST_CLASSES and \
                self.split not in [split_enum.TEST, split_enum.TEST_NOVEL]:
            return False
        # reject normal class file if episode not in split
        if cls_label in nc_const.NORMAL_CLASSES:
            cur_ep = "/".join(path.parts[-3:-1])
            ep_ind = np.argwhere(self.ep_to_split == cur_ep)
            if ep_ind.shape[0] != 1:
                print("Warning found %i splits for episode: %s" % (ep_ind.shape[0], cur_ep))
            split_label = self.ep_to_split[ep_ind[0, 0], 1]
            if split_label == "train" and self.split not in [split_enum.TRAIN]:
                return False
            if split_label == "valid" and \
                    self.split not in [split_enum.VALID, split_enum.VALID_NORM]:
                return False
            if split_label == "test" and self.split not in [split_enum.TEST, split_enum.TEST_NORM]:
                return False
        # file has passed all tests
        return True


class EpisodeDataset(DatasetFolder):
    def __init__(self,
                 split: str,
                 transform: Optional[Callable] = None) -> None:
        download_datasets()
        # validate split choice
        if split not in set(item.value for item in ep_const.SplitEnum):
            raise ValueError(f"NovelCraft split '{split}' not one of following:\n" +
                             "\n".join(set(item.value for item in ep_const.SplitEnum)))
        self.split = split
        # split data
        self.ep_to_split = pd.read_csv(nc_const.DATASET_SPLITS).to_numpy()[:, :2]
        # init dataset
        root = nc_const.DATASET_ROOT
        super().__init__(root, default_loader, None, transform, None, self.is_valid_file)

    def find_classes(self, directory: str) -> Tuple[List[str], Dict[str, int]]:
        classes = []
        split_enum = ep_const.SplitEnum
        if self.split != split_enum.TEST_NOVEL:
            classes += ep_const.NORMAL_CLASSES
        if self.split == split_enum.TEST or self.split == split_enum.TEST_NOVEL:
            classes += ep_const.TEST_CLASSES
        # classes always mapped to same target index but only includes non-empty classes
        return classes, {cls: ep_const.ALL_CLASS_TO_IDX[cls] for cls in classes}

    def is_valid_file(self, path: str) -> bool:
        path = Path(path)
        # reject file if not png
        if path.suffix.lower() != ".png":
            return False
        # reject file if from NovelCraft+
        if "normal_" in str(path):
            return False
        # reject normal class file if episode not in split
        cls_label = path.parts[-3]
        split_enum = ep_const.SplitEnum
        if cls_label in ep_const.NORMAL_CLASSES:
            cur_ep = "/".join(path.parts[-3:-1])
            ep_ind = np.argwhere(self.ep_to_split == cur_ep)
            if ep_ind.shape[0] != 1:
                print("Warning found %i splits for episode: %s" % (ep_ind.shape[0], cur_ep))
            split_label = self.ep_to_split[ep_ind[0, 0], 1]
            if split_label == "train" and self.split not in [split_enum.TRAIN]:
                return False
            if split_label == "valid" and \
                    self.split not in [split_enum.VALID]:
                return False
            if split_label == "test" and self.split not in [split_enum.TEST, split_enum.TEST_NORM]:
                return False
        return True

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        # overload item retrieval to return (sample, path) since target isn't needed
        sample, target = super().__getitem__(index)
        path, target = self.samples[index]
        # get path relative to the dataset root, so path is "class/ep_num/frame_num"
        return sample, str(Path(path).relative_to(ep_const.DATASET_ROOT))
