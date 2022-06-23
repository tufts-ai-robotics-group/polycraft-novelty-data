from pathlib import Path
import shutil
from typing import Callable, Dict, List, Optional, Tuple
import urllib.request

import pandas as pd
import numpy as np
from torchvision.datasets import DatasetFolder
from torchvision.datasets.folder import default_loader

import polycraft_nov_data.data_const as data_const


def download_datasets():
    """Download Polycraft datasets if not downloaded
    """
    # assume data is downloaded if folder contains subfolders
    if sum((1 if f.is_dir() else 0) for f in data_const.DATASET_ROOT.iterdir()) < 1:
        # download, extract, and delete zip of the data
        zip_path = data_const.DATASET_ROOT / Path("polycraft_data.zip")
        urllib.request.urlretrieve(data_const.DATA_URL, zip_path)
        shutil.unpack_archive(zip_path, data_const.DATASET_ROOT)
        zip_path.unlink()


class NovelCraft(DatasetFolder):
    def __init__(self,
                 split: str,
                 transform: Optional[Callable] = None,
                 target_transform: Optional[Callable] = None) -> None:
        download_datasets()
        # make split specify novel, norm, or both
        if split not in set(item.value for item in data_const.SplitEnum):
            raise ValueError(f"NovelCraft split '{split}' not one of following:\n" +
                             "\n".join(set(item.value for item in data_const.SplitEnum)))
        self.split = split
        # novel percentage data
        self.id_to_percent = pd.read_csv(data_const.DATASET_TARGETS).to_numpy()
        # split data
        self.ep_to_split = pd.read_csv(data_const.DATASET_SPLITS).to_numpy()[:, :2]
        # init dataset
        root = data_const.DATASET_ROOT
        super().__init__(root, default_loader, None, transform, target_transform,
                         self.is_valid_file)

    def find_classes(self, directory: str) -> Tuple[List[str], Dict[str, int]]:
        classes = []
        if self.split not in [data_const.SplitEnum.VALID_NOVEL, data_const.SplitEnum.TEST_NOVEL]:
            classes += data_const.NORMAL_CLASSES
        if self.split in [data_const.SplitEnum.VALID, data_const.SplitEnum.VALID_NOVEL]:
            classes += data_const.NOVEL_VALID_CLASSES
        if self.split in [data_const.SplitEnum.TEST, data_const.SplitEnum.TEST_NOVEL]:
            classes += data_const.NOVEL_TEST_CLASSES
        return classes, {cls: data_const.ALL_CLASS_TO_IDX[cls] for cls in classes}

    def is_valid_file(self, path: str) -> bool:
        path = Path(path)
        # reject file if not png
        if not path.suffix.lower() == ".png":
            return False
        # reject file if not above novel percentage
        cls_label = path.parts[-3]
        cur_id = "/".join(path.parts[-3:-1] + (path.stem,))
        if cls_label != "normal":
            id_ind = np.argwhere(self.id_to_percent == cur_id)
            if id_ind.shape[0] != 1:
                print("Warning found %i targets for image: %s" % (id_ind.shape[0], cur_id))
            else:
                novel_percent = self.id_to_percent[id_ind[0, 0], 1]
                if novel_percent < data_const.NOV_THRESH:
                    return False
        # reject file if not in split based on class
        if cls_label in data_const.NOVEL_VALID_CLASSES and \
                self.split not in [data_const.SplitEnum.VALID, data_const.SplitEnum.VALID_NOVEL]:
            return False
        if cls_label in data_const.NOVEL_TEST_CLASSES and \
                self.split not in [data_const.SplitEnum.TEST, data_const.SplitEnum.TEST_NOVEL]:
            return False
        # reject normal class file if not episode not in split
        if cls_label in data_const.NORMAL_CLASSES:
            cur_ep = "/".join(path.parts[-3:-1])
            ep_ind = np.argwhere(self.ep_to_split == cur_ep)
            if ep_ind.shape[0] != 1:
                print("Warning found %i splits for episode: %s" % (id_ind.shape[0], cur_ep))
            split_label = self.ep_to_split[ep_ind[0, 0], 1]
            if split_label == "train" and \
                    self.split not in [data_const.SplitEnum.TRAIN]:
                return False
            if split_label == "valid" and \
                    self.split not in [data_const.SplitEnum.VALID, data_const.SplitEnum.VALID_NORM]:
                return False
            if split_label == "test" and \
                    self.split not in [data_const.SplitEnum.TEST, data_const.SplitEnum.TEST_NORM]:
                return False
        # file has passed all tests
        return True
