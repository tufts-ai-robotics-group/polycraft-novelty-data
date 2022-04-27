from pathlib import Path
import shutil
import urllib.request

import pandas as pd
import numpy as np
from torch.utils.data import Dataset
from torchvision.datasets import ImageFolder

import polycraft_nov_data.data_const as data_const


class TrippleDataset(Dataset):
    """Combine three datasets (we have one for each scale)
    """
    def __init__(self, dataset1, dataset2, dataset3):
        self.dataset1 = dataset1
        self.dataset2 = dataset2
        self.dataset3 = dataset3

    def __getitem__(self, index):
        return self.dataset1[index], self.dataset2[index], self.dataset3[index]

    def __len__(self):
        return len(self.dataset1)


class QuattroDataset(Dataset):
    """Combine four datasets (we have one for scale 0.5 and scale 0.75 and two
       for scale 1 (32x32 patch and 16x16 patch)
    """
    def __init__(self, dataset1, dataset2, dataset3, dataset4):
        self.dataset1 = dataset1
        self.dataset2 = dataset2
        self.dataset3 = dataset3
        self.dataset4 = dataset4

    def __getitem__(self, index):
        return (
            self.dataset1[index],
            self.dataset2[index],
            self.dataset3[index],
            self.dataset4[index]
        )

    def __len__(self):
        return len(self.dataset1)


def download_datasets():
    """Download Polycraft datasets if not downloaded
    """
    # assume data is downloaded if folder contains more than the 3 default files
    if sum(1 for _ in data_const.DATASET_ROOT.iterdir()) <= 3:
        # download, extract, and delete zip of the data
        zip_path = data_const.DATASET_ROOT / Path("polycraft_data.zip")
        urllib.request.urlretrieve(data_const.DATA_URL, zip_path)
        shutil.unpack_archive(zip_path, data_const.DATASET_ROOT)
        zip_path.unlink()


class PolycraftDataset(ImageFolder):
    def __init__(self, transform=None):
        download_datasets()
        super().__init__(data_const.DATASET_ROOT, transform=transform)
        self.class_to_idx = PolycraftDataset.correct_class_to_idx()

    @staticmethod
    def correct_class_to_idx():
        # update class_to_idx for easier classification
        class_ordering = data_const.NORMAL_CLASSES + data_const.NOVEL_VALID_CLASSES + \
            data_const.NOVEL_TEST_CLASSES
        return {c: i for i, c in enumerate(class_ordering)}

    @staticmethod
    def make_dataset(directory, class_to_idx, extensions=None, is_valid_file=None):
        class_to_idx = PolycraftDataset.correct_class_to_idx()
        # make ImageFolder dataset
        instances = ImageFolder.make_dataset(directory, class_to_idx, extensions, is_valid_file)
        # load target CSV
        target_df = pd.read_csv(data_const.DATASET_TARGETS)
        ids = target_df["id"].to_numpy()
        novel_percents = target_df["novel_percent"].to_numpy()
        reject_indices = []
        # apply correction to novel labels
        for i, instance in enumerate(instances):
            raw_path, raw_target = instance
            path = Path(raw_path)
            novel_target = class_to_idx[path.parts[-3]]
            cur_id = "/".join(path.parts[-3:-1] + (path.stem,))
            if raw_target != class_to_idx["normal"]:
                index = np.argwhere(ids == cur_id)
                if index.shape[0] != 1:
                    print("Warning found %i targets for image: %s" % (index.shape[0], cur_id))
                else:
                    if novel_percents[index] >= data_const.NOV_THRESH:
                        instances[i] = (raw_path, novel_target)
                    else:
                        reject_indices.append(i)
        # remove images with ambiguous labels, starting at end for consistent indexing
        for i in reversed(reject_indices):
            del instances[i]
        return instances


def polycraft_dataset(transform=None):
    return PolycraftDataset(transform=transform)
