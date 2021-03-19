# -*- coding: utf-8 -*-
"""
Created on Mon Mar  8 10:56:44 2021

@author: SchneiderS
"""
from torch.utils.data import DataLoader, ConcatDataset
import numpy as np
import torch
import matplotlib.pyplot as plt

import polycraft_nov_data.dataloader as mydataloader


def create_data_generators(batch_size=1, shuffle=True, classes_to_include=None):
    """ Create train/valid/test loaders for this dataset

    Args
    ----
    batch_size (int, optional)
        batch_size for all data loaders. Defaults to 1.

    shuffle (bool, optional)
        Whether we randomize the order of the datasets. Defaults to True.

    classes_to_include : list of strings or None
         If None, will include all examples
         If provided, will only include examples from the specified classes by name

    Returns
    -------
    train_loader : Pytorch DataLoader
                Contains batches of (N_CHANNELS, IM_SIZE, IM_SIZE) images for training.
                with pixel intensity values scaled to float between 0.0 and 1.0.
                Only images of the requested class labels will be present.
    valid_loader : Pytorch DataLoader
        As above, for validation set
    test_loader  : Pytorch DataLoader
        As above, for test set
    """

    total_noi_i = 10  # Total number of processed images from one environemnt i
    noe = 7  # Numer of environments
    n_p = 32  # Patch size, patch --> n_p x n_p
    scale = 0.75  # Scale factor, use 1 for original scale
    novelty = classes_to_include  # 'novel_item', 'novel_height', 'normal'

    device = 'cpu'
    kwargs = {'num_workers': 1, 'pin_memory': True} if device == 'cuda' else {}

    datasets = []

    for i in range(noe):
        dataset_env_i = mydataloader.PolycraftDataset(
            nov_type=novelty, noi=total_noi_i, env_idx=i, p_size=n_p, scale_factor=scale)
        datasets.append(dataset_env_i)

    final_dataset = ConcatDataset(datasets)

    total_noi = len(final_dataset)  # Total number of processed images from all datasets

    train_noi = int(0.7 * total_noi)  # Number of images used for training (70 %)
    valid_noi = int(0.15 * total_noi)  # Number of images used for validation (15 %)
    test_noi = total_noi - train_noi - valid_noi  # Number of images used for testing (15 %)
    train_dataset, valid_dataset, test_dataset = torch.utils.data.random_split(
        final_dataset, [train_noi, valid_noi, test_noi])

    train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True, **kwargs)
    valid_loader = DataLoader(valid_dataset, batch_size=1, shuffle=True, **kwargs)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=True, **kwargs)

    return train_loader, valid_loader, test_loader


# Train Network.
if __name__ == '__main__':

    train_dataloader, valid_dataloader, test_dataloader = create_data_generators(
        batch_size=1, shuffle=True, classes_to_include='novel_item')
    #datast = create_data_generators_for_ITEM_NOVELTY(batch_size = 1, shuffle = True, classes_to_include = None)

    print('Size of training loader', len(train_dataloader))
    print('Size of validation loader', len(valid_dataloader))
    print('Size of test loader', len(test_dataloader))

    for i, sample in enumerate(train_dataloader):  # loop over dataset
        patches1 = sample[0]
        nov_dic = sample[1]
        image = sample[2][0]
        plt.imshow(image)

    fig1 = plt.figure()

    # Plot last image and its patches
    for r in range(1, patches1.shape[1]*patches1.shape[2] + 1):

        fig1.add_subplot(patches1.shape[1], patches1.shape[2], r)
        flat_patches = torch.flatten(patches1, start_dim=1, end_dim=2)
        img = flat_patches[0, r-1, :, :, :].numpy()
        plt.imshow(np.transpose(img, (1, 2, 0)))
        plt.tick_params(top=False, bottom=False, left=False,
                        right=False, labelleft=False, labelbottom=False)
    plt.show()

    fig2 = plt.figure()
    plt.imshow(np.transpose(flat_patches[0, int(r/2), :, :, :], (1, 2, 0)))
    plt.show()
    """
    for i, sample in enumerate(valid_dataloader):#loop over dataset
        patches2 = sample[0]
        nov_dic = sample[1]
        
        valid_img = patches2[0,0,0,: :,:].numpy()
        #valid_img = np.transpose(valid_img, (1, 2, 0))
        #plt.imshow(valid_img)
    
    for i, sample in enumerate(test_dataloader):#loop over dataset
        patches3 = sample[0]
        nov_dic = sample[1]
        
        test_img = patches3[0,0,0,: :,:].numpy()
        test_img = np.transpose(test_img, (1, 2, 0))
        plt.imshow(test_img)
        
    """
