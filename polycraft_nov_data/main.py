import numpy as np
import torch
import matplotlib.pyplot as plt

import polycraft_nov_data.dataloader as mydataloader


# Train Network.
if __name__ == '__main__':

    train_dataloader, valid_dataloader, test_dataloader = mydataloader.create_data_generators(
                                                            shuffle=True,
                                                            novelty_type='normal')

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
