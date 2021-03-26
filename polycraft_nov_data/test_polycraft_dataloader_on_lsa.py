import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

from polycraft_dataloader import create_data_generators
from polycraft_dataloader import normalize_img
from polycraft_nov_det.models.lsa.LSA_cifar10_no_est import LSACIFAR10NoEst as LSANet
from polycraft_nov_det.plot import plot_reconstruction_rgb



def train():

    # batch size depends on scale we use 0.25 --> 6, 0.5 --> 42, 0.75 --> 110, 1 --> 195
    scale = 0.5
    batch_size = 42

    pc_input_shape = (3, 32, 32)  # color channels, height, width

    # get dataloaders
    print('Download zipped files (if necessary), extract the ones we want to have and generate dataloaders.')
    train_loader, valid_loader, test_loader = create_data_generators(
                                                            shuffle=True, 
                                                            novelty_type='normal', scale_level=scale)
   
    print('Size of training loader', len(train_loader))
    print('Size of validation loader', len(valid_loader))
    print('Size of test loader', len(test_loader))

    # get Tensorboard writer
    writer = SummaryWriter("runs")

    # define training constants
    lr = 1e-3
    epochs = 1000
    loss_func = nn.MSELoss()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # construct model
    model = LSANet(pc_input_shape, batch_size)
    model.to(device)

    # construct optimizer
    optimizer = optim.Adam(model.parameters(), lr)

    # train model
    for epoch in range(epochs):

        print('---------- Epoch ', epoch, ' -------------')

        train_loss = 0

        for i, sample in enumerate(train_loader):

            optimizer.zero_grad()

            # sample contains all patches of one screen image and its novelty description
            patches = sample[0]
            nov_dic = sample[1]

            # Dimensions of flat_patches: [1, batch_size, color channels, height of patch, width of patch]
            flat_patches = torch.flatten(patches, start_dim=1, end_dim=2)

            #img = flat_patches[0, 5, :, :, :].numpy()
            #plt.imshow(np.transpose(img, (1,2,0)))
            #plt.tick_params(top=False, bottom=False, left=False, right=False, labelleft=False, labelbottom=False)
            # plt.show()

            x = flat_patches[0].float().to(device)

            x_rec, z = model(x)

            batch_loss = loss_func(x_rec, x)
            batch_loss.backward()

            optimizer.step()

            # logging
            train_loss += batch_loss.item() * batch_size

        # calculate and record train loss
        av_train_loss = train_loss / len(train_loader)
        writer.add_scalar("Average Train Loss", av_train_loss, epoch)
        print('Average training loss  ', av_train_loss)

        # get validation loss
        valid_loss = 0

        for i, target in enumerate(valid_loader):

            # sample contains all patches of one screen image and its novelty description
            patches = sample[0]
            nov_dic = sample[1]

            # Dimensions of flat_patches: [1, batch_size, color channels, height of patch, width of patch]
            flat_patches = torch.flatten(patches, start_dim=1, end_dim=2)

            x = flat_patches[0].float().to(device)
            x_rec, z = model(x)
            
            batch_loss = loss_func(x, x_rec)
            valid_loss += batch_loss.item() * batch_size

        av_valid_loss = valid_loss / len(valid_loader)
        writer.add_scalar("Average Validation Loss", av_valid_loss, epoch)

        print('Average validation loss  ', av_valid_loss)

        # get reconstruction visualization
        x_rec = normalize_img(x_rec)
        writer.add_figure("Reconstruction Vis", plot_reconstruction_rgb(x, x_rec), epoch)

        # TODO add latent space visualization (try PCA or t-SNE for projection)
        # save model
        if ((epoch + 1) %  20) == 0 :
            torch.save(model.state_dict(), "saved_statedict/LSA_polycraft_no_est_%d.pt" % (epoch + 1,))

    return model


    # Train Network.
if __name__ == '__main__':

    LSA = train()
