import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np

from polycraft_dataloader import create_data_generators
from polycraft_dataloader import normalize_img
from polycraft_nov_det.models.lsa.LSA_cifar10_no_est import LSACIFAR10NoEst as LSANet

def check_image_for_novelty(models_state_dict_path):
    
    # batch size depends on scale we use 0.25 --> 6, 0.5 --> 42, 0.75 --> 110, 1 --> 195
    scale = 0.5
    batch_size = 42
    
    # get dataloaders
    print('Download zipped files (if necessary), extract the ones we want to have and generate dataloaders.')
    train_loader, valid_loader, test_loader = create_data_generators(
                                                            shuffle=True, 
                                                            novelty_type='novel_item', item_to_include = 'minecraft:tnt', scale_level=scale)
    
    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
    pc_input_shape = (3, 32, 32)  # color channels, height, width
     
    #Load parameters of trained model
    model = LSANet(pc_input_shape, batch_size)
    model.load_state_dict(torch.load(models_state_dict_path, map_location=torch.device('cpu')))
    model.eval()
    model.to(device)
    
    loss_func = nn.MSELoss()
    loss_func_pp = nn.MSELoss(reduction = 'none')
    
    
    for i, sample in enumerate(train_loader):

        # sample contains all patches of one screen image and its novelty description
        patches = sample[0]
        nov_dic = sample[1]

        # Dimensions of flat_patches: [1, batch_size, color channels, height of patch, width of patch]
        flat_patches = torch.flatten(patches, start_dim=1, end_dim=2)

        #Apply model to image patches
        x = flat_patches[0].float().to(device)
        x_rec, z = model(x)
        x_rec = normalize_img(x_rec)
            
        
        fig1 = plt.figure()
        
        #Plot input
        for r in range(1, patches.shape[1]*patches.shape[2] + 1):
        
            fig1.add_subplot(patches.shape[1], patches.shape[2], r)
            img = x[r-1, :, :, :]
                
            plt.imshow(np.transpose((img).detach().numpy(), (1, 2, 0)))
            plt.tick_params(top=False, bottom=False, left=False,
                                right=False, labelleft=False, labelbottom=False)
            
        fig1.suptitle('Input image')
        plt.show()
            
        fig2 = plt.figure()
            
        # Plot reconstructed input
        for r in range(1, patches.shape[1]*patches.shape[2] + 1):
        
            fig2.add_subplot(patches.shape[1], patches.shape[2], r)
            img_rec = x_rec[r-1, :, :, :]
                
               
            plt.imshow(np.transpose((img_rec).detach().numpy(), (1, 2, 0)))
            plt.tick_params(top=False, bottom=False, left=False,
                                right=False, labelleft=False, labelbottom=False)
        fig2.suptitle('Reconsctructed image')
        plt.show()
            
        fig3 = plt.figure(dpi = 1200)
            
        # Plot reconstruction error
        for r in range(1, patches.shape[1]*patches.shape[2] + 1):
        
            fig3.add_subplot(patches.shape[1], patches.shape[2], r)
            
            img = x[r-1, :, :, :]
            img_rec = x_rec[r-1, :, :, :]
                
            recloss_img = loss_func_pp(img, img_rec)
            recloss_img = torch.mean(recloss_img, 0)
            
            rec = loss_func(img, img_rec)
           
            fig3.gca().set_title(round(rec.item(), 4), fontsize=4)
                
            plt.subplots_adjust(hspace = 1)   
            plt.imshow(recloss_img.detach().numpy())
            plt.tick_params(top=False, bottom=False, left=False,
                                right=False, labelleft=False, labelbottom=False)
       
        plt.show()
        
        
           
if __name__ == '__main__':
    
    state_dict_path = 'saved_statedict/saved_statedict_polycraft_scale0_5/LSA_polycraft_no_est_1000.pt'  
    check_image_for_novelty(state_dict_path)