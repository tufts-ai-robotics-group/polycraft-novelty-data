import os
import re
import torch
import zipfile
import json
import torchvision

from torch.utils.data import Dataset, DataLoader, ConcatDataset
from skimage import io
from torchvision.datasets.utils import download_url
from skimage.transform import rescale
import torchvision.transforms as transforms

import polycraft_nov_data.image_transforms as image_transforms


minecraft_item_names = [
        'anvil', 'beacon', 'bed', 'bedrock', 'bookshelf', 'brewing_stand',
        'cactus', 'cake', 'cauldron', 'clay', 'coal_block', 'cobblestone',
        'crafting_table', 'daylight_detector', 'deadbush', 'diamond_block',
        'dirt', 'dispenser', 'dropper', 'emerald_block', 'enchanting_table',
        'glowstone', 'gravel', 'hopper', 'ice', 'iron_block', 'jukebox',
        'lapis_block', 'leaves', 'lever', 'log', 'mycelium', 'netherrack',
        'noteblock', 'obsidian', 'piston', 'planks', 'prismarine', 'pumpkin',
        'quartz_block', 'reeds', 'sand', 'sandstone', 'sapling', 'sea_lantern',
        'slime', 'snow', 'soul_sand', 'sponge', 'stone', 'stonebrick',
        'tallgrass', 'tnt', 'torch', 'vine', 'waterlily', 'web', 'wheat', 'wool'
    ]
minecraft_item_names = ["minecraft:" + name for name in minecraft_item_names]


def preprocess_image(image, scale_factor, p_size):
    image = crop_and_scale(image, scale_factor)
    image = extract_patches(image, p_size)
    return image


def crop_and_scale(image, scale_factor):

    image = image[0:234, :, :]
    image = rescale(image, (scale_factor, scale_factor, 1), anti_aliasing=True)

    return image


def extract_patches(image, p_size):

    # Extract patches
    stride = int(p_size/2)  # patch stride
    image = torch.from_numpy(image)
    patches = image.unfold(0, p_size, stride).unfold(1, p_size, stride)

    return patches

def create_random_data_generators(image_scale, batch_size, data_dir):
    """
    Preprocess Images
    * Crop away polycraft bar 
    * Change image resolution by scale factor image_scale
    * Extract patches of size batch_size x batch_size randomly
    and return dataloader.
       
    :scale: scale used for decrease in resolution, set to 1 for no original resolution
    :param batch_size: size of the extracted batch
    :param data_dir: root directory where the dataset is
     
    :return: train, validation and test dataloader
    """

    trnsfrm = transforms.Compose([
            transforms.ToTensor(),
            image_transforms.CropUI(),
            image_transforms.ScaleImage(image_scale),
            transforms.RandomCrop([32,32]),
        ])

    data = torchvision.datasets.ImageFolder(root = data_dir, transform = trnsfrm)
    
    total_noi = len(data)
    train_noi = int(0.7 * total_noi)  # Number of images used for training (70 %)
    valid_noi = int(0.15 * total_noi)  # Number of images used for validation (15 %)
    test_noi = total_noi - train_noi - valid_noi  # Number of images used for testing (15 %)
    
    train_dataset, valid_dataset, test_dataset = torch.utils.data.random_split(
        data, [train_noi, valid_noi, test_noi],
        generator=torch.Generator().manual_seed(42)
    )

    train_loader = DataLoader(train_dataset, batch_size, shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size, shuffle=True)

    return train_loader, valid_loader, test_loader

def create_data_generators(shuffle=True, novelty_type='normal', item_to_include='None',
                           scale_level=1):
    """ Create train/valid/test loaders for this dataset

    Args
    ----
    shuffle (bool, optional)
        Whether we randomize the order of the datasets. Defaults to True.

    novelty_type (string)
         'normal' --> No novelty
         'novel_item' --> "New" minecraft item
         'novel_heigth' --> z-position of item is set to value greater 4
         If not given, will include all non-novel/normal examples
         If provided, will only include examples from the specified novelty type
    item_to_include (string)
         If given, the specific item is searched for and its json/images used.
         'minecraft:bed', 'minecraft:clay', ...
    scale (float)
         Can be set explicitely in test_polycraft_dataloader_on_lsa.py file according
         to batch size, set to 1 (no rescaling) per default

    Returns
    -------
    train_loader : Pytorch DataLoader
                Contains batches of (N_CHANNELS, IM_SIZE, IM_SIZE) images for training.
                with pixel intensity values scaled to float between 0.0 and 1.0.

    valid_loader : Pytorch DataLoader
        As above, for validation set
    test_loader  : Pytorch DataLoader
        As above, for test set
    """

    total_noi_i = 499 # Number of processed images from one environemnt i (max. 499)
    noe = 9  # Numer of environments (max. 9 )
    n_p = 32  # Patch size, patch --> n_p x n_p

    novelty = novelty_type
    datasets = []

    for i in range(noe):

        # Load only images of the environment which includes images of the stated novel item.
        if item_to_include is not None and novelty == 'novel_item':
            dataset_env_i = PolycraftDatasetWithSpecificItem(
                nov_type=novelty, noi=total_noi_i, env_idx=i, p_size=n_p, scale_factor=scale_level,
                item_name=item_to_include)
            datasets.append(dataset_env_i)
            # We only process the one environment with the item (maybe change this
            # if we have more than one environement per novel_item!?)
            break

        # No specific item given which should be included.
        else:
            dataset_env_i = PolycraftDatasetNoSpecificItem(
                nov_type=novelty, noi=total_noi_i, env_idx=i, p_size=n_p, scale_factor=scale_level)
            datasets.append(dataset_env_i)

    final_dataset = ConcatDataset(datasets)

    total_noi = len(final_dataset)  # Total number of processed images from all datasets

    if(total_noi < 7):
        print('Number of samples too small for splitting dataset in training-/valid-/test set.')

    train_noi = int(0.7 * total_noi)  # Number of images used for training (70 %)
    valid_noi = int(0.15 * total_noi)  # Number of images used for validation (15 %)
    test_noi = total_noi - train_noi - valid_noi  # Number of images used for testing (15 %)
    train_dataset, valid_dataset, test_dataset = torch.utils.data.random_split(
        final_dataset, [train_noi, valid_noi, test_noi],
        generator=torch.Generator().manual_seed(42)
    )

    train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=1, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=True)

    return train_loader, valid_loader, test_loader


def convert_item_to_encoded_item(item):

    encoded_item = []

    for item_name in minecraft_item_names:
        if item_name == item:
            encoded_item.append(1)
        else:
            encoded_item.append(0)

    return encoded_item


def check_if_item_exists(name):

    item_exists = False

    for item_name in minecraft_item_names:
        if item_name == name:
            item_exists = True
            print('----------------------------------------------------------')
            print('Item ', name, ' is a valid minecraft item.')
            print('----------------------------------------------------------')
    if not item_exists:
        print('-----------------------------------------------------------------------')
        print('Item ', name, ' does not exists in Minecraft, please choose a valid item name!')
        print('-----------------------------------------------------------------------')

    return item_exists


def nov_dict_to_encoded_nov(json):

    vector = [[], []]
    encoded_item = []

    # First check if we have a novelty described in the json file:
    if bool(json):

        if json['novelty'][0]['noveltyType'] == 'NewItem':
            vector[0].append(1)
            item = json['novelty'][0]['noveltyItem']
            encoded_item = convert_item_to_encoded_item(item)
            vector[1].append(encoded_item)

        if json['novelty'][0]['noveltyType'] == 'newPosition':
            vector[0].append(2)

    else:

        vector[0].append(0)

    return vector


def count_png_in_env(path, env_name, env_idx):

    img_ctr = 0

    with zipfile.ZipFile(path, 'r') as zip:
        file_names = zip.namelist()

        for file_name in file_names:

            # We need the path separation symbol in order to not count
            # environemnts which only contain the env_idx (4 --> 41)
            if file_name.find(env_name + str(env_idx) + '/') != -1 or \
                    file_name.find(env_name + str(env_idx) + '\\') != -1:

                if file_name.find('.png') != -1:

                    img_ctr += 1

    return img_ctr


class PolycraftDatasetNoSpecificItem(Dataset):

    def __init__(self, nov_type, noi, env_idx, p_size, scale_factor):

        self.nov_type = nov_type
        self.allpatches = []
        self.allnovbins = []
        self.all_images = []

        self.root_normal = 'datasets' + os.sep + 'normal_data'
        self.file_name_normal = 'normal_data.zip'

        self.root_nov = 'datasets' + os.sep + 'novel_data'
        self.file_name_nov_item = 'item_novelty.zip'
        self.file_name_nov_height = 'height_novelty.zip'

        self.path_normal = self.root_normal + os.sep + self.file_name_normal
        self.path_nov_item = self.root_nov + os.sep + self.file_name_nov_item
        self.path_nov_height = self.root_nov + os.sep + self.file_name_nov_height

        self.scale_factor = scale_factor
        self.p_size = p_size

        # print('path_normal', self.path_normal) #\\

        nov_dict = {}
        old_img = []

        find_json = False
        find_png = False

        # Load zipped files from BOX
        path, root, env_name = self.load_zipped_files()

        noi_check = count_png_in_env(path, env_name, env_idx)

        # Check if there are noi images in the environment, if not use the maximum number
        if(noi >= noi_check):
            noi = noi_check
            print('noi is too large, all available images are used.')

        ctr = 0

        while (ctr < noi):

            ctr += 1

            with zipfile.ZipFile(path, 'r') as zip:
                file_names = zip.namelist()

                for file_name in file_names:

                    if file_name.find(env_name + str(env_idx)) != -1:

                        # Unzip novelty description file
                        if file_name.find('novelty_description.json') != -1:

                            zip.extract(file_name, root)
                            json_path = root + os.sep + file_name
                            # print(json_path)
                            find_json = True

                            with open(json_path) as json_file:
                                nov_dict = json.load(json_file)

                        # Unzip  images
                        if file_name.find('.png') != -1:

                            if file_name not in old_img:
                                old_img.append(file_name)

                                zip.extract(file_name, root)
                                find_png = True

                                # Read image, remove "Minecraft score bar", rescale
                                png_path = root + os.sep + file_name
                                image = io.imread(png_path)  # Height x Width x RGB Channels
                                patches = preprocess_image(image, self.scale_factor, self.p_size)

                                # Convert novelty description json file to encoded vector
                                nov_vector = nov_dict_to_encoded_nov(nov_dict)

                                break

            if not find_png and not find_json:
                print('-----------------------')
                print('Files not found, check if environemnt exists!')
                print('-----------------------')

            self.allpatches.append(patches)
            self.allnovbins.append(nov_vector)
            self.all_images.append(image)

    def __len__(self):

        # Total number of images
        return len(self.allpatches)

    def __getitem__(self, index):

        return(self.allpatches[index], self.allnovbins[index], self.all_images[index])

    def load_zipped_files(self):
        url_normal_data = 'https://tufts.box.com/shared/static/t5s7pss0het9p2n1wp81f1ewonyk99hm.zip'
        url_nov_item = 'https://tufts.box.com/shared/static/7dow5ah9anotzmqncw7z7ey83t09sfny.zip'
        url_nov_height = 'https://tufts.box.com/shared/static/3k85fitc1t50i8t0ez6nl5rlw807ruib.zip'
        if self.nov_type == 'normal':
            download_url(url_normal_data, self.root_normal, self.file_name_normal)
            path = self.path_normal
            root = self.root_normal
            env_name = 'env_'

        elif self.nov_type == 'novel_item':
            download_url(url_nov_item, self.root_nov, self.file_name_nov_item)
            path = self.path_nov_item
            root = self.root_nov
            env_name = 'novelty_'

        elif self.nov_type == 'novel_height':
            download_url(url_nov_height, self.root_nov, self.file_name_nov_height)
            path = self.path_nov_height
            root = self.root_nov
            env_name = 'novelty_'

        else:
            print('No valid novelty type!')

        return path, root, env_name


class PolycraftDatasetWithSpecificItem(PolycraftDatasetNoSpecificItem):

    def __init__(self, nov_type, noi, env_idx, p_size, scale_factor, item_name):

        self.nov_type = nov_type
        self.allpatches = []
        self.allnovbins = []
        self.all_images = []
        self.item_name = item_name

        self.root_normal = 'datasets' + os.sep + 'normal_data'
        self.file_name_normal = 'normal_data.zip'

        self.root_nov = 'datasets' + os.sep + 'novel_data'
        self.file_name_nov_item = 'item_novelty.zip'
        self.file_name_nov_height = 'height_novelty.zip'

        self.path_normal = self.root_normal + os.sep + self.file_name_normal
        self.path_nov_item = self.root_nov + os.sep + self.file_name_nov_item
        self.path_nov_height = self.root_nov + os.sep + self.file_name_nov_height

        self.scale_factor = scale_factor
        self.p_size = p_size

        is_item_minecraft_item = True
        is_item_present = True

        print('item name', self.item_name)

        if self.item_name is not None:
            # Let's check if the item is a valid minecraft item!
            is_item_minecraft_item = check_if_item_exists(self.item_name)

        nov_dict = {}
        old_img = []
        class_path = ''

        # Load zipped files from BOX
        path, root, env_name = self.load_zipped_files()

        ctr = 0
        env_idx = 0

        while ctr < noi and is_item_minecraft_item:

            ctr += 1

            print('ctr', ctr)

            with zipfile.ZipFile(path, 'r') as zip:
                file_names = zip.namelist()

                for file_name in file_names:

                    # if file_name.find(env_name + str(env_idx) ) != -1:

                    # Unzip novelty description file
                    if file_name.find('novelty_description.json') != -1:

                        zip.extract(file_name, root)
                        json_path = root + os.sep + file_name

                        # print(json_path)

                        with open(json_path) as json_file:
                            nov_dict = json.load(json_file)

                            if self.item_name is None:
                                print('json path ', json_path)
                                class_path = file_name

                                env_idx = [int(s) for s in re.findall(r'\d+', json_path)][0]
                                noi_check = count_png_in_env(path, env_name, env_idx)

                                if(noi >= noi_check):
                                    noi = noi_check
                                    print('noi is too large, all available images are used.')

                                break

                        if self.item_name is not None:

                            # Is the item present in the environment?
                            is_item_present = self.look_for_item_in_json(nov_dict)

                            if is_item_present:
                                class_path = file_name
                                env_idx = [int(s) for s in re.findall(r'\d+', json_path)][0]

                                noi_check = count_png_in_env(path, env_name, env_idx)

                                print('-----ITEM FOUND -----------')
                                print('noi: ', noi, ', noi check: ',
                                      noi_check, ', env with item: ', env_idx)
                                print(json_path)

                                # Check if there are noi images in the environment,
                                # if not use the maximum number
                                if(noi >= noi_check):
                                    noi = noi_check
                                    print('noi is too large, all available images are used.')

            print('Class path', class_path)
            with zipfile.ZipFile(path, 'r') as zip:
                file_names = zip.namelist()

                for file_name in file_names:

                    if file_name.find(class_path.replace('novelty_description.json', '')) != -1:

                        if file_name.find('.png') != -1:

                            if file_name not in old_img:
                                old_img.append(file_name)

                                zip.extract(file_name, root)

                                # Read image, remove "Minecraft score bar", rescale
                                png_path = root + os.sep + file_name
                                image = io.imread(png_path)  # Height x Width x RGB Channels
                                patches = preprocess_image(image, self.scale_factor, self.p_size)

                                # Convert novelty description json file to encoded vector
                                nov_vector = nov_dict_to_encoded_nov(nov_dict)

                                break

            self.allpatches.append(patches)
            self.allnovbins.append(nov_vector)
            self.all_images.append(image)

    def look_for_item_in_json(self, json):

        is_class_in_dataset = False

        # First check if we have a novelty described in the json file:
        if bool(json):

            if json['novelty'][0]['noveltyType'] == 'NewItem':

                item = json['novelty'][0]['noveltyItem']

                if item == self.item_name:
                    is_class_in_dataset = True

        return is_class_in_dataset
