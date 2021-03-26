# -*- coding: utf-8 -*-
"""
Created on Thu May 21 17:42:15 2020

@author: Sarah
"""
import torch
from torch.utils.data import Dataset
import os
from skimage import io
from torchvision.datasets.utils import download_url
import zipfile
import json
from skimage.transform import rescale
import re


def normalize_img(img):
    minval = img.min()
    maxval = img.max()

    diff = maxval - minval

    if diff > 0:
        img_norm = (img - minval) / diff
    else:
        img_norm = torch.zeros(img.size())

    return img_norm


def convert_item_to_encoded_item(item):

    encoded_item = []

    minecraft_item_names = ['minecraft:anvil', 'minecraft:beacon', 'minecraft:bed', 'minecraft:bedrock',
                            'minecraft:bookshelf', 'minecraft:brewing_stand', 'minecraft:cactus', 'minecraft:cake',
                            'minecraft:cauldron', 'minecraft:clay', 'minecraft:coal_block', 'minecraft:cobblestone',
                            'minecraft:crafting_table', 'minecraft:daylight_detector', 'minecraft:deadbush',
                            'minecraft:diamond_block', 'minecraft:dirt', 'minecraft:dispenser', 'minecraft:dropper',
                            'minecraft:emerald_block', 'minecraft:enchanting_table', 'minecraft:glowstone',
                            'minecraft:gravel', 'minecraft:hopper', 'minecraft:ice', 'minecraft:iron_block',
                            'minecraft:jukebox', 'minecraft:lapis_block', 'minecraft:leaves', 'minecraft:lever',
                            'minecraft:log', 'minecraft:mycelium', 'minecraft:netherrack', 'minecraft:noteblock',
                            'minecraft:obsidian', 'minecraft:piston', 'minecraft:planks', 'minecraft:prismarine',
                            'minecraft:pumpkin', 'minecraft:quartz_block', 'minecraft:reeds', 'minecraft:sand',
                            'minecraft:sandstone', 'minecraft:sapling', 'minecraft:sea_lantern', 'minecraft:slime',
                            'minecraft:snow', 'minecraft:soul_sand', 'minecraft:sponge', 'minecraft:stone',
                            'minecraft:stonebrick', 'minecraft:tallgrass', 'minecraft:tnt', 'minecraft:torch',
                            'minecraft:vine', 'minecraft:waterlily', 'minecraft:web', 'minecraft:wheat',
                            'minecraft:wool']

    for item_name in minecraft_item_names:
        if item_name == item:
            encoded_item.append(1)
        else:
            encoded_item.append(0)

    return encoded_item


def check_if_item_exists(name):

    item_exists = False
    minecraft_item_names = ['minecraft:anvil', 'minecraft:beacon', 'minecraft:bed', 'minecraft:bedrock',
                            'minecraft:bookshelf', 'minecraft:brewing_stand', 'minecraft:cactus', 'minecraft:cake',
                            'minecraft:cauldron', 'minecraft:clay', 'minecraft:coal_block', 'minecraft:cobblestone',
                            'minecraft:crafting_table', 'minecraft:daylight_detector', 'minecraft:deadbush',
                            'minecraft:diamond_block', 'minecraft:dirt', 'minecraft:dispenser', 'minecraft:dropper',
                            'minecraft:emerald_block', 'minecraft:enchanting_table', 'minecraft:glowstone',
                            'minecraft:gravel', 'minecraft:hopper', 'minecraft:ice', 'minecraft:iron_block',
                            'minecraft:jukebox', 'minecraft:lapis_block', 'minecraft:leaves', 'minecraft:lever',
                            'minecraft:log', 'minecraft:mycelium', 'minecraft:netherrack', 'minecraft:noteblock',
                            'minecraft:obsidian', 'minecraft:piston', 'minecraft:planks', 'minecraft:prismarine',
                            'minecraft:pumpkin', 'minecraft:quartz_block', 'minecraft:reeds', 'minecraft:sand',
                            'minecraft:sandstone', 'minecraft:sapling', 'minecraft:sea_lantern', 'minecraft:slime',
                            'minecraft:snow', 'minecraft:soul_sand', 'minecraft:sponge', 'minecraft:stone',
                            'minecraft:stonebrick', 'minecraft:tallgrass', 'minecraft:tnt', 'minecraft:torch',
                            'minecraft:vine', 'minecraft:waterlily', 'minecraft:web', 'minecraft:wheat',
                            'minecraft:wool']

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

    with zipfile.ZipFile(path, 'r') as zipObj:
        listOfFileNames = zipObj.namelist()

        for fileName in listOfFileNames:

            # We need the path separation symbol in order to not count
            # environemnts which only contain the env_idx (4 --> 41)
            if fileName.find(env_name + str(env_idx) + '/') != -1 or \
                    fileName.find(env_name + str(env_idx) + '\\') != -1:

                if fileName.find('.png') != -1:

                    img_ctr += 1

    return img_ctr


class PolycraftDataset(Dataset):

    def __init__(self, nov_type, noi, env_idx, p_size, scale_factor, item_name):

        self.nov_type = nov_type
        self.allpatches = []
        self.allnovbins = []
        self.all_images = []
        self.item_name = item_name

        self.root_normal = 'datasets' + os.sep + 'normal_data'
        self.filename_normal = 'normal_data.zip'

        self.root_nov = 'datasets' + os.sep + 'novel_data'
        self.filename_nov_item = 'item_novelty.zip'
        self.filename_nov_height = 'height_novelty.zip'

        self.path_normal = self.root_normal + os.sep + self.filename_normal
        self.path_nov_item = self.root_nov + os.sep + self.filename_nov_item
        self.path_nov_height = self.root_nov + os.sep + self.filename_nov_height

        self.scale_factor = scale_factor
        self.p_size = p_size

        is_item_minecraft_item = True
        is_item_present = True

        print('item name', self.item_name)

        if self.item_name != None:
            # Let's check if the item is a valid minecraft item!
            is_item_minecraft_item = check_if_item_exists(self.item_name)

        nov_dict = {}
        nov_dict_item = {}
        old_img = []
        class_path = ''

        # Load zipped files from BOX
        path, root, env_name = self.load_zipped_files()

        ctr = 0
        env_idx = 0

        while ctr < noi and is_item_minecraft_item:

            ctr += 1

            print('ctr', ctr)

            with zipfile.ZipFile(path, 'r') as zipObj:
                listOfFileNames = zipObj.namelist()

                for fileName in listOfFileNames:

                    # if fileName.find(env_name + str(env_idx) ) != -1:

                    # Unzip novelty description file
                    if fileName.find('novelty_description.json') != -1:

                        zipObj.extract(fileName, root)
                        json_path = root + os.sep + fileName

                        # print(json_path)
                        find_json = True

                        with open(json_path) as json_file:
                            nov_dict = json.load(json_file)

                            if self.item_name == None:
                                print('json path ', json_path)
                                class_path = fileName
                                nov_dict_item = nov_dict

                                env_idx = [int(s) for s in re.findall(r'\d+', json_path)][0]
                                noi_check = count_png_in_env(path, env_name, env_idx)

                                if(noi >= noi_check):
                                    noi = noi_check
                                    print('noi and/or noe too large, all available images are used.')

                                break

                        if self.item_name != None:

                            # Is the item present in the environment?
                            is_item_present = self.look_for_item_in_json(nov_dict)

                            if is_item_present:
                                class_path = fileName
                                env_idx = [int(s) for s in re.findall(r'\d+', json_path)][0]
                                nov_dict_item = nov_dict

                                noi_check = count_png_in_env(path, env_name, env_idx)

                                print('-----ITEM FOUND -----------')
                                print('noi: ', noi, ', noi check: ',
                                      noi_check, ', env with item: ', env_idx)
                                print(json_path)

                                # Check if there are noi images in the environment, if not use the maximum number
                                if(noi >= noi_check):
                                    noi = noi_check
                                    print('noi and/or noe too large, all available images are used.')

            print('Class path', class_path)
            with zipfile.ZipFile(path, 'r') as zipObj:
                listOfFileNames = zipObj.namelist()

                for fileName in listOfFileNames:

                    if fileName.find(class_path.replace('novelty_description.json', '')) != -1:

                        if fileName.find('.png') != -1:

                            if not fileName in old_img:
                                old_img.append(fileName)

                                zipObj.extract(fileName, root)
                                find_png = True

                                # Read image, remove "Minecraft score bar", rescale, normalize between 0 and 1
                                png_path = root + os.sep + fileName
                                image = io.imread(png_path)  # Height x Width x RGB Channels
                                print('png_path', png_path)
                                print('--------------------------------------')

                                image = self.crop_and_normalize(image)

                                # Extract p_size x p_size patches, patches overlap each other by half
                                patches = self.extract_patches(image)

                                # Convert novelty description json file to encoded novelty description vector
                                nov_vector = nov_dict_to_encoded_nov(nov_dict_item)

                                break

            self.allpatches.append(patches)
            self.allnovbins.append(nov_vector)
            self.all_images.append(image)

    def __len__(self):

        # Total number of images
        return len(self.allpatches)

    def __getitem__(self, index):

        return(self.allpatches[index], self.allnovbins[index], self.all_images[index])

    def load_zipped_files(self):
        if self.nov_type == 'normal':

            url_normal_data = 'https://tufts.box.com/shared/static/t5s7pss0het9p2n1wp81f1ewonyk99hm.zip'
            download_url(url_normal_data, self.root_normal, self.filename_normal)
            path = self.path_normal
            root = self.root_normal
            env_name = 'env_'

        elif self.nov_type == 'novel_item':
            url_nov_item = 'https://tufts.box.com/shared/static/7dow5ah9anotzmqncw7z7ey83t09sfny.zip'
            download_url(url_nov_item, self.root_nov, self.filename_nov_item)
            path = self.path_nov_item
            root = self.root_nov
            env_name = 'novelty_'

        elif self.nov_type == 'novel_height':

            url_nov_height = 'https://tufts.box.com/shared/static/3k85fitc1t50i8t0ez6nl5rlw807ruib.zip'
            download_url(url_nov_height, self.root_nov, self.filename_nov_height)
            path = self.path_nov_height
            root = self.root_nov
            env_name = 'novelty_'

        else:
            print('No valid novelty type!')

        return path, root, env_name

    def crop_and_normalize(self, image):

        image = image[0:234, :, :]
        image = rescale(image, (self.scale_factor, self.scale_factor, 1), anti_aliasing=True)
        image = normalize_img(image)

        return image

    def extract_patches(self, image):

        # Extract patches
        stride = int(self.p_size/2)  # patch stride
        image = torch.from_numpy(image)
        patches = image.unfold(0, self.p_size, stride).unfold(1, self.p_size, stride)

        return patches

    def look_for_item_in_json(self, json):

        is_class_in_dataset = False

        # First check if we have a novelty described in the json file:
        if bool(json):

            if json['novelty'][0]['noveltyType'] == 'NewItem':

                item = json['novelty'][0]['noveltyItem']

                if item == self.item_name:
                    is_class_in_dataset = True

        return is_class_in_dataset