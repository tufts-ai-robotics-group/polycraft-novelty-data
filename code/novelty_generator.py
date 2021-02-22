# Author: Gyan Tatiya
# Email: Gyan.Tatiya@tufts.edu

import argparse
import ast
import csv
import json
import os
import random
import shutil
import sys
from datetime import datetime

import numpy as np
import matplotlib.pyplot as plt

sys.path.append('../utils')
from utils import get_paths, get_middle_sockets, recv_socket_data


def get_random_command(nav_command, nav_commands_list, prob):
    if np.random.binomial(n=1, p=prob) == 1:
        pass
    else:
        nav_commands_list = nav_commands_list[nav_commands_list != nav_command]
        nav_command = np.random.choice(nav_commands_list, size=1)[0]

    if args['print_output']:
        print("Command2: ", nav_command)

    return nav_command


def get_novel_map(sense_all_nonav, novelty_type):
    """
    Create a new map based on novelty_type
    novelty_type: 90, 180, 270, "vertical", "horizontal"
    90, 180, 270 will rotate the map
    "vertical", "horizontal" will mirror the map
    """

    # Finding x_max, y_max and items_id from SENSE_ALL NONAV
    x_max, y_max = 0, 0
    items_list = []
    for a_xzy, item in sense_all_nonav['map'].items():
        items_list.append(item['name'])
        x, y = int(a_xzy.split(',')[0]), int(a_xzy.split(',')[2])
        if x > x_max:
            x_max = x
        if y > y_max:
            y_max = y
    x_max += 1
    y_max += 1

    items_id = {}
    for item in sorted(set(items_list)):
        items_id.setdefault(item, len(items_id) + 1)

    # Filling a 2D list to plot as map
    map_to_plot = np.zeros((y_max, x_max))  # Y (row) is before X (column) in matrix

    for a_xzy in sense_all_nonav['map']:
        x, y = int(a_xzy.split(',')[0]), int(a_xzy.split(',')[2])
        map_to_plot[y][x] = items_id[sense_all_nonav['map'][a_xzy]['name']]

    if args['print_output']:
        visualize_env_v2(map_to_plot, sense_all_nonav['player'], novelty_type)

    sense_all_nonav_map = {}
    for x in range(x_max):
        for y in range(y_max):
            if novelty_type == 90:
                # 90 degree rotation
                sense_all_nonav_map[str(y) + ',4,' + str(x)] = sense_all_nonav['map'][
                    str(x_max - x - 1) + ',4,' + str(y)]
            elif novelty_type == 180:
                # 180 degree rotation
                sense_all_nonav_map[str(x) + ',4,' + str(y)] = sense_all_nonav['map'][
                    str(x_max - x - 1) + ',4,' + str(y_max - y - 1)]
            elif novelty_type == 270:
                # 270 degree rotation
                sense_all_nonav_map[str(y) + ',4,' + str(x)] = sense_all_nonav['map'][
                    str(x) + ',4,' + str(y_max - y - 1)]
            elif novelty_type == 'vertical':
                sense_all_nonav_map[str(x) + ',4,' + str(y)] = sense_all_nonav['map'][
                    str(x_max - x - 1) + ',4,' + str(y)]
            elif novelty_type == 'horizontal':
                sense_all_nonav_map[str(x) + ',4,' + str(y)] = sense_all_nonav['map'][
                    str(x) + ',4,' + str(y_max - y - 1)]
    sense_all_nonav['map'] = sense_all_nonav_map

    x, z, y = sense_all_nonav['player']['pos']
    if novelty_type == 90:
        # 90 degree rotation
        sense_all_nonav['player']['pos'] = [y, z, x_max - x - 1]
        if sense_all_nonav['player']['facing'] == 'NORTH':
            sense_all_nonav['player']['facing'] = 'WEST'
        elif sense_all_nonav['player']['facing'] == 'SOUTH':
            sense_all_nonav['player']['facing'] = 'EAST'
        elif sense_all_nonav['player']['facing'] == 'WEST':
            sense_all_nonav['player']['facing'] = 'SOUTH'
        elif sense_all_nonav['player']['facing'] == 'EAST':
            sense_all_nonav['player']['facing'] = 'NORTH'
    elif novelty_type == 180:
        # 180 degree rotation
        sense_all_nonav['player']['pos'] = [x_max - x - 1, z, y_max - y - 1]
        if sense_all_nonav['player']['facing'] == 'NORTH':
            sense_all_nonav['player']['facing'] = 'SOUTH'
        elif sense_all_nonav['player']['facing'] == 'SOUTH':
            sense_all_nonav['player']['facing'] = 'NORTH'
        elif sense_all_nonav['player']['facing'] == 'WEST':
            sense_all_nonav['player']['facing'] = 'EAST'
        elif sense_all_nonav['player']['facing'] == 'EAST':
            sense_all_nonav['player']['facing'] = 'WEST'
    elif novelty_type == 270:
        # 270 degree rotation
        sense_all_nonav['player']['pos'] = [y_max - y - 1, z, x]
        if sense_all_nonav['player']['facing'] == 'NORTH':
            sense_all_nonav['player']['facing'] = 'EAST'
        elif sense_all_nonav['player']['facing'] == 'SOUTH':
            sense_all_nonav['player']['facing'] = 'WEST'
        elif sense_all_nonav['player']['facing'] == 'WEST':
            sense_all_nonav['player']['facing'] = 'NORTH'
        elif sense_all_nonav['player']['facing'] == 'EAST':
            sense_all_nonav['player']['facing'] = 'SOUTH'
    elif novelty_type == 'vertical':
        sense_all_nonav['player']['pos'] = [x_max - x - 1, z, y]
        if sense_all_nonav['player']['facing'] == 'WEST':
            sense_all_nonav['player']['facing'] = 'EAST'
        elif sense_all_nonav['player']['facing'] == 'EAST':
            sense_all_nonav['player']['facing'] = 'WEST'
    elif novelty_type == 'horizontal':
        sense_all_nonav['player']['pos'] = [x, z, y_max - y - 1]
        if sense_all_nonav['player']['facing'] == 'NORTH':
            sense_all_nonav['player']['facing'] = 'SOUTH'
        elif sense_all_nonav['player']['facing'] == 'SOUTH':
            sense_all_nonav['player']['facing'] = 'NORTH'

    return sense_all_nonav


def get_novel_name(sense_all_nonav, original_novel_item_name):
    """
    Change item names
    """

    for location in sense_all_nonav['map']:
        for orininal_name in original_novel_item_name:
            if sense_all_nonav['map'][location]['name'] == orininal_name:
                sense_all_nonav['map'][location]['name'] = original_novel_item_name[orininal_name]

    return sense_all_nonav


def visualize_env_v2(map_to_plot, player, novelty_type):
    x, z, y = player['pos']

    x2, y2 = 0, 0
    if player['facing'] == 'NORTH':
        x2, y2 = 0, -0.01
    elif player['facing'] == 'SOUTH':
        x2, y2 = 0, 0.01
    elif player['facing'] == 'WEST':
        x2, y2 = -0.01, 0
    elif player['facing'] == 'EAST':
        x2, y2 = 0.01, 0

    plt.figure("Original map. Novelty: " + str(novelty_type), figsize=[5, 4.5])
    plt.imshow(map_to_plot, cMAP="gist_ncar")
    plt.arrow(x, y, x2, y2, head_width=0.5, head_length=0.5, color='black')
    plt.title('NORTH\n' + 'Agent is facing ' + player['facing'])
    plt.xlabel('SOUTH')
    plt.ylabel('WEST')
    plt.colorbar()
    plt.pause(0.01)
    plt.clf()


def generate_lvl0(task, novelty_description_dict, backup_json=True ):
    _, _, POLYCRAFT_PATH = get_paths(os.getcwd(), 'PAL')
    task_json_path = POLYCRAFT_PATH + os.sep + 'available_tests' + os.sep + task
    task_json = json.loads(open(task_json_path, 'r').read())

    # Finding map size
    x_max, z_max, y_max = task_json['features'][2]['pos2']

    """
    # changing agent facing direction
    a_dir = np.random.choice(4, size=1)[0]
    if a_dir == 0:
        task_json['features'][0]['lookDir'] = [0, 0, 0]  # South
    elif a_dir == 1:
        task_json['features'][0]['lookDir'] = [0, 90, 0]  # West
    elif a_dir == 2:
        task_json['features'][0]['lookDir'] = [0, 180, 0]  # North
    elif a_dir == 3:
        task_json['features'][0]['lookDir'] = [0, -90, 0]  # East

    # changing agent location
    all_blockList = []
    x_new = int(np.random.choice(range(2, x_max - 1), size=1)[0])
    y_new = int(np.random.choice(range(2, y_max - 1), size=1)[0])
    all_blockList.append([x_new, task_json['features'][0]['pos'][1], y_new])
    task_json['features'][0]['pos'][0], task_json['features'][0]['pos'][2] = x_new, y_new
    task_json['features'][0]['pos2'][0], task_json['features'][0]['pos2'][2] = x_new, y_new
    """
    
    logic_json_path = POLYCRAFT_TUFTS_PATH + os.sep + "datasets" + os.sep + 'env_state.json'
    pos_change = True #Do we want a repositioning of Objects? (Necessary if we only want ONLY lvl-1 novelty.)
    num_of_pos_change = 1 # How many objects do we want to reposition?
    
    shuffled_blocks = (task_json['features'][1]['blockList'])
    random.shuffle(shuffled_blocks)
    count = 0
    all_blockList = []
    
    if pos_change:

        # changing block location
        for block in shuffled_blocks:
            while True:
               
                x_old = block['blockPos'][0]
                y_old = block['blockPos'][2]
                z_old = block['blockPos'][1]
                item_name = block['blockName']
            
                #x_new = int(np.random.choice(range(2, x_max - 1), size=1)[0])
                #y_new = int(np.random.choice(range(2, y_max - 1), size=1)[0])
                x_new = x_old
                y_new = y_old
                
                # z = 4 --> block is on the floor, we want it to fly --> z > 4
                z_new = int(np.random.choice(range(6, 18), size=1)[0]) 
                
                # making sure each blockPos is unique
                if [x_new, z_new, y_new] not in all_blockList:
                    
             
                    count += 1
                    all_blockList.append([x_new, z_new, y_new])
                    block['blockPos'][0], block['blockPos'][1], block['blockPos'][2] = x_new, z_new, y_new
                    novelty_description_dict['novelty'].append({'noveltyType': 'newPosition', 'newPosItem': item_name, \
                    'oldCoord': [x_old, z_old, y_old], 'newCoord': [x_new, z_new, y_new] })
                    break
                    
            if count == num_of_pos_change:
                break
                
            

    if backup_json:
        # Making a backup of original task and saving new task with same name
        task_json_path_new, ext = os.path.splitext(task_json_path)
        task_json_path_new += "_ORIGINAL"
        task_json_path_new += ext
        if not os.path.isfile(task_json_path_new):
            shutil.copy(task_json_path, task_json_path_new)
        with open(task_json_path, 'w') as outfile:
            json.dump(task_json, outfile)
        with open(logic_json_path, 'w') as outfile:
            json.dump(task_json, outfile)
        return all_blockList, novelty_description_dict
    else:
        return all_blockList, task_json, novelty_description_dict


def generate_lvl1(task, num_of_items, novelty_description_dict ):

    _, _, POLYCRAFT_PATH = get_paths(os.getcwd(), 'PAL')
    task_json_path = POLYCRAFT_PATH + os.sep + 'available_tests' + os.sep + task
   
    logic_json_path = POLYCRAFT_TUFTS_PATH + os.sep + "datasets" + os.sep + 'env_state.json'

    task_json_path_new, ext = os.path.splitext(task_json_path)
    task_json_path_new += "_ORIGINAL"
    task_json_path_new += ext
    print(task_json_path)
    
    
    if os.path.isfile(task_json_path_new):
        task_new, ext = os.path.splitext(task)
        all_blockList, task_json, novelty_description_dict = generate_lvl0(task_new+"_ORIGINAL"+ext, novelty_description_dict, backup_json=False)
    else:
        all_blockList, novelty_description_dict = generate_lvl0(task, novelty_description_dict)
        task_json = json.loads(open(task_json_path, 'r').read())
    
   
    # Finding map size
    x_max, z_max, y_max = task_json['features'][2]['pos2']

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
                             

    random.shuffle(minecraft_item_names)
    
    
    count = 0
    for item_name in minecraft_item_names:
        while True:
            x_new = int(np.random.choice(range(2, x_max - 1), size=1)[0])
            y_new = int(np.random.choice(range(2, y_max - 1), size=1)[0])

            # making sure each blockPos is unique
            if [x_new, z_max, y_new] not in all_blockList:
                all_blockList.append([x_new, z_max, y_new])
                count += 1
                task_json['features'][1]['blockList'].append(
                    {'blockPos': [x_new, z_max, y_new], 'blockName': item_name})
                novelty_description_dict['novelty'].append({'noveltyType': 'NewItem', 'noveltyItem': item_name, 'noveltyPosition': [x_new, z_max, y_new]})
               
                break

        if count == num_of_items:
            break
            
  
 
    with open(task_json_path, 'w') as outfile:
        json.dump(task_json, outfile)
        
    with open(logic_json_path, 'w') as outfile:
        json.dump(task_json, outfile)

    return all_blockList, novelty_description_dict


if __name__ == "__main__":
    # Global Variables
    HOST = '127.0.0.1'
    POLYCRAFT_PORT, AGENT_PORT = 9000, 9001

    SOCK_POLY, SOCK_AGENT = get_middle_sockets(HOST, POLYCRAFT_PORT, AGENT_PORT)

    ap = argparse.ArgumentParser()
    ap.add_argument("-novelty", default='navigation', required=True, help="type of novelty")
    ap.add_argument("-parameter", help="parameter")
    ap.add_argument("-parameter2", help="parameter 2")
    ap.add_argument("-print_output", default="", help="print stuff")
    ap.add_argument("-save_json", default="", help="save json")
    ap.add_argument("-save_sense_screen", default="", help="save sense screen")
    args = vars(ap.parse_args())
    if args['print_output']:
        print("args: ", args)

    
    if args['save_sense_screen']:
        CSV_FILENAME = "novelty-" + args['novelty'] + "_" + datetime.now().strftime("%Y-%m-%d-%H-%M-%S") + "_SENSE-SCREEN.csv"
    else:
        CSV_FILENAME = "novelty-" + args['novelty'] + "_" + datetime.now().strftime("%Y-%m-%d-%H-%M-%S") + ".csv"
    
    # _, POLYCRAFT_TUFTS_PATH, _ = get_paths(os.getcwd(), 'polycraft')
    _, POLYCRAFT_TUFTS_PATH, _ = get_paths(os.getcwd(), 'PAL')
    os.makedirs(POLYCRAFT_TUFTS_PATH + os.sep + "datasets", exist_ok=True)
    CSV_FILEPATH = POLYCRAFT_TUFTS_PATH + os.sep + "datasets" + os.sep + 'env_state_detailed_2.csv'
    with open(CSV_FILEPATH, 'w') as f:
        writer = csv.writer(f, lineterminator="\n")
        writer.writerow(["Command", "JSON"])
           
    #Write json file with novelty descriptions like novelty type, novelty name, position, ...
    novelty_json_path = POLYCRAFT_TUFTS_PATH + os.sep + "datasets" + os.sep + 'novelty_description.json'
    novelty_description_dict = {}
    novelty_description_dict['novelty'] = []
    
    induce_novelty = True #I run this file also if I do not introduce a novelty in order to get the same json/csv files. 
    
    RL_NAV_COMMANDS = np.array(['MOVE_NORTH', 'MOVE_SOUTH', 'MOVE_EAST', 'MOVE_WEST'])  # RL agent
    PLAN_NAV_COMMANDS = np.array(['SMOOTH_MOVE W', 'SMOOTH_MOVE X', 'SMOOTH_MOVE A', 'SMOOTH_MOVE D'])  # Planning agent

    while True:
        command = SOCK_AGENT.recv(1024).decode().strip()  # get command from agent
        if args['print_output']:
            print("Command: ", command)

        # Modify command

        # Novelty in navigation commands
        # 1. Pick random action some times
        if args['novelty'] == 'navigation_random':
            # if agent sends a navigation command
            # 80% times send the exact command to polycraft, 20% times send some other random navigation command
            if args['parameter'] == 'rl_agent' and command in RL_NAV_COMMANDS:
                command = get_random_command(command, RL_NAV_COMMANDS, float(args['parameter2']))
            elif args['parameter'] == 'planning_agent' and command in PLAN_NAV_COMMANDS:
                command = get_random_command(command, PLAN_NAV_COMMANDS, float(args['parameter2']))

        # 2. Remap action
        elif args['novelty'] == 'navigation_remapping':
            if args['parameter'] == 'rl_agent' and command in RL_NAV_COMMANDS:
                if command == 'MOVE_NORTH':
                    command = 'MOVE_SOUTH'
                elif command == 'MOVE_SOUTH':
                    command = 'MOVE_NORTH'
                elif command == 'MOVE_WEST':
                    command = 'MOVE_EAST'
                elif command == 'MOVE_EAST':
                    command = 'MOVE_WEST'
                if args['print_output']:
                    print("Command2: ", command)
            elif args['parameter'] == 'planning_agent' and command in PLAN_NAV_COMMANDS:
                if command == 'SMOOTH_MOVE W':
                    command = 'SMOOTH_MOVE X'
                elif command == 'SMOOTH_MOVE X':
                    command = 'SMOOTH_MOVE W'
                elif command == 'SMOOTH_MOVE A':
                    command = 'SMOOTH_MOVE D'
                elif command == 'SMOOTH_MOVE D':
                    command = 'SMOOTH_MOVE A'
                if args['print_output']:
                    print("Command2: ", command)

        # 3. Level 0 Novelty
        elif args['novelty'] == 'lvl-0':
            # Changing blocks' location and agent's direction, location
            if command.startswith('RESET'):
                task_filename = command.split(" ")[2].split('/')[-1]
                
                if induce_novelty:
                    generate_lvl0(task_filename, novelty_description_dict)

        # 4. Level 1 Novelty
        elif args['novelty'] == 'lvl-1':
            # Placing new items in the environment. It includes Level 0 novelty. Maximum new items can be 59
            if command.startswith('RESET'):
                task_filename = command.split(" ")[2].split('/')[-1]
                
                if induce_novelty:
                    generate_lvl1(task_filename, int(args['parameter']), novelty_description_dict)

        SOCK_POLY.send(str.encode(command + '\n'))  # send command to polycraft

        output = recv_socket_data(SOCK_POLY)  # get JSON from polycraft
        if args['print_output']:
            # print("JSON: ", json.loads(output))
            pass
            
        
        with open(novelty_json_path, 'w') as novfile:
            json.dump(novelty_description_dict, novfile)
        
        # Modify JSON

        # 1. Novelty in map
        if args['novelty'] == 'map':
            if command == 'SENSE_ALL NONAV':
                if args['parameter'] == 'rotate':
                    # Rotate degree: 90, 180, 270
                    output = get_novel_map(json.loads(output), int(args['parameter2']))
                elif args['parameter'] == 'mirror':
                    # Mirror: 'vertical', 'horizontal'
                    output = get_novel_map(json.loads(output), args['parameter2'])
                output = str.encode(json.dumps(output))  # convert to string (dumps), then bytes (encode)

        # 2. Novelty in item name
        elif args['novelty'] == 'name':
            original_novel_name = ast.literal_eval(args['parameter'])  # {'minecraft:crafting_table': 'minecraft:table'}
            if command == 'SENSE_ALL NONAV':
                output = get_novel_name(json.loads(output), original_novel_name)
                output = str.encode(json.dumps(output))  # convert to string (dumps), then bytes (encode)

        SOCK_AGENT.sendall(output)  # send JSON to the agent

       
        with open(CSV_FILEPATH, 'a') as f:  # append to the file created
            writer = csv.writer(f, lineterminator="\n")
            writer.writerow([command, output])
                
      
        if  command not in ['START'] and not command.startswith('RESET'):
            command = "SENSE_SCREEN"
            if args['print_output']:
                print("Command: ", command)
            SOCK_POLY.send(str.encode(command + '\n'))  # send command to polycraft
            output = recv_socket_data(SOCK_POLY)  # get JSON from polycraft

            output = json.loads(output)
                
            with open(CSV_FILEPATH, 'a') as f:  # append to the file created
                writer = csv.writer(f, lineterminator="\n")
                writer.writerow([command, output])
                
             
       