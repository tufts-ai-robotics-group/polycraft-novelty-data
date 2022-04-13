# Dataset

Folders with data, divided up by generation technique. "normal" contains non-novel data.

At the moment we distinguish between two types of novelties:

### 1. Novel Items

In the standard pogo_nonov.json game environment file there are five trees and a crafting table item. We can add new items (f. e. 'minecraft:anvil', 'minecraft:bed', ...) by running the novelty_generator.py file. A more detailed description on how we can induce these new items is given in the item_novelty folder.

### 2. Given Novelties

Novelties from TA1 team: "fence" and "tree_easy".

### Folder Structure

Each folder contains a folder for each episode. Each episode is made up of:

1. Screen images (i.png) with standard resolution 256 pixel x 256 pixel of the novelty in the game environment.

2. A JSON file (i.json) with a description of the game environment, for example it states which items are present in the environment and where they are positioned. 

Each folder besides normal also contains a JSON file (novelty_description.json) with a categorization of the novelty and corresponding information about item name / position / texture details. 
