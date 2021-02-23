# Novel Data

Folders with novel data, divided up by novelty generation technique.

At the moment we distinguish between three types of novelties:

### 1. Novel items
  In the standard pogo_nonov.json game environment file there are five trees and
  a crafting table item. We can add new items (f. e. 'minecraft:anvil', 'minecraft:bed', ...)
  by running the novelty_generator.py file. A more detailed description on how we can
  induce these new items is given in the item_novelty folder. 

### 2. Novel item positions 
  In the standard game environment, items are assumed to be on the floor, they
  are positioned at "height position" z = 4. 
  We can introduce "flying items" by changing the "height position" to z > 4.
  A more detailed description on how we can change item positions is given in the height_novelty folder. 

### 3. Novel textures
  TO-DO: Use https://github.com/StephenGss/PAL/tree/master/run/resourcepacks to 
  alter textures. 

### novelty_x folder structure

Each novelty_x folder contains 

1. Screen images (screen_image_i.png) with standard resolution 256 pixel x 256 pixel  
  of the novelty in the game environment.

2. A json file (novelty_description.json) with a categorization of the novelty and corresponding
  information about item name / position / texture details. 
  
3. A json file (env_state.json) with a general description of the game environment , f. e. it states which items
  are present in the environment and where they are positioned. 
  
4. A csv file  (env_state_detailed.csv) with detailed description of the game and its enviroment, f. e. information 
  about the player, a binary respresentation of the game map, ... 
