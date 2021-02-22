# Dataset Generation Description

Substitute these novelty_generator.py and polycraft_interface.py with the corresponding files in the
polycraft_tufts respository in order to generate dataset images. 

### General procedure 

1. Launch Polycraft (For example run PAL/PolycraftAIGym/LaunchTournament.py)

2. Run polycraft_tufts/novelty_generator/novelty_generator.py with arguments depending on the novelty
   you want to have (See polycraft-novelty-data/polycraft_nov_data/novel_data folders for novelty
   specific details)
  
3. Run polycraft_tufts/utils/save_screen.py 
    At the moment the agent is positioned at random locations and screen images are generated. 
 
4. Run polycraft_tufts/utils/keyboard_interface.py (you might need to change the port number: PORT = 9001) 
    and press ESC.
 
A dataset folder with json-, csv- and png files is generated in the polycraft_tufts folder.
 
 
