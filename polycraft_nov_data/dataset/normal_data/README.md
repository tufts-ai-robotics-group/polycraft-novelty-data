# Normal Data

Data from a normal environment, without novelties.

## Steps to Reproduce

1. Launch Polycraft (For example run PAL/PolycraftAIGym/LaunchTournament.py)

2. Set induce_novelty = False in line 380 in  polycraft_tufts/novelty_generator/novelty_generator.py file.
   No novelty will be induced, however, we run this file to get the env_state.json and env_state_detailed.csv
   file. 
  
3. Run polycraft_tufts/utils/save_screen.py 
    At the moment the agent is positioned at random locations and screen images are generated. 
 
4. Run polycraft_tufts/utils/keyboard_interface.py (you might need to change the port number: PORT = 9001) 
    and press ESC.
    
 A dataset folder with json-, csv- and png files is generated in the polycraft_tufts folder. Every image can
 be kept because they all show the "normal environment". 
