# Height novelty

Normally all items are positioned at height z = 4. If we set it to a number > 4, we can create "flying" objects as
a new novelty. 


## Steps to Reproduce

1. Launch Polycraft (For example run PAL/PolycraftAIGym/LaunchTournament.py)

2. Set induce_novelty = True in line 380 in polycraft_tufts/novelty_generator/novelty_generator.py file. 
   Set pos_change = True (line 209). 
   
   Run polycraft_tufts/novelty_generator/novelty_generator.py with arguments -novelty lvl-0.
   
3. Run polycraft_tufts/utils/save_screen.py 

4. Run polycraft_tufts/utils/keyboard_interface.py (you might need to change the port number: PORT = 9001) and press ESC.

A dataset folder with json-, csv- and png files is generated in the polycraft_tufts folder. 
At the moment we have to check the png files and choose the ones which show the novelty, because the save_screen's agent
is positioned at random locations. 

