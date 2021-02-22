# New Item

Data that contains novel items.

## Steps to Reproduce

1. Launch Polycraft (For example run PAL/PolycraftAIGym/LaunchTournament.py)

2. Set induce_novelty = True in line 380 in polycraft_tufts/novelty_generator/novelty_generator.py file. 
   If you want to have height / position change AND a novel item, set pos_change = True (line 209). For
   a novel item without any change in item positions set pos_change = False.
   
   Run polycraft_tufts/novelty_generator/novelty_generator.py with arguments -novelty lvl-1 -parameter x. 

3. Run polycraft_tufts/utils/save_screen.py 

4. Run polycraft_tufts/utils/keyboard_interface.py (you might need to change the port number: PORT = 9001) and press ESC.

A dataset folder with json-, csv- and png files is generated in the polycraft_tufts folder. 
At the moment we have to check the png files and choose the ones which show the novelty, because the save_screen's agent
is positioned at random locations. 
