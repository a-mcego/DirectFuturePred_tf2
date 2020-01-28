import vizdoom as vzd
from DFP_helpers import *

#Scenario: Just playing Doom 1 map E1M2.
WAD_NAME = "doom.wad"
MAP_NAME = "e1m2"
MEAS = [get_nth_var(0),get_nth_var(1),get_nth_var(2), get_damagecount, get_deadness]
VIZDOOM_VARS = [vzd.GameVariable.HEALTH, vzd.GameVariable.ARMOR, vzd.GameVariable.SECRETCOUNT]
MEAS_PREPROCESS_COEFS = [0.01,0.01,1.0,0.02,2.0]
GOAL_MEAS_COEFS = [1.,1.,1.,1.,-1.]
action_list = Actions([[vzd.Button.MOVE_LEFT,vzd.Button.MOVE_RIGHT],[vzd.Button.MOVE_FORWARD,vzd.Button.MOVE_BACKWARD],[vzd.Button.TURN_LEFT,vzd.Button.TURN_RIGHT],[vzd.Button.SPEED],[vzd.Button.ATTACK,vzd.Button.USE,vzd.Button.SELECT_WEAPON1,vzd.Button.SELECT_WEAPON2,vzd.Button.SELECT_WEAPON3,vzd.Button.SELECT_WEAPON4]])
EPISODE_LENGTH = 180*35
USE_DEPTH_BUFFER = True
USE_LABELED_RECTS = True
GOAL_TIMES = [1,2,4,8,16,32,64]
GOAL_TEMPORAL_COEFS = [0.2, 0.2, 0.2, 0.5, 0.5, 1., 1.]

GAME_RESOLUTION = (160,120)
SCALED_RESOLUTION = (84,84)
