import vizdoom as vzd
from DFP_helpers import *

#Scenario 1 from the Direct Future Prediction paper
WAD_NAME = "d1_basic.wad"
MAP_NAME = "map01"
VIZDOOM_VARS = [vzd.GameVariable.HEALTH]
MEAS = [get_nth_var(0)]
MEAS_TYPES = [MeasType.ABSOLUTE]
MEAS_PREPROCESS_COEFS = [0.01]
GOAL_MEAS_COEFS = [1.]
action_list = Actions([[vzd.Button.MOVE_FORWARD],[vzd.Button.TURN_LEFT],[vzd.Button.TURN_RIGHT]])
EPISODE_LENGTH = 60*35
USE_DEPTH_BUFFER = False
USE_LABELED_RECTS = False
GOAL_TIMES = [1,2,4,8,16,32]
GOAL_TEMPORAL_COEFS = [0., 0., 0., 0.5, 0.5, 1.0]

GAME_RESOLUTION = (160,120)
SCALED_RESOLUTION = (84,84)
COLORMODE = ColorMode.GRAYSCALE
