from __future__ import print_function
import vizdoom as vzd
from matplotlib import pyplot as plt
import tensorflow as tf
from random import choice
import time
import numpy as np
import sys
import cv2
import threading
import concurrent.futures

if len(sys.argv) != 2:
    print("Give logfile name as argument.")
    exit(0)

from DFP_helpers import *

#---START---USER-SUPPLIED-SETTINGS---

#these are scenarios, pick one.
from d1_basic import *
#from playdoom import *

#with this, the network only predicts how much the measurements have changed
#and not the measurements themselves
PREDICT_ONLY_DELTAS = True

#RANDOM_CHOICE_TYPE = RandomChoiceType.UNIFORM
RANDOM_CHOICE_TYPE = RandomChoiceType.SOFTMAX

TRAINING_TYPE = TrainingType.EXPERIENCE_REPLAY
#TRAINING_TYPE = TrainingType.FULL_EPISODES
#TRAINING_TYPE = TrainingType.FULL_EPISODES_RECURRENT

FRAMESKIP = 4
BATCH_SIZE = 64 #how many steps in one batch of training the network
MEMORY_SIZE = 20000 #how many steps we keep in the experience memory
TEST_FREQUENCY = 2500 #how often we test the network, in steps trained

#MEMORY_FULL_STRATEGY = MemoryFullStrategy.DELETE_OLDEST
MEMORY_FULL_STRATEGY = MemoryFullStrategy.DELETE_EVERY_OTHER

#having goal as an input to the network seems to be unnecessary.
#here's a way to turn it off! :)
USE_GOAL_INPUT = False

#shall we input the last action made to the new timestep?
USE_LAST_ACTION_INPUT = True

#how many ViZDoom instances we learn with?
N_VIZDOOM_INSTANCES = 4

#---END---USER-SUPPLIED-SETTINGS---

N_GOAL_TIMES = len(GOAL_TIMES)
MEAS_POSTPROCESS_COEFS = [1.0/x for x in MEAS_PREPROCESS_COEFS]
N_MEASUREMENTS = len(MEAS)
N_ACTIONS = action_list.total
assert(len(GOAL_TIMES) == len(GOAL_TEMPORAL_COEFS))
assert(len(GOAL_MEAS_COEFS) == len(MEAS))
assert(len(MEAS_PREPROCESS_COEFS) == len(MEAS))
assert(len(MEAS_POSTPROCESS_COEFS) == len(MEAS))

GOAL_TIMES = tf.convert_to_tensor(GOAL_TIMES,dtype=tf.int32)
GOAL_TEMPORAL_COEFS = tf.convert_to_tensor(GOAL_TEMPORAL_COEFS,dtype=tf.float32)
GOAL_MEAS_COEFS = tf.convert_to_tensor(GOAL_MEAS_COEFS,dtype=tf.float32)
MEAS_PREPROCESS_COEFS = tf.convert_to_tensor(MEAS_PREPROCESS_COEFS,dtype=tf.float32)
MEAS_POSTPROCESS_COEFS = tf.convert_to_tensor(MEAS_POSTPROCESS_COEFS,dtype=tf.float32)
MEAS_MASK = tf.expand_dims(tf.convert_to_tensor([t != MeasType.DELTA for t in MEAS_TYPES],dtype=tf.float32),axis=0)

class DoomGame:
    def new_episode(self):
        self.game.new_episode()
        self.episode_memory = []

        self.goal = tf.expand_dims(tf.einsum("a,b->ab",GOAL_MEAS_COEFS,GOAL_TEMPORAL_COEFS),axis=0)

        self.last_action = tf.zeros([],dtype=tf.int32)
        self.episode_length = 0
        self.recurrent_state = None
        self.episode_done = False

    def __init__(self,mode):
        game = vzd.DoomGame()
        game.set_doom_scenario_path(WAD_NAME)
        game.set_doom_map(MAP_NAME)
        
        #TODO: add more resolutions
        if GAME_RESOLUTION == (160,120):
            game.set_screen_resolution(vzd.ScreenResolution.RES_160X120)
        elif GAME_RESOLUTION == (320,240):
            game.set_screen_resolution(vzd.ScreenResolution.RES_320X240)
        elif GAME_RESOLUTION == (640,480):
            game.set_screen_resolution(vzd.ScreenResolution.RES_640X480)
        else:
            print("Game resolution", GAME_RESOLUTION, "not supported")

        if COLORMODE == ColorMode.GRAYSCALE:
            game.set_screen_format(vzd.ScreenFormat.GRAY8)
        elif COLORMODE == ColorMode.COLOR:
            game.set_screen_format(vzd.ScreenFormat.RGB24)
        else:
            print("Color mode",repr(COLORMODE),"not supported.")
        game.set_depth_buffer_enabled(USE_DEPTH_BUFFER)
        game.set_labels_buffer_enabled(USE_LABELED_RECTS)
        game.set_automap_buffer_enabled(False)
        game.set_objects_info_enabled(True)
        game.set_sectors_info_enabled(False)

        game.set_render_hud(True)
        game.set_render_crosshair(False)
        game.set_render_weapon(True)
        game.set_render_decals(False)
        game.set_render_particles(False)
        game.set_render_effects_sprites(True)
        game.set_render_messages(True)
        game.set_render_corpses(True)
        game.set_render_screen_flashes(True)
        
        game.set_doom_skill(4)

        for button in action_list.flattened:
            game.add_available_button(button)
            
        for measurement in VIZDOOM_VARS:
            game.add_available_game_variable(measurement)

        game.set_episode_start_time(1)
        game.set_episode_timeout(EPISODE_LENGTH)
        game.set_window_visible(False)
        game.set_sound_enabled(False)
        game.set_living_reward(-1)
        game.set_mode(mode)
        game.init()
        self.game = game
    
class DoomModel(tf.keras.Model):
    def __init__(self, filename):
        self.lrelu = lambda x: tf.keras.activations.relu(x, alpha=0.2)
        
        self.filename = filename
    
        super(DoomModel, self).__init__()
        self.c1 = tf.keras.layers.Conv2D(32, [8,8], strides=[4,4], padding="same", activation=self.lrelu)
        self.c2 = tf.keras.layers.Conv2D(64, [4,4], strides=[2,2], padding="same", activation=self.lrelu)
        self.c3 = tf.keras.layers.Conv2D(64, [3,3], strides=[1,1], padding="same", activation=self.lrelu)
        self.c4 = tf.keras.layers.Dense(512, activation=self.lrelu)
        
        self.m1 = tf.keras.layers.Dense(128, activation=self.lrelu)
        self.m2 = tf.keras.layers.Dense(128, activation=self.lrelu)
        self.m3 = tf.keras.layers.Dense(128, activation=self.lrelu)
        
        if USE_GOAL_INPUT:
            self.g1 = tf.keras.layers.Dense(128, activation=self.lrelu)
            self.g2 = tf.keras.layers.Dense(128, activation=self.lrelu)
            self.g3 = tf.keras.layers.Dense(128, activation=self.lrelu)
            
        if USE_LAST_ACTION_INPUT:
            self.last_action = tf.keras.layers.Embedding(action_list.total, 128)
        
        if TRAINING_TYPE == TrainingType.FULL_EPISODES_RECURRENT:
            self.lstm1 = tf.keras.layers.LSTM(512,return_sequences=True,return_state=True)

        self.action1 = tf.keras.layers.Dense(512, activation=self.lrelu)
        self.action2 = tf.keras.layers.Dense(N_MEASUREMENTS*N_GOAL_TIMES*N_ACTIONS)

        self.expect1 = tf.keras.layers.Dense(512, activation=self.lrelu)
        self.expect2 = tf.keras.layers.Dense(N_MEASUREMENTS*N_GOAL_TIMES)
        

    @tf.function
    def call(self, image, measurements, goal, last_actions, bool_mask=None, in_state=None):
        image_out = image
        image_out = self.c1(image_out)
        image_out = self.c2(image_out)
        image_out = self.c3(image_out)
        image_out = self.c4(tf.reshape(image_out,[image_out.shape[0],-1]))
        
        #here we mask out the measurements we don't want to give to the network...
        meas_out = tf.multiply(measurements,MEAS_MASK)
        meas_out = self.m1(meas_out)
        meas_out = self.m2(meas_out)
        meas_out = self.m3(meas_out)
        
        if USE_GOAL_INPUT:
            goal_out = tf.reshape(goal,[goal.shape[0],-1])
            goal_out = self.g1(goal_out)
            goal_out = self.g2(goal_out)
            goal_out = self.g3(goal_out)
            concated = [image_out,meas_out,goal_out]
        else:
            concated = [image_out,meas_out]
            
        if USE_LAST_ACTION_INPUT:
            embedded = self.last_action(last_actions)
            concated.append(embedded)
        
        concated = tf.concat(concated,axis=-1)
        
        if TRAINING_TYPE == TrainingType.FULL_EPISODES_RECURRENT:
            concated = self.lstm1(tf.expand_dims(concated,axis=0), initial_state=in_state)
            ret_state = concated[1:]
            concated = tf.squeeze(concated[0],axis=0)
        else:
            ret_state = None

        acts = self.action1(concated)
        acts = self.action2(acts)
        acts = tf.add(acts,-tf.reduce_mean(acts,axis=-1,keepdims=True))
        acts = tf.reshape(acts,[acts.shape[0],N_ACTIONS,N_MEASUREMENTS,N_GOAL_TIMES])
        
        expects = self.expect1(concated)
        expects = self.expect2(expects)
        expects = tf.reshape(expects,[expects.shape[0],1,N_MEASUREMENTS,N_GOAL_TIMES])
        expects = tf.broadcast_to(expects, acts.shape)
        
        total = tf.add(acts,expects)

        if PREDICT_ONLY_DELTAS:
            #...but note that here we use the *non-masked* measurements
            #   to make it possible to predict deltas!
            total = tf.add(total,tf.expand_dims(tf.expand_dims(measurements,axis=1),axis=-1))

        return total,ret_state

class Memories:
    def __init__(self, memory_size):
        self.mem_episode = []
        self.mem_targets = []
        self.memory_size = memory_size
    
    def get_current_len(self):
        return len(self.mem_episode)

    def add(self,episode,targets):
        assert(len(episode) == len(targets))
        #print("Adding!", len(episode), len(targets))
        self.mem_episode.extend(episode)
        self.mem_targets.extend(targets)
        
        newlen = len(self.mem_episode)
        
        if newlen > MEMORY_SIZE:
            if MEMORY_FULL_STRATEGY == MemoryFullStrategy.DELETE_OLDEST:
                crop = newlen - MEMORY_SIZE
                self.mem_episode = self.mem_episode[crop:]
                self.mem_targets = self.mem_targets[crop:]
            elif MEMORY_FULL_STRATEGY == MemoryFullStrategy.DELETE_EVERY_OTHER:
                self.mem_episode = self.mem_episode[::2]
                self.mem_targets = self.mem_targets[::2]
            else:
                print("MemoryFullStrategy",repr(MEMORY_FULL_STRATEGY),"not supported.")
        
    def get_random_memories(self, n):
        indices = np.random.randint(self.get_current_len(), size=(n))
        
        rand_episode = [self.mem_episode[i] for i in indices]
        rand_targets = [self.mem_targets[i] for i in indices]
        
        return rand_episode, rand_targets
        
    def get_all_memories(self):
        return self.mem_episode, self.mem_targets
        
    def clear(self):
        self.mem_episode = []
        self.mem_targets = []  
        
dm = DoomModel(sys.argv[1])
#TODO: implement model loading!

memories = Memories(memory_size=MEMORY_SIZE)

games = [DoomGame(vzd.Mode.PLAYER) for _ in range(N_VIZDOOM_INSTANCES)]

#TODO: implement learning rate schedule
optimizer = tf.keras.optimizers.Adam(learning_rate=1e-4, beta_1=0.95, beta_2=0.999, epsilon=1e-4)

@tf.function
def train_func(screen_buf,gamevars,goal,actions,last_actions,targets,padding_mask):
    with tf.GradientTape() as tape:
        if padding_mask is None:
            out,_ = dm(screen_buf,gamevars,goal,last_actions)
        else:
            out,_ = dm(screen_buf,gamevars,goal,last_actions,tf.cast(padding_mask,tf.bool))
        
        consec = tf.range(actions.shape[0])
        total = tf.stack([consec,actions],axis=-1)
        out2 = tf.gather_nd(out,total)
        
        #hack: target less than -1 million means the target doesn't exist
        #      i tried using NaN for this but failed for whatever reason
        loss = tf.where(targets < -1000000.0,tf.zeros_like(out2),tf.square(targets-out2))
        if padding_mask is not None:
            loss = tf.multiply(loss,tf.reshape(padding_mask,[padding_mask.shape[0],1,1]))
    
    gradients = tape.gradient(loss, dm.trainable_variables)
    optimizer.apply_gradients(zip(gradients, dm.trainable_variables))
    
    return loss,out2

def get_padded_length(n):
    lengths = [150,200,300,400,600,800,1200,1600]
    for l in lengths:
        if l >= n:
            return l
    return n

def pad_tensor_to_len(tensor, new_len):
    added_amount = new_len-tensor.shape[0]
    if added_amount < 0:
        print("pad_tensor_to_len: Added amount less than 0")

    pad = tf.zeros(shape=([added_amount]+tensor.shape[1:]),dtype=tensor.dtype)
    return tf.concat([tensor,pad],axis=0)

def train(eps, targets):
    screen_buf = tf.concat([e[0] for e in eps],axis=0)
    gamevars = tf.concat([e[1] for e in eps],axis=0)
    goal = tf.concat([e[2] for e in eps],axis=0)
    actions = tf.convert_to_tensor([e[3] for e in eps],dtype=tf.int32)
    last_actions = tf.convert_to_tensor([e[4] for e in eps],dtype=tf.int32)
    targets = tf.cast(tf.transpose(tf.stack(targets,axis=0),perm=[0,2,1]),dtype=tf.float32)
    padding_mask = None
    
    if TRAINING_TYPE == TrainingType.FULL_EPISODES or TRAINING_TYPE == TrainingType.FULL_EPISODES_RECURRENT:
        original_len = actions.shape[0]
        padded_len = get_padded_length(original_len)
        
        #this is ugly repetitive code, wonder if there's a better way
        screen_buf = pad_tensor_to_len(screen_buf, padded_len)
        gamevars = pad_tensor_to_len(gamevars, padded_len)
        goal = pad_tensor_to_len(goal, padded_len)
        actions = pad_tensor_to_len(actions, padded_len)
        last_actions = pad_tensor_to_len(last_actions, padded_len)
        targets = pad_tensor_to_len(targets, padded_len)
        padding_mask = pad_tensor_to_len(tf.ones(original_len,dtype=tf.float32),padded_len)
    
    train_func(screen_buf,gamevars,goal,actions,last_actions,targets,padding_mask)

epsilon_func = lambda step: (0.02 + 145000. / (float(step) + 150000.))

#one step of the network
#returns the action chosen
def one_step(game, last_action, episode_memory=None, epsilon=None, in_recurrent=None, goal=None):
    state = game.get_state()

    screen_buf = state.screen_buffer
    screen_buf = cv2.resize(screen_buf, (SCALED_RESOLUTION[1],SCALED_RESOLUTION[0]))
    screen_buf = tf.add(tf.multiply(tf.cast(screen_buf,tf.float32),1.0/255.0),-0.5)
    if COLORMODE == ColorMode.GRAYSCALE:
        screen_buf = tf.reshape(screen_buf,[1,screen_buf.shape[0],screen_buf.shape[1],1])
    elif COLORMODE == ColorMode.COLOR:
        screen_buf = tf.reshape(screen_buf,[1,screen_buf.shape[0],screen_buf.shape[1],3])
    
    if USE_DEPTH_BUFFER:
        depth_buf = state.depth_buffer
        depth_buf = cv2.resize(depth_buf, (SCALED_RESOLUTION[1],SCALED_RESOLUTION[0]))
        depth_buf = tf.add(tf.multiply(tf.cast(depth_buf,tf.float32),1.0/255.0),-0.5)
        depth_buf = tf.reshape(depth_buf,[1,depth_buf.shape[0],depth_buf.shape[1],1])
        screen_buf = tf.concat([screen_buf,depth_buf],axis=-1)

    if USE_LABELED_RECTS:
        canvas_buf = np.zeros([GAME_RESOLUTION[1],GAME_RESOLUTION[0]],dtype=np.float32)
        for lab in state.labels:
            canvas_buf[lab.y:(lab.y+lab.height),lab.x:(lab.x+lab.width)] = 1.0
        canvas_buf = cv2.resize(canvas_buf, (SCALED_RESOLUTION[1],SCALED_RESOLUTION[0]))
        canvas_buf = tf.reshape(canvas_buf,[1,canvas_buf.shape[0],canvas_buf.shape[1],1])
        screen_buf = tf.concat([screen_buf,canvas_buf],axis=-1)

    gamevars = tf.expand_dims(tf.convert_to_tensor([x(game,state) for x in MEAS], dtype=np.float32),axis=0)
    gamevars = tf.multiply(gamevars, MEAS_PREPROCESS_COEFS) #PREPROCESS
    dm_ret = dm(screen_buf,gamevars,goal,tf.reshape(last_action,[1]),in_state=in_recurrent)
    actions = dm_ret[0]
    recurrent_state = dm_ret[1]
    actions = tf.squeeze(actions,axis=0) #[N_ACTIONS,N_MEASUREMENTS,N_GOAL_TIMES]
    
    if epsilon is None:
        chosen_action = tf.multiply(actions,tf.reshape(GOAL_TEMPORAL_COEFS,[1,1,N_GOAL_TIMES]))
        chosen_action = tf.einsum("abc,b->ac", chosen_action, GOAL_MEAS_COEFS)
        chosen_action = tf.reduce_sum(chosen_action,axis=-1)
        chosen_action = tf.cast(tf.argmax(chosen_action,axis=0),tf.int32)
    else:
        rand = np.random.rand(1)
        if rand[0] >= epsilon:
            chosen_action = tf.multiply(actions,tf.reshape(GOAL_TEMPORAL_COEFS,[1,1,N_GOAL_TIMES]))
            chosen_action = tf.einsum("abc,b->ac", chosen_action, GOAL_MEAS_COEFS)
            chosen_action = tf.reduce_sum(chosen_action,axis=-1)
            chosen_action = tf.cast(tf.argmax(chosen_action,axis=0),tf.int32)
        else:
            if RANDOM_CHOICE_TYPE == RandomChoiceType.UNIFORM:
                chosen_action = tf.random.uniform(shape=[], maxval=N_ACTIONS, dtype=tf.int64)
            elif RANDOM_CHOICE_TYPE == RandomChoiceType.SOFTMAX:
                chosen_action = tf.multiply(actions,tf.reshape(GOAL_TEMPORAL_COEFS,[1,1,N_GOAL_TIMES]))
                chosen_action = tf.einsum("abc,b->ac", chosen_action, GOAL_MEAS_COEFS)
                chosen_action = tf.reduce_sum(chosen_action,axis=-1)
                softmaxed = tf.nn.softmax(chosen_action,axis=0)
                chosen_action = choose(softmaxed)
            else:
                print("Random choice type",repr(RANDOM_CHOICE_TYPE),"not supported.")

    if episode_memory is not None:
        episode_memory.append([screen_buf,gamevars,goal,chosen_action,last_action])

    action_vector = action_list.number_to_vector(chosen_action)
    
    game.set_action(action_vector)
    return state,chosen_action,recurrent_state
    
states=0
test_state_counter=0
added_states=0
startstate = states
starttime = time.time()

def game_thread(game):
    epsilon = epsilon_func(states+game.episode_length)
    _, game.last_action,game.recurrent_state = one_step(game.game,game.last_action,game.episode_memory,epsilon,game.recurrent_state,goal=game.goal)
    game.episode_length += 1
    
    if game.game.is_player_dead():
        game.episode_done = True
        return
    if not game.episode_done:
        game.game.advance_action(FRAMESKIP)
    

while True:
    for game in games:
        game.new_episode()

    while True:
        all_done = np.all([game.episode_done for game in games])
        if all_done:
            break
    
        for game in games:
            if game.game.is_episode_finished():
                game.episode_done = True
                continue
        
        thread_list = []
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=N_VIZDOOM_INSTANCES) as executor:
            executor.map(game_thread, games)
        
        #for game in games:
        #    new_t = threading.Thread(target=game_thread, args=(game,))
        #    new_t.start()
        #    thread_list.append(new_t)
            
            
        #for t in thread_list:
        #    t.join()
        
        
    for game in games:
        assert(game.episode_length == len(game.episode_memory))
        states += game.episode_length
        added_states += game.episode_length
        test_state_counter += game.episode_length

        player_died = game.game.is_player_dead()
        
        #create targets for each timestep and add memory to memories
        meas_memory = np.ndarray(shape=(len(game.episode_memory),N_GOAL_TIMES,N_MEASUREMENTS))
        gt = np.asarray(GOAL_TIMES,dtype=np.int32)
        for i in range(meas_memory.shape[0]):
            ts = gt + i
            #hack: if the player died, continue the last state forever to indicate theres nothing you can do after you die
            if player_died:
                ts = np.minimum(ts,meas_memory.shape[0]-1)
                
            #hack: target less than -1 million means the target doesn't exist
            #      i tried using NaN for this but failed for whatever reason
            meas_memory[i] = [-2000000.0]*N_MEASUREMENTS
            for i2 in range(meas_memory.shape[1]):
                t = ts[i2]
                if t < meas_memory.shape[0]:
                    meas_memory[i,i2] = game.episode_memory[t][1].numpy()
        memories.add(game.episode_memory,meas_memory)
    
    #train with memories
    if TRAINING_TYPE == TrainingType.EXPERIENCE_REPLAY:
        while added_states >= BATCH_SIZE:
            ep, tgt = memories.get_random_memories(BATCH_SIZE)
            train(ep, tgt)
            added_states -= BATCH_SIZE
    elif TRAINING_TYPE == TrainingType.FULL_EPISODES or TRAINING_TYPE == TrainingType.FULL_EPISODES_RECURRENT:
        ep, tgt = memories.get_all_memories()
        train(ep, tgt)
        memories.clear()
    else:
        print("Training type",repr(TRAINING_TYPE),"not supported")
        
    if test_state_counter >= TEST_FREQUENCY:
        test_state_counter -= TEST_FREQUENCY
        game = games[0] #test with the first instance
        game.new_episode()
        while not game.game.is_episode_finished():
            game_state, game.last_action, _ = one_step(game.game, game.last_action, goal=game.goal)
        
            game.episode_length += 1
            if game.game.is_player_dead():
                break

            game.game.advance_action(FRAMESKIP)
            
        print(states, "steps seen,", [x(game.game, game_state) for x in MEAS],", survived for", game.episode_length, "steps,", (states-startstate)*1./(time.time()-starttime)," steps/s")
        startstate = states
        starttime = time.time()
        with open("training_"+sys.argv[1]+".log.txt","a") as file:
           file.write("{0} {1} {2}\n".format(states, [x(game.game, game_state) for x in MEAS], game.episode_length))
           
        #TODO: implement model saving!


game.game.close()

"""
idea for additional data point:
ssectors visited
yes, SSECTORS, not SECTORS.
or maybe visited blockmap blocks?
or maybe seen linedefs?

or all of the above! :D

IDEAS:
some sort of "surprise" system. possibly the same as "curiosity" as described by that one paper
"""