from __future__ import print_function
import vizdoom as vzd
from matplotlib import pyplot as plt
import tensorflow as tf
from random import choice
import time
import numpy as np
import sys
import cv2
from enum import Enum

if len(sys.argv) != 2:
    print("Give model savefile as argument.")
    exit(0)

class RandomChoiceType(Enum):
    UNIFORM = 1
    SOFTMAX = 2
    
class TrainingType(Enum):
    EXPERIENCE_REPLAY = 1
    FULL_EPISODES = 2
    
#helper class for actions
class Actions:
    def __init__(self, actions):
        self.actions = actions
        self.total=1
        self.flattened = []
        self.groups = []
        for group in self.actions:
            self.total *= len(group)+1
            for act in group:
                self.flattened.append(act)
        
    def number_to_vector(self, n):
        ret = []
        for group in self.actions:
            length = len(group)+1
            value = n%length
            n //= length
            added = [0.0]*(length-1)
            if value>0:
                added[value-1] = 1.0
            ret.extend(added)
        return ret


def get_nth_var(n):
    def h(game,state):
        return state.game_variables[n]
    return h

damage_first_time = True
damage_total_health = 0.0
def get_damagecount(game,state):
    global damage_first_time
    global damage_total_health
    if damage_first_time:
        damage_first_time = False
        damage_total_health = 0.0
        for obj in state.objects:
            if obj.is_sentient > 0.5 and obj.name != 'DoomPlayer':
                damage_total_health += obj.health
    total_health = 0.0
    for obj in state.objects:
        if obj.is_sentient > 0.5 and obj.name != 'DoomPlayer':
            total_health += obj.health
    return damage_total_health - total_health

def get_deadness(game,state):
    return float(game.is_player_dead())

#---START---USER-SUPPLIED-SETTINGS---

#Scenario 1 from the Direct Future Prediction paper
"""WAD_NAME = "d1_basic.wad"
MAP_NAME = "map01"
VIZDOOM_VARS = [vzd.GameVariable.HEALTH]
MEAS = [get_nth_var(0)]
MEAS_PREPROCESS_COEFS = [0.01]
GOAL_MEAS_COEFS = [1.]
action_list = Actions([[vzd.Button.MOVE_FORWARD],[vzd.Button.TURN_LEFT],[vzd.Button.TURN_RIGHT]])
EPISODE_LENGTH = 60*35
USE_DEPTH_BUFFER = False
USE_LABELED_RECTS = False"""

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

TOPK = 16 #only used with the choose_topk function

GOAL_TIMES = [1,2,4,8,16,32,64]
GOAL_TEMPORAL_COEFS = [0.2, 0.2, 0.2, 0.5, 0.5, 1., 1.]
N_GOAL_TIMES = len(GOAL_TIMES)

IMG_SIZE = 84

#RANDOM_CHOICE_TYPE = RandomChoiceType.UNIFORM
RANDOM_CHOICE_TYPE = RandomChoiceType.SOFTMAX

TRAINING_TYPE = TrainingType.EXPERIENCE_REPLAY
#TRAINING_TYPE = TrainingType.FULL_EPISODES
#the FULL_EPISODES training type uses way too much RAM because
#TF ends up making a new graph for every episode of different length.

FRAMESKIP = 4
BATCH_SIZE = 64 #how many steps in one batch of training the network
MEMORY_SIZE = 20000 #how many steps we keep in the experience memory
TEST_FREQUENCY = 2500 #how often we test the network, in steps trained

#having goal as an input to the network seems to be unnecessary.
#here's a way to turn it off! :)
USE_GOAL_INPUT = False

#---END---USER-SUPPLIED-SETTINGS---

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

#choose index from probabilities:
def choose_topk(vec,TOPK):
    vec = tf.reshape(vec,[-1])
    veclen = vec.shape[-1]
    
    sm = tf.math.exp(vec)
    
    #these 3 lines set any probability less than 1/length to zero
    limit = tf.reduce_sum(sm)/tf.cast(veclen,tf.float32)
    mask = tf.cast(tf.greater(sm,limit),tf.float32)
    sm = tf.multiply(sm,mask)

    tops = tf.math.top_k(sm,TOPK)
    cum_sm = tf.math.cumsum(tops.values)
    point = tf.random.uniform([],0.0,tf.reduce_sum(tops.values))
    chosen = tf.reduce_sum(tf.cast(tf.math.less(cum_sm,point),tf.int32))
    chosen = tops.indices[chosen]
    return chosen
    
def choose(vec):
    vec = tf.reshape(vec,[-1])
    veclen = vec.shape[-1]
    sm = tf.math.exp(vec)
    cum_sm = tf.math.cumsum(sm)
    point = tf.random.uniform([],0.0,tf.reduce_sum(sm))
    chosen = tf.cast(tf.reduce_sum(tf.cast(tf.math.less(cum_sm,point),tf.int32)),tf.int32)
    return chosen

def init(game,mode):
    game.set_doom_scenario_path(WAD_NAME)
    game.set_doom_map(MAP_NAME)

    #game.set_screen_resolution(vzd.ScreenResolution.RES_640X480)
    #game.set_screen_resolution(vzd.ScreenResolution.RES_320X240)
    game.set_screen_resolution(vzd.ScreenResolution.RES_160X120)
    #game.set_screen_format(vzd.ScreenFormat.RGB24)
    game.set_screen_format(vzd.ScreenFormat.GRAY8)
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
    game.set_window_visible(True)
    game.set_sound_enabled(False)
    game.set_living_reward(-1)
    game.set_mode(mode)
    game.init()
    
class DeltaTracker:
    def set_delta(self, current):
        try:
            self.delta = current-self.prev
            self.prev = current
            return self.delta
        except AttributeError:
            self.prev = current
            self.delta = 0
            return 0
    
    def last_delta(self):
        return self.delta

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
        
        self.action1 = tf.keras.layers.Dense(512, activation=self.lrelu)
        self.action2 = tf.keras.layers.Dense(N_MEASUREMENTS*N_GOAL_TIMES*N_ACTIONS)

        self.expect1 = tf.keras.layers.Dense(512, activation=self.lrelu)
        self.expect2 = tf.keras.layers.Dense(N_MEASUREMENTS*N_GOAL_TIMES)

    @tf.function
    def call(self, image, measurements, goal):
        image_out = image
        image_out = self.c1(image_out)
        image_out = self.c2(image_out)
        image_out = self.c3(image_out)
        image_out = self.c4(tf.reshape(image_out,[image_out.shape[0],-1]))
        
        meas_out = measurements
        meas_out = self.m1(meas_out)
        meas_out = self.m2(meas_out)
        meas_out = self.m3(meas_out)
        
        if USE_GOAL_INPUT:
            goal_out = tf.reshape(goal,[goal.shape[0],-1])
            goal_out = self.g1(goal_out)
            goal_out = self.g2(goal_out)
            goal_out = self.g3(goal_out)
            concated = tf.concat([image_out,meas_out,goal_out],axis=-1)
        else:
            concated = tf.concat([image_out,meas_out],axis=-1)
        
        acts = self.action1(concated)
        acts = self.action2(acts)
        acts = tf.add(acts,-tf.reduce_mean(acts,axis=-1,keepdims=True))
        acts = tf.reshape(acts,[acts.shape[0],N_ACTIONS,N_MEASUREMENTS,N_GOAL_TIMES])
        
        expects = self.expect1(concated)
        expects = self.expect2(expects)
        expects = tf.reshape(expects,[expects.shape[0],1,N_MEASUREMENTS,N_GOAL_TIMES])
        expects = tf.broadcast_to(expects, acts.shape)
        
        #TODO: add measurements to the total too!
        #      that way we're only predicting the change, not the actual measurements.
        
        total = tf.add(acts,expects)
        return total

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
            crop = newlen - MEMORY_SIZE
            self.mem_episode = self.mem_episode[crop:]
            self.mem_targets = self.mem_targets[crop:]
        
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
memories = Memories(memory_size=MEMORY_SIZE)

PRINTTIME = 64

game = vzd.DoomGame()
init(game,vzd.Mode.PLAYER)

optimizer = tf.keras.optimizers.Adam(learning_rate=1e-4, beta_1=0.95, beta_2=0.999, epsilon=1e-4)


@tf.function
def train_func(screen_buf,gamevars,goal,actions,targets):
    with tf.GradientTape() as tape:
        out = dm(screen_buf,gamevars,goal)
        consec = tf.range(actions.shape[0])
        total = tf.stack([consec,actions],axis=-1)
        out2 = tf.gather_nd(out,total)
        
        #hack: target less than -1 million means the target doesn't exist
        #      i tried using NaN for this but failed for whatever reason
        loss = tf.where(targets < -1000000.0,tf.zeros_like(out2),tf.square(targets-out2))

    gradients = tape.gradient(loss, dm.trainable_variables)
    optimizer.apply_gradients(zip(gradients, dm.trainable_variables))

def train(eps, targets):
    screen_buf = tf.concat([e[0] for e in eps],axis=0)
    gamevars = tf.concat([e[1] for e in eps],axis=0)
    goal = tf.concat([e[2] for e in eps],axis=0)
    actions = tf.convert_to_tensor([e[3] for e in eps],dtype=tf.int32)
    targets = tf.cast(tf.transpose(tf.stack(targets,axis=0),perm=[0,2,1]),dtype=tf.float32)
    train_func(screen_buf,gamevars,goal,actions,targets)
    

epsilon_func = lambda step: (0.02 + 145000. / (float(step) + 150000.))

states=0
test_state_counter=0
added_states=0
startstate = states
starttime = time.time()
while True:
    game.new_episode()
    episode_memory = []
    goal = tf.expand_dims(tf.einsum("a,b->ab",GOAL_MEAS_COEFS,GOAL_TEMPORAL_COEFS),axis=0)
    while not game.is_episode_finished():
        state = game.get_state()
        screen_buf = state.screen_buffer
        screen_buf = cv2.resize(screen_buf, (IMG_SIZE,IMG_SIZE))
        screen_buf = tf.add(tf.multiply(tf.cast(screen_buf,tf.float32),1.0/255.0),-0.5)
        screen_buf = tf.reshape(screen_buf,[1,screen_buf.shape[0],screen_buf.shape[1],1])
        
        if USE_DEPTH_BUFFER:
            depth_buf = state.depth_buffer
            depth_buf = cv2.resize(depth_buf, (IMG_SIZE,IMG_SIZE))
            depth_buf = tf.add(tf.multiply(tf.cast(depth_buf,tf.float32),1.0/255.0),-0.5)
            depth_buf = tf.reshape(depth_buf,[1,depth_buf.shape[0],depth_buf.shape[1],1])
            screen_buf = tf.concat([screen_buf,depth_buf],axis=-1)

        if USE_LABELED_RECTS:
            canvas_buf = np.zeros([120,160],dtype=np.float32)
            for lab in state.labels:
                canvas_buf[lab.y:(lab.y+lab.height),lab.x:(lab.x+lab.width)] = 1.0
            canvas_buf = cv2.resize(canvas_buf, (IMG_SIZE,IMG_SIZE))
            canvas_buf = tf.reshape(canvas_buf,[1,canvas_buf.shape[0],canvas_buf.shape[1],1])
            screen_buf = tf.concat([screen_buf,canvas_buf],axis=-1)

        gamevars = tf.expand_dims(tf.convert_to_tensor([x(game,state) for x in MEAS], dtype=np.float32),axis=0)
        gamevars = tf.multiply(gamevars, MEAS_PREPROCESS_COEFS) #PREPROCESS
        actions = dm(screen_buf,gamevars,goal)
        actions = tf.squeeze(actions,axis=0) #[N_ACTIONS,N_MEASUREMENTS,N_GOAL_TIMES]
        
        epsilon = epsilon_func(states)
        rand = np.random.rand(1)
        
        #debug print
        #print([x(game,state) for x in MEAS], game.is_player_dead())
        
        #crude epsilon thingy system
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
            
        episode_memory.append([screen_buf,gamevars,goal,chosen_action])

        action_vector = action_list.number_to_vector(chosen_action)
        
        states += 1
        added_states += 1
        test_state_counter += 1

        if game.is_player_dead():
            break
        
        game.set_action(action_vector)
        game.advance_action(FRAMESKIP)
        
    player_died = game.is_player_dead()
        
    #add memory to memories
    meas_memory = np.ndarray(shape=(len(episode_memory),N_GOAL_TIMES,N_MEASUREMENTS))
    for i in range(len(episode_memory)):
        mem = episode_memory[i]
        for i2 in range(len(GOAL_TIMES)):
            t = GOAL_TIMES[i2]
            #hack: if the player died, continue the last state forever to indicate theres nothing you can do after you die
            if player_died:
                future_timestep = min(i+t,len(episode_memory)-1)
            else:
                future_timestep = i+t
            if future_timestep < len(episode_memory):
                meas_memory[i,i2] = episode_memory[future_timestep][1].numpy()
            else:
                #hack: target less than -1 million means the target doesn't exist
                #      i tried using NaN for this but failed for whatever reason
                meas_memory[i,i2] = [-2000000.0]*N_MEASUREMENTS
    memories.add(episode_memory,meas_memory)
    
    #train with memories
    if TRAINING_TYPE == TrainingType.EXPERIENCE_REPLAY:
        while added_states >= BATCH_SIZE:
            ep, tgt = memories.get_random_memories(BATCH_SIZE)
            train(ep, tgt)
            added_states -= BATCH_SIZE
    elif TRAINING_TYPE == TrainingType.FULL_EPISODES:
        ep, tgt = memories.get_all_memories()
        train(ep, tgt)
        memories.clear()
    else:
        print("Training type",repr(TRAINING_TYPE),"not supported")
        
    if test_state_counter >= TEST_FREQUENCY:
        test_state_counter -= TEST_FREQUENCY
        game.new_episode()
        goal = tf.expand_dims(tf.einsum("a,b->ab",GOAL_MEAS_COEFS,GOAL_TEMPORAL_COEFS),axis=0)
        test_states_seen = 0
        while not game.is_episode_finished():
            state = game.get_state()
            screen_buf = state.screen_buffer
            #TODO: make it support RGB as well.
            screen_buf = cv2.resize(screen_buf, (IMG_SIZE,IMG_SIZE))
            screen_buf = tf.add(tf.multiply(tf.cast(screen_buf,tf.float32),1.0/255.0),-0.5)
            screen_buf = tf.reshape(screen_buf,[1,screen_buf.shape[0],screen_buf.shape[1],1])

            if USE_DEPTH_BUFFER:
                depth_buf = state.depth_buffer
                depth_buf = cv2.resize(depth_buf, (IMG_SIZE,IMG_SIZE))
                depth_buf = tf.add(tf.multiply(tf.cast(depth_buf,tf.float32),1.0/255.0),-0.5)
                depth_buf = tf.reshape(depth_buf,[1,depth_buf.shape[0],depth_buf.shape[1],1])
                screen_buf = tf.concat([screen_buf,depth_buf],axis=-1)

            if USE_LABELED_RECTS:
                canvas_buf = np.zeros([120,160],dtype=np.float32)#TODO: make 160,120 depend on original screen_buf shape
                for lab in state.labels:
                    canvas_buf[lab.y:(lab.y+lab.height),lab.x:(lab.x+lab.width)] = 1.0
                canvas_buf = cv2.resize(canvas_buf, (IMG_SIZE,IMG_SIZE))
                canvas_buf = tf.reshape(canvas_buf,[1,canvas_buf.shape[0],canvas_buf.shape[1],1])
                screen_buf = tf.concat([screen_buf,canvas_buf],axis=-1)

            gamevars = tf.expand_dims(tf.convert_to_tensor([x(game,state) for x in MEAS], dtype=np.float32),axis=0)
            gamevars = tf.multiply(gamevars, MEAS_PREPROCESS_COEFS) #PREPROCESS
            actions = dm(screen_buf,gamevars,goal)
            actions = tf.reshape(actions,[N_ACTIONS,N_MEASUREMENTS,N_GOAL_TIMES])
            chosen_action = tf.multiply(actions,tf.reshape(GOAL_TEMPORAL_COEFS,[1,1,N_GOAL_TIMES]))
            chosen_action = tf.einsum("abc,b->ac", chosen_action, GOAL_MEAS_COEFS)
            chosen_action = tf.reduce_sum(chosen_action,axis=-1)
            chosen_action = tf.argmax(chosen_action,axis=0)
            action_vector = action_list.number_to_vector(chosen_action)

            test_states_seen += 1

            if game.is_player_dead():
                break
            
            game.set_action(action_vector)
            game.advance_action(FRAMESKIP)
        print(states, "steps seen,", [x(game, state) for x in MEAS],", survived for", test_states_seen, "steps,", (states-startstate)*1./(time.time()-starttime)," steps/s")
        startstate = states
        starttime = time.time()
        with open("training_"+sys.argv[1]+".log.txt","a") as file:
           file.write("{0} {1} {2}\n".format(states, [x(game, state) for x in MEAS], test_states_seen))

game.close()

"""
idea for additional data point:
ssectors visited
yes, SSECTORS, not SECTORS.
or maybe visited blockmap blocks?
or maybe seen linedefs?

or all of the above! :D

TODO:
learning rate schedule!
model saving!

IDEAS:
do X, something should happen, but it doesnt happen. this should ring alarm bells with the bot.
add previous action to the network input
"""