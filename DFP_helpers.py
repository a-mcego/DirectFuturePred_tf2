from enum import Enum
import tensorflow as tf
import numpy as np

class RandomChoiceType(Enum):
    UNIFORM = 1
    SOFTMAX = 2
    
class TrainingType(Enum):
    EXPERIENCE_REPLAY = 1
    FULL_EPISODES = 2

class MemoryFullStrategy(Enum):
    DELETE_OLD = 1
    DELETE_EVERY_OTHER = 2

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

