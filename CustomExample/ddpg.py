import tensorflow as tf
import numpy as np
import random
from collections import deque

TRAIN_INTERVAL = 4
BATCH_SIZE = 32

class DDPG:
    def __init__(self, session , width, height, n_action, memory_size):
        self.session = session
        self.n_action = n_action
        self.width = width
        self.height = height
        self.BATCH_SIZE = BATCH_SIZE
        self.REPLAY_MEMORY = memory_size

        self.state = None
        self.memory = deque()

    def learn(self):

    def get_action(self, state):

    def remember(self, state, action, reward, terminal):
        next_state = np.reshape(state, (self.width, self.height, 1))
        next_state = np.append(self.state[:, :, 1:], next_state, axis=2)  # 가장 오래된 프레임을 제외한 나머지 프레임을 state 뒤에 붙임

        self.memory.append((self.state, next_state, action, reward, terminal))

        if len(self.memory) > self.REPLAY_MEMORY:
            self.memory.popleft()

        self.state = next_state

    def sample(self):
        sample_memory = random.sample(self.memory, self.BATCH_SIZE)

        state = [memory[0] for memory in sample_memory]
        next_state = [memory[1] for memory in sample_memory]
        action = [memory[2] for memory in sample_memory]
        reward = [memory[3] for memory in sample_memory]
        terminal = [memory[4] for memory in sample_memory]

        return state, next_state, action, reward, terminal

    def InitState(self, state):
        state = [state for _ in range(self.STATE_LEN)]
        self.state = np.stack(state, axis=2)

    def _build_base_net(self, name):
        with tf.variable_scope(name):
            model = tf.layers.conv2d(self.input_X, 32, [4, 4], padding='same', activation=tf.nn.relu)
            model = tf.layers.conv2d(model, 64, [4, 4], padding='same', activation=tf.nn.relu)
            model = tf.layers.conv2d(model, 64, [2, 2], padding='same', activation=tf.nn.relu)
            model = tf.contrib.layers.flatten(model)
            model = tf.layers.dense(model, 512, activation=tf.nn.relu)

    def _build_critic(self):
        model = self._build_base_net('CRITIC')

    def _build_actor(self):
        model = self._build_base_net('ACTOR')