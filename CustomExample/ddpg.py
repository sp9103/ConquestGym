import tensorflow as tf
import numpy as np
import random
from collections import deque

TRAIN_INTERVAL = 4
BATCH_SIZE = 32
GAMMA = 0.99
TAU = 0.01

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

        self.input_X = tf.placeholder(tf.float32, [None, width, height, self.STATE_LEN])  # 네트워크 입력용
        self.input_A = tf.placeholder(tf.float32, [None])
        self.input_Y = tf.placeholder(tf.float32, [None])
        self.input_Q = tf.placeholder(tf.float32, [None])

        self.main_Q = self._build_critic("Main_Q")
        self.target_Q = self._build_critic("Target_Q")

        self.main_A = self._build_actor("Main_A")
        self.target_Q = self._build_actor("Target_A")

        ## network param
        self.am_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='Main_A')
        self.at_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='Target_A')
        self.cm_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='Main_Q')
        self.ct_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='Target_Q')

        td_error = tf.losses.mean_squared_error(labels=self.input_Y, predictions=self.main_Q)
        self.ctrain = tf.train.AdamOptimizer().minimize(td_error, var_list=self.cm_params)

        #actor는 어떻게 업데이트 해야하지?
        a_loss = -tf.reduce_
        self.atrain = tf.train.AdamOptimizer().minimize(a_loss, var_list=self.am_params)

    def net_update(self):
        for at, am, ct, cm in zip(self.at_params, self.am_params, self.cm_params, self.ct_params):
            tf.assign(at, (1 - TAU) * at + TAU * am)
            tf.assign(ct, (1 - TAU) * ct + TAU * cm)

    def learn(self):
        state, next_state, action, reward, terminal = self.sample()

        #about Critic
        q = self.session.run(self.target_Q, feed_dict={self.input_X:next_state})
        Y = []
        for i in range(self.BATCH_SIZE):
            if terminal[i]:
                Y.append(reward[i])
            else:
                Y.append(reward[i] + self.GAMMA * q)
        self.session.run(self.ctrain, feed_dict={self.input_Y: Y, self.input_X: state})

        #about Actor

        #update network
        self.net_update()

    def get_action(self, state):
        self.session.run(self.main_A, feed_dict={self.input_X: [state]})

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

    def _build_critic(self, scope):
        with tf.variable_scope(scope):
            model = tf.layers.conv2d(self.input_X, 32, [4, 4], padding='same', activation=tf.nn.relu)
            model = tf.layers.conv2d(model, 64, [4, 4], padding='same', activation=tf.nn.relu)
            model = tf.layers.conv2d(model, 64, [2, 2], padding='same', activation=tf.nn.relu)
            model = tf.contrib.layers.flatten(model)
            model = tf.layers.dense(model, 128, activation=tf.nn.relu)

            model_a = tf.layers.dense(self.input_A, 128, activation=tf.nn.relu)
            model = model + model_a                                                 #이부분이 의미가 맞는지 확인이 필요함
            Q = tf.layers.dense(model, 1)

        return Q

    def _build_actor(self, scope):
        model = tf.layers.conv2d(self.input_X, 32, [4, 4], padding='same', activation=tf.nn.relu)
        model = tf.layers.conv2d(model, 64, [4, 4], padding='same', activation=tf.nn.relu)
        model = tf.layers.conv2d(model, 64, [2, 2], padding='same', activation=tf.nn.relu)
        model = tf.contrib.layers.flatten(model)
        model = tf.layers.dense(model, 512, activation=tf.nn.relu)
        a = tf.layers.dense(model, self.n_action)
        a = tf.nn.softmax(a)

        return a


