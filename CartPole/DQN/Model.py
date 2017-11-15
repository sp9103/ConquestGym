import tensorflow as tf
import numpy as np
import random
from collections import deque

class DQN:
    REPLAY_MEMORY = 10000
    BATCH_SIZE = 32
    GAMMA = 0.99
    STATE_LEN = 4

    def __init__(selfs, session, dim, n_action):
        selfs.session = session
        selfs.n_action = n_action
        selfs.memory = deque()
        selfs.state = None
        selfs.dimension = dim           #state dimesion

        selfs.input_X = tf.placeholder(tf.float32, [None, dim * selfs.STATE_LEN])
        selfs.input_A = tf.placeholder(tf.int64, [None])
        selfs.input_Y = tf.placeholder(tf.float32, [None])

        selfs.Q = selfs._build_network('main')
        selfs.cost, selfs.train_op = selfs._build_op()

        selfs.target_Q = selfs._build_network('target')

    def _build_network(selfs, name):
        with tf.variable_scope(name):
            model = tf.layers.dense(selfs.input_X, 32, activation=tf.nn.relu)
            model = tf.layers.dense(model, 32, activation=tf.nn.relu)
            model = tf.layers.dense(model, 16, activation=tf.nn.relu)

            Q = tf.layers.dense(model, selfs.n_action, activation=None)

        return Q

    def _build_op(self):
        # it will be updated DQN -> Doubling DQN
        one_hot = tf.one_hot(self.input_A, self.n_action, 1.0, 0.0)
        Q_value = tf.reduce_sum(tf.multiply(self.Q, one_hot), axis=1)
        cost = tf.reduce_mean(tf.square(self.input_Y - Q_value))
        train_op = tf.train.AdamOptimizer(1e-3).minimize(cost)

        return cost, train_op

    def update_target_network(self):
        copy_op = []

        main_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='main')
        target_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='target')

        for main_var, target_var in zip(main_vars, target_vars):
            copy_op.append(target_var.assign(main_var.value()))

        self.session.run(copy_op)

    def get_action(self):
        Q_value = self.session.run(self.Q,
                                   feed_dict={self.input_X: [self.state]})

        action = np.argmax(Q_value[0])

        return action

    def init_state(self, state):
        state = [state for _ in range(self.STATE_LEN)]
        self.state = np.reshape(state, self.dimension * self.STATE_LEN)

    def remember(self, state, action, reward, terminal):
        #next_state = np.reshape(state, (self.width, self.height, 1))
        pop_state = self.state[self.dimension:]
        next_state = np.append(pop_state, state, axis=0)

        self.memory.append((self.state, next_state, action, reward, terminal))

        if len(self.memory) > self.REPLAY_MEMORY:
            self.memory.popleft()

        self.state = next_state

    def _sample_memory(self):
        sample_memory = random.sample(self.memory, self.BATCH_SIZE)

        state = [memory[0] for memory in sample_memory]
        next_state = [memory[1] for memory in sample_memory]
        action = [memory[2] for memory in sample_memory]
        reward = [memory[3] for memory in sample_memory]
        terminal = [memory[4] for memory in sample_memory]

        return state, next_state, action, reward, terminal

    def train(self):
        state, next_state, action, reward, terminal = self._sample_memory()

        target_Q_value = self.session.run(self.target_Q,
                                          feed_dict={self.input_X: next_state})

        # DQN 의 손실 함수에 사용할 핵심적인 값을 계산하는 부분입니다. 다음 수식을 참고하세요.
        # if episode is terminates at step j+1 then r_j
        # otherwise r_j + γ*max_a'Q(ð_(j+1),a';θ')
        # input_Y 에 들어갈 값들을 계산해서 넣습니다.
        Y = []
        for i in range(self.BATCH_SIZE):
            if terminal[i]:
                Y.append(reward[i])
            else:
                Y.append(reward[i] + self.GAMMA * np.max(target_Q_value[i]))

        _, cost_val = self.session.run([self.train_op, self.cost],
                         feed_dict={
                             self.input_X: state,
                             self.input_A: action,
                             self.input_Y: Y
                         })

        return cost_val
