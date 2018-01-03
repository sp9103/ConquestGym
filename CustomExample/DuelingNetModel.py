#Deuling Network model

#Component
#Dueling Network Architectures for Deep Reinforcement Learning
#Prioritized Experience Replay
#Deep Reinforcement Learning with Double Q-learning
#Playing Atari with Deep Reinforcement Learning

import tensorflow as tf
import numpy as np

from SumTree import Memory      #Prioritized Experience Replay

class DuelDQN:
    REPLAY_MEMORY = 10000       #Replay buffer size
    BATCH_SIZE = 32             #training mini batchsize
    GAMMA = 0.99                #과거 상태 가감 계수
    STATE_LEN = 4               #입력 샘플 하나당 볼 과거 프레임의 수

    def __init__(self, session, width, height, n_action):
        self.session = session
        self.n_action = n_action
        self.width = width
        self.height = height

        self.state = None

        self.input_X = tf.placeholder(tf.float32, [None, width, height, self.STATE_LEN])    #네트워크 입력용
        self.input_A = tf.placeholder(tf.int64, [None])
        self.input_Y = tf.placeholder(tf.float32, [None])

        self.memory = Memory(capacity=self.REPLAY_MEMORY)
        self.ISWeights = tf.placeholder(tf.float32, [None], name='IS_weights')

        self.Q = self._build_network('main')
        self.target_Q = self._build_network('target')

        self.cost, self.train_op, self.abs_error = self._build_op()

    def _build_network(self, name):
        with tf.variable_scope(name):
            model = tf.layers.conv2d(self.input_X, 32, [4, 4], padding='same', activation=tf.nn.relu)
            model = tf.layers.conv2d(model, 64, [4, 4], padding='same', activation=tf.nn.relu)
            model = tf.layers.conv2d(model, 64, [2, 2], padding='same', activation=tf.nn.relu)
            model = tf.contrib.layers.flatten(model)
            model = tf.layers.dense(model, 512, activation=tf.nn.relu)
            model_V = tf.layers.dense(model, 1)                     #value function
            model_A = tf.layers.dense(model, self.n_action)         #action function
            reduce_A = tf.reduce_mean(model_A, keep_dims=True, axis=1)
            Q = model_V + (model_A - reduce_A)       #이부분 확인해봐야함

        return Q

    def _build_op(self):
        one_hot = tf.one_hot(self.input_A, self.n_action, 1.0, 0.0)
        Q_value = tf.reduce_sum(tf.multiply(self.Q, one_hot), axis=1)
        TD_diff = self.input_Y - Q_value
        abs_error = tf.abs(TD_diff)                                 #for updating Sumtree
        mul = tf.multiply(self.ISWeights, tf.square(TD_diff))
        cost = tf.reduce_sum(mul)

        train_op = tf.train.AdamOptimizer(1e-6).minimize(cost)
        return cost, train_op, abs_error

    def update_target_network(self):
        copy_op = []

        main_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='main')
        target_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='target')

        for main_var, target_var in zip(main_vars, target_vars):
            copy_op.append(target_var.assign(main_var.value()))

        self.session.run(copy_op)

    def get_action(self):
        Q_value = self.session.run(self.Q, feed_dict={self.input_X: [self.state]})
        action = np.argmax(Q_value[0])

        return action

    def init_state(self, state):
        state = [state for _ in range(self.STATE_LEN)]
        self.state = np.stack(state, axis=2)

    def remember(self, state, action, reward, terminal):
        next_state = np.reshape(state, (self.width, self.height, 1))
        next_state = np.append(self.state[:, :, 1:], next_state, axis=2)    #가장 오래된 프레임을 제외한 나머지 프레임을 state 뒤에 붙임

        transition = (self.state, next_state, action, reward, terminal)
        self.memory.store(transition)

        self.state = next_state

    def _sample_memory(self):
        tree_idx = np.empty((self.BATCH_SIZE,), dtype=np.int32)

        tree_idx, batch_memory, ISWeights = self.memory.sample(self.BATCH_SIZE)

        # 현재 Array에 Append 하는 형식으로 구현되어 있는데 추후 속도 향상을 위해 미리 메트릭스를 만드는 방식이 필요함
        action = []
        reward = []
        terminal = []
        state = []
        next_state = []

        for i in range(self.BATCH_SIZE):
            state.append(batch_memory[i][0])
            next_state.append(batch_memory[i][1])
            action.append(batch_memory[i][2])
            reward.append(batch_memory[i][3])
            terminal.append(batch_memory[i][4])

        return state, next_state, action, reward, terminal, tree_idx, ISWeights

    def train(self):
        state, next_state, action, reward, terminal, tree_idx, ISWeights = self._sample_memory()

        target_Q_value = self.session.run(self.target_Q,
                                          feed_dict={self.input_X: next_state})
        Q_value = self.session.run(self.Q,
                                   feed_dict={self.input_X: next_state})
        main_action = np.argmax(Q_value, axis=1)

        Y = []
        for i in range(self.BATCH_SIZE):
            if terminal[i]:
                Y.append(reward[i])
            else:
                Q_prime = target_Q_value[i][main_action[i]]
                Y.append(reward[i] + self.GAMMA * Q_prime)

        _, _, abs_error = self.session.run([self.train_op, self.cost, self.abs_error],
                                           feed_dict={self.input_X: state,
                                                      self.input_A: action,
                                                      self.input_Y: Y,
                                                      self.ISWeights: ISWeights
                                                      })
        self.memory.batch_update(tree_idx, abs_error)  # update priority