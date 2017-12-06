import tensorflow as tf
from util import fanin

def def_var(dim, val):
    return tf.Variable(tf.random_uniform(dim, -val, val))

class NetworkBase:
    def __init__(self, sess, dim_state, dim_action, batch_size, tau, learning_rate):
        self.sess = sess
        
        self.dim_state = dim_state
        self.dim_action = dim_action

        # learning parameters
        self.batch_size = batch_size
        self.tau = tau
        self.learning_rate = learning_rate

        self.optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate)

        
    # refer to http://pemami4911.github.io/blog/2016/08/21/ddpg-rl.html#Tensorflow
    def update_target_network(self, b_init=False):
        if b_init:
            self.sess.run(self.__update_params0)
        else:
            self.sess.run(self.__update_params)

    def init_param_updater(self):
        self.__update_params0 = \
                [self.target_weights[i].assign(self.weights[i]) for i in range(len(self.weights))]
        self.__update_params = \
                [self.target_weights[i].assign(self.tau * self.weights[i] + (1-self.tau) * self.target_weights[i]) for i in range(len(self.weights))]


class ActorNetwork(NetworkBase):
    def __init__(self, sess, dim_state, dim_action, batch_size=64, tau=1e-3, learning_rate=1e-4):
        super().__init__(sess, dim_state, dim_action, batch_size, tau, learning_rate)

        self.network, self.state = self.__build_network('actor')
        self.weights = tf.trainable_variables()

        self.action_gradient = tf.placeholder(tf.float32, [None, self.dim_action])
        self.gradient = tf.gradients(self.network, self.weights, -self.action_gradient)
        self.train = self.optimizer.apply_gradients(zip(self.gradient, self.weights))

        self.target_network, self.target_state = self.__build_network('actor_target')
        self.target_weights = tf.trainable_variables()[len(self.weights):]

        self.init_param_updater()

        self.base_ind = len(self.weights) + len(self.target_weights)

    def __build_network(self, name):
        with tf.variable_scope(name):
            state = tf.placeholder(tf.float32, [None, self.dim_state])

            f1 = fanin(self.dim_state)
            W1 = def_var([self.dim_state, 400], f1)
            b1 = def_var([400], f1)

            f2 = fanin(400 + self.dim_action)
            W2 = def_var([400, 300], f2)
            b2 = def_var([300], f2)

            f3 = 3e-3
            W3 = def_var([300, self.dim_action], f3)
            b3 = def_var([self.dim_action], f3)

            layer1 = tf.nn.relu(tf.matmul(state, W1) + b1)
            layer2 = tf.nn.relu(tf.matmul(layer1, W2) + b2)
            model = tf.tanh(tf.matmul(layer2, W3) + b3)

            return model, state

    def act(self, states): # \mu(s|\theta^\mu
        return self.sess.run(self.network, 
                feed_dict={self.state: states})

    def target_act(self, states): # \mu'(s|\theta^{\mu'}
        return self.sess.run(self.target_network, 
                feed_dict={self.target_state: states})

    def update_policy(self, states, action_grads):
        self.sess.run(self.train, 
                feed_dict={self.state: states, self.action_gradient: action_grads})


class CriticNetwork(NetworkBase):
    def __init__(self, sess, dim_state, dim_action, base_ind, batch_size=64, tau=1e-3, learning_rate=1e-3):
        super().__init__(sess, dim_state, dim_action, batch_size, tau, learning_rate)

        self.network, self.state, self.action = self.__build_network('critic')
        self.weights = tf.trainable_variables()[base_ind:]
        base_ind = base_ind + len(self.weights)

        self.predicted_Q = tf.placeholder(tf.float32, [None])
        self.loss = tf.losses.mean_squared_error(self.network, self.predicted_Q)
        self.train = self.optimizer.minimize(self.loss)

        self.target_network, self.target_state, self.target_action = self.__build_network('critic_target')
        self.target_weights = tf.trainable_variables()[base_ind:]

        self.init_param_updater()

        self.gradient = tf.gradients(self.network, self.action)

    def __build_network(self, name):
        with tf.variable_scope(name):
            state = tf.placeholder(tf.float32, [None, self.dim_state])
            action = tf.placeholder(tf.float32, [None, self.dim_action])

            f1 = fanin(self.dim_state)
            W1 = def_var([self.dim_state, 400], f1)
            b1 = def_var([400], f1)

            f2 = fanin(400 + self.dim_action)
            W2 = def_var([400, 300], f2)
            b2 = def_var([300], f2)
            W2_action = def_var([self.dim_action, 300], f2)

            f3 = 3e-4
            W3 = def_var([300], f3)
            b3 = def_var([1], f3)

            layer1 = tf.nn.relu(tf.matmul(state, W1) + b1)
            layer2 = tf.nn.relu(tf.matmul(layer1, W2) + tf.matmul(action, W2_action) + b2)
            model = tf.tensordot(layer2, W3, 1) + b3

            return model, state, action

    def Q(self, states, actions):
        return self.sess.run(self.network, 
                feed_dict={self.state: states, self.action: actions})

    def target_Q(self, states, actions):
        return self.sess.run(self.target_network, 
                feed_dict={self.target_state: states, self.target_action: actions})

    def update_critic(self, states, actions, predQs):
        self.sess.run(self.train,
                feed_dict={self.state: states, self.action: actions, self.predicted_Q: predQs})

    def dQda(self, states, actions):
        return self.sess.run(self.gradient,
                feed_dict={self.state: states, self.action: actions})

