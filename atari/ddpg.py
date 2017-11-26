import tensorflow as tf

class NetworkBase:
    def __init__(self, sess, dim_state, dim_action, batch_size=16, tau=.01, learning_rate=.001):
        self.sess = sess
        
        self.dim_state = dim_state
        self.dim_action = dim_action

        # learning parameters
        self.batch_size = batch_size
        self.tau = tau
        self.learning_rate = learning_rate

    # update target network
    def update_target_network(self):
        tau = self.tau
        # refer to http://pemami4911.github.io/blog/2016/08/21/ddpg-rl.html#Tensorflow
        # which one is better between multiple run and single run
        self.sess.run([self.target_weights[i].assign(tau * self.weights[i] + (1-tau) * self.target_weights[i]) for i in range(len(self.weights))])


class ActorNetwork(NetworkBase):
    def __init__(self, sess, dim_state, dim_action):
        super().__init__(sess, dim_state, dim_action)
        self.network, self.state = self.__build_network(self.network_name())
        self.weights = tf.trainable_variables()
        self.target_network, _ = self.__build_network(self.network_name() + '_target')
        self.target_weights = tf.trainable_variables()[len(self.weights):]

    # use dqn structure
    def __build_network(self, name):
        with tf.variable_scope(name):
            state = tf.placeholder(tf.float32, (None,)+self.dim_state)
            # conv0: input=[84,84,4], output=[20,20,16]
            conv0 = tf.layers.conv2d(state, 16, 8, strides=4, padding='valid', activation=tf.nn.relu)
            # conv1: input=[20,20,16], output=[9,9,32]
            conv1 = tf.layers.flatten(tf.layers.conv2d(conv0, 32, 4, strides=2, padding='valid', activation=tf.nn.relu))
            # fcn2: input=[9*9*32], output=[256]
            fcn2 = tf.layers.dense(conv1, 256, activation=tf.nn.relu)
            # fcn3: input=[256], output=[dim_action]
            out = tf.layers.dense(fcn2, self.dim_action)

            return out, state
        
    # select action under the current policy
    def act(self, s_t):
        pass
        #self.sess.run(..., feed_dict={self.state: state})

    # update policy
    def update_policy(self):
        a = 1


class CriticNetwork(NetworkBase):
    def __init__(self, sess, dim_state, dim_action):
        super().__init__(sess, dim_state, dim_action)

    # update critic
    def update_critic(self):
        a = 1
