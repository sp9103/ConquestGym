import tensorflow as tf

class NetworkBase:
    def __init__(self, sess, dim_state, dim_action, batch_size, tau, learning_rate):
        self.sess = sess
        
        self.dim_state = dim_state
        self.dim_action = dim_action

        # learning parameters
        self.batch_size = batch_size
        self.tau = tau
        self.learning_rate = learning_rate

    def build_network(self, name):
        with tf.variable_scope(name):
            state = tf.placeholder(tf.float32, [None, self.dim_state])
            fcn0 = tf.layers.dense(state, 400, activation=tf.nn.relu)
            fcn1 = tf.layers.dense(fcn0, 300, activation=tf.nn.relu)
            f_init = tf.random_uniform_initializer(-self.UNIFORM_MAX_BOUND, self.UNIFORM_MAX_BOUND)
            model = tf.layers.dense(fcn1, self.dim_action, kernel_initializer=f_init, bias_initializer=f_init, activation=tf.tanh)

            return model, state

    # refer to http://pemami4911.github.io/blog/2016/08/21/ddpg-rl.html#Tensorflow
    # which one is better between multiple run and single run
    def update_target_network(self, b_init=False):
        if b_init:
            self.sess.run([self.target_weights[i].assign(self.weights[i]) for i in range(len(self.weights))])
        else:
            tau = self.tau
            self.sess.run([self.target_weights[i].assign(tau * self.weights[i] + (1-tau) * self.target_weights[i]) for i in range(len(self.weights))])


class ActorNetwork(NetworkBase):
    UNIFORM_MAX_BOUND = 3e-3

    def __init__(self, sess, dim_state, dim_action, batch_size=64, tau=1e-3, learning_rate=1e-4):
        super().__init__(sess, dim_state, dim_action, batch_size, tau, learning_rate)

        self.network, self.state = self.build_network('actor')
        self.weights = tf.trainable_variables()
        self.target_network, _ = self.build_network('actor_target')
        self.target_weights = tf.trainable_variables()[len(self.weights):]

        self.base_ind = len(self.weights) + len(self.target_weights)

        self.optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate)
        
    def act(self, states): # \mu(s|\theta^\mu
        return self.sess.run(self.network, feed_dict={self.state: states})

    def target_act(self, states): # \mu'(s|\theta^{\mu'}
        return self.sess.run(self.target_network, feed_dict={self.state: states})

    def update_policy(self, states, grads):
        train = self.optimizer.apply_gradients([grads, states])
        self.sess.run(train)


class CriticNetwork(NetworkBase):
    UNIFORM_MAX_BOUND = 3e-4

    def __init__(self, sess, dim_state, dim_action, base_ind, batch_size=64, tau=1e-3, learning_rate=1e-3):
        super().__init__(sess, dim_state, dim_action, batch_size, tau, learning_rate)

        self.network, self.state = self.build_network('critic')
        self.weights = tf.trainable_variables()[base_ind:]
        base_ind = base_ind + len(self.weights)

        self.target_network, _ = self.build_network('critic_target')
        self.target_weights = tf.trainable_variables()[base_ind:]

    # update critic
    def update_critic(self):
        a = 1
