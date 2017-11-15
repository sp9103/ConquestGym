import tensorflow as tf

# input: 84x84 gray image
# output: action
width = 84
height = 84
num_action = 3
num_state = 4
x = tf.placeholder(tf.float32, [None, width, height, num_state])
y = tf.placeholder(tf.float32, [None, num_action])

def build_dqn():
    # conv1: input=[84,84,4], output=[20,20,16]
    model = tf.layers.conv2d(x, 16, 8, strides=4, padding='valid', activation=tf.nn.relu)
    # conv2: input=[20,20,16], output=[9,9,32]
    model = tf.layers.conv2d(model, 32, 4, strides=2, padding='valid', activation=tf.nn.relu)
    # fcn3: input=[9,9,32], output=[256]
    model = tf.layers.flatten(model)
    model = tf.layers.dense(model, 256, activation=tf.nn.relu)
    # fcn4: input=[256], output=[num_action]
    model = tf.layers.dense(model, num_action)

    return model
