import gym
import numpy as np
import matplotlib.pyplot as plt

from agent import RandomAgent
from ddpg import *
from replay_buffer import ReplayBuffer
import util

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('target', nargs='?', default='SpaceInvaders-v0')
    parser.add_argument('--num_game', default=10000)
    args = parser.parse_args()

    env = gym.make(args.target)
    agent = RandomAgent(env.action_space)

    reward = 0
    bRender = True
    bTrain = True

    dim_state = (84,84,4)
    dim_action = env.action_space.shape[0]

    sess = tf.Session()

    actor = ActorNetwork(sess, dim_state, dim_action)
    critic = CriticNetwork(sess, dim_state, dim_action)

    sess.run(tf.global_variables_initializer())

    actor.update_target_network()
    critic.update_target_network()
    
    R = ReplayBuffer()

    for _ in range(args.num_game):
        print('new game has begun')
        ob = env.reset()
        s_t = util.phi(ob)

        r_t = 0
        done = False
        while not done:
            if bRender:
                env.render()

            a_t = agent.act(s_t, r_t, done)
            s_new, r_t, done, _ = env.step(a_t)

            if bTrain:
                R.store(s_t, a_t, r_t, s_new)

                batch = R.sample(batch_size)
                # update critic by minimizing loss
                # update the actor policy using the sampled gradient
                actor.update_target_network()
                critic.update_target_network()

    env.close()
