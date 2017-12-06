import gym
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import gc

from ddpg import *
from replay_buffer import ReplayBuffer
from ou_noise import OUNoise
import util

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    # parser.add_argument('target', nargs='?', default='SpaceInvaders-v0')
    parser.add_argument('target', nargs='?', default='LunarLander-v2')
    parser.add_argument('--num_game', default=30000)
    args = parser.parse_args()

    env = gym.make(args.target)

    gc.enable()

    reward = 0
    bRender = False
    bTrain = True
    batch_size = 64
    gamma = .99

    dim_state = env.observation_space.shape[0]
    dim_action = env.action_space.shape[0]

    print('dim_state: {}, dim_action: {}'.format(dim_state, dim_action))
    
    sess = tf.Session()

    actor = ActorNetwork(sess, dim_state, dim_action)
    critic = CriticNetwork(sess, dim_state, dim_action, actor.base_ind)

    saver = tf.train.Saver(tf.global_variables())
    ckpt = tf.train.get_checkpoint_state('./model')
    if ckpt and tf.train.checkpoint_exists(ckpt.model_checkpoint_path):
        saver.restore(sess, ckpt.model_checkpoint_path)
    else:
        sess.run(tf.global_variables_initializer())

    rewards = tf.placeholder(tf.float32, [None])
    tf.summary.scalar('avg.reward/ep.', tf.reduce_mean(rewards))

    writer = tf.summary.FileWriter('logs', sess.graph)
    summary_merged = tf.summary.merge_all()

    actor.update_target_network(True)
    critic.update_target_network(True)
    
    R = ReplayBuffer()
    N = OUNoise(dim_action, sigma=.3)
    
    reward_list = []
    
    time_step = 0

    for ep in range(args.num_game):
        N.reset()
        s_t = env.reset()
        r_t = 0
        done = False
        reward = 0
        cnt = 0
        while not done:
            if bRender:# and cnt % 3 == 0:
                env.render()

            a_t = actor.act([s_t]) + N.noise()
            a = sess.run(tf.argmax(a_t[0]))
            s_new, r_t, done, _ = env.step(a)
            reward = reward + r_t

            if bTrain:
                R.store(s_t, a_t[0], r_t, s_new, done)

                batch = R.sample(batch_size)
                
                s_s = np.array([_[0] for _ in batch])
                a_s = np.array([_[1] for _ in batch])
                y_s = []

                for it in batch:
                    yi = it[2]
                    if not it[-1]:
                        yi += gamma * critic.target_Q([it[3]], actor.target_act([it[3]]))[0]
                    y_s.append(yi)
                y_s = np.array(y_s)

                critic.update_critic(s_s, a_s, y_s)
                ag_s = critic.dQda(s_s, a_s)

                actor.update_policy(s_s, ag_s[0])

                actor.update_target_network()
                critic.update_target_network()

            s_t = s_new
            cnt += 1
            time_step += 1

            reward_list.append(reward)

        print('ep #{} score: {}'.format(ep, reward))

        if bTrain:
            if ep % 10 == 0:
                summary = sess.run(summary_merged, feed_dict={rewards: reward_list})
                writer.add_summary(summary, time_step)
                reward_list = []

            if ep > 0 and ep % 50 == 0:
                saver.save(sess, 'model/' + args.target + '.ckpt', global_step=time_step)
                gc.collect()
                print(gc.get_stats())

    env.close()
