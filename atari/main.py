import gym
import dqn
from agent import RandomAgent

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('target', nargs='?', default='SpaceInvaders-v0')
    parser.add_argument('--num_game', default=100)
    args = parser.parse_args()

    env = gym.make(args.target)
    agent = RandomAgent(env.action_space)
    
    Q = dqn.build_dqn()

    reward = 0
    done = False
    render = True

    for _ in range(args.num_game):
        ob = env.reset()
        while True:
            action = agent.act(ob, reward, done)
            ob, reward, done, _ = env.step(action)
            if render:
                env.render()
            if done:
                break

    env.close()
