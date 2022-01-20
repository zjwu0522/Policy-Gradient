import numpy as np
import gym
from utils import *
from agent import *
from config import *

def train(agent, env, n_episode, max_t, scale=1):
    rewards_log = []
    average_log = []
    
    for i in range(1, n_episode+1):    
        state = env.reset()
        done = False
        t = 0
        state_history = [list(state)]
        action_history = []
        done_history = []
        reward_history = []
        episodic_reward = 0
        
        while not done and t < max_t:
            action = agent.act(state)
            next_state, reward, done, _ = env.step(action)
            episodic_reward += reward
            action_history.append(action)
            done_history.append(done)
            reward_history.append(reward * scale)
            state = next_state
            state_history.append(list(state))
            t += 1
            
        state_values, log_probs, rewards, discounted_rewards, dones = agent.process_data(state_history, action_history, reward_history, done_history, 64)
        agent.learn(state_values, log_probs, rewards, discounted_rewards, dones)
        
        rewards_log.append(episodic_reward)
        average_log.append(np.mean(rewards_log[-100:]))
        
        print('\rEpisode {} Reward {:.2f}, Average Reward {:.2f}'.format(i, episodic_reward, average_log[-1]), end='')
        if i % 100 == 0:
            print()
            
    return rewards, average_log

if __name__ == '__main__':
    env = gym.make(RAM_ENV_NAME)
    agent = Agent(env.observation_space.shape[0], env.action_space.n, LEARNING_RATE, GAMMA, DEVICE, SHARE, MODE, CRITIC, NORMALIZE)
    rewards_log, _ = train(agent, env, RAM_NUM_EPISODE, MAX_T, SCALE)
    np.save('{}_rewards.npy'.format(RAM_ENV_NAME), rewards_log)