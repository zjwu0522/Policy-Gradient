import numpy as np
import gym
from utils import *
from agent import *
from config_discrete import *

def train(agent, env, n_episode, n_update=4, update_frequency=1, max_t=1500, scale=1):
    rewards_log = []
    average_log = []
    state_history = []
    action_history = []
    done_history = []
    reward_history = []
        
    for i in range(1, n_episode+1):    
        state = env.reset()
        done = False
        t = 0
        if len(state_history) == 0:
            state_history.append(list(state))
        else:
            state_history[-1] = list(state)
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
        
        if i % update_frequency == 0:
            states, actions, log_probs, rewards, dones = agent.process_data(state_history, action_history, reward_history, done_history, 64)
            for _ in range(n_update):
                agent.learn(states, actions, log_probs, rewards, dones)
            state_history = []
            action_history = []
            done_history = []
            reward_history = []
        
        rewards_log.append(episodic_reward)
        average_log.append(np.mean(rewards_log[-100:]))
        
        print('\rEpisode {} Reward {:.2f}, Average Reward {:.2f}'.format(i, episodic_reward, average_log[-1]), end='')
        if not done:
            print('\nEpisode {} did not end'.format(i))
        if i % 200 == 0:
            print()
            
    return rewards_log, average_log

if __name__ == '__main__':
    env = gym.make(RAM_DISCRETE_ENV_NAME)
    agent = Agent_discrete(state_size=env.observation_space.shape[0], 
                           action_size=env.action_space.n, 
                           lr=LEARNING_RATE, 
                           beta=BETA, 
                           eps=EPS, 
                           tau=TAU, 
                           gamma=GAMMA, 
                           device=DEVICE,
                           hidden=HIDDEN_DISCRETE,
                           share=SHARE, 
                           mode=MODE, 
                           use_critic=CRITIC, 
                           normalize=NORMALIZE)
    rewards_log, _ = train(agent=agent, 
                           env=env, 
                           n_episode=RAM_NUM_EPISODE, 
                           n_update=N_UPDATE, 
                           update_frequency=UPDATE_FREQUENCY, 
                           max_t=MAX_T, 
                           scale=SCALE)
    np.save('{}_rewards.npy'.format(RAM_DISCRETE_ENV_NAME), rewards_log)