import torch
import torch.optim as optim
import numpy as np

from networks import *

import math

class Agent:
    
    def __init__(self, state_size, action_size, lr, gamma, device, share=False, mode='MC', use_critic=False, normalize=False):
        self.state_size = state_size
        self.action_size = action_size
        self.lr = lr
        self.gamma = gamma
        self.device = device
        self.share = share
        self.mode = mode
        self.use_critic = use_critic
        self.normalize = normalize
        
        if self.share:
            self.Actor_Critic = Actor_Critic(self.state_size, self.action_size).to(self.device)
            self.optimizer = optim.Adam(self.Actor_Critic.parameters(), lr)
        else:
            self.Actor = Actor(state_size, action_size).to(self.device)
            self.Critic = Critic(state_size).to(self.device)
            self.actor_optimizer = optim.Adam(self.Actor.parameters(), lr)
            self.critic_optimizer = optim.Adam(self.Critic.parameters(), lr)
            
    def act(self, states):
        with torch.no_grad():
            states = torch.tensor(states).view(-1, self.state_size).to(self.device)
            if self.share:
                log_probs, _ = self.Actor_Critic(states)
            else:
                log_probs = self.Actor(states)
            probs = log_probs.exp().view(-1).cpu().numpy()
            action = np.random.choice(a=self.action_size, size=1, replace=False, p=probs)[0]
        return action
    
    def process_data(self, states, actions, rewards, dones, batch_size):
        
        states = torch.tensor(states, dtype=torch.float).to(self.device)
        actions = torch.tensor(actions, dtype=torch.long).to(self.device).view(-1, 1)
        dones = torch.tensor(dones, dtype=torch.float).to(self.device).view(-1,1)
        
        #calculate log probabilities and state values
        N = states.size(0) # N-1 is the length of actions, rewards and dones
        log_probs = torch.zeros((N, self.action_size)).to(self.device)
        state_values = torch.zeros((N, 1)).to(self.device)
        step = math.ceil(N/batch_size)
        
        for ind in range(step):
            if self.share:
                output1, output2 = self.Actor_Critic(states[ind*batch_size:(ind+1)*batch_size, :])
            else:
                output1 = self.Actor(states[ind*batch_size:(ind+1)*batch_size, :])
                output2 = self.Critic(states[ind*batch_size:(ind+1)*batch_size, :])
            log_probs[ind*batch_size:(ind+1)*batch_size, :] = output1
            state_values[ind*batch_size:(ind+1)*batch_size, :] = output2 
        
        log_probs = log_probs[:-1, :]# remove the last one, which corresponds to no actions
        log_probs = torch.gather(log_probs, dim=1, index=actions)
        
        #calculate discounted rewards, gamma^t r_t
        L = len(rewards)
        rewards = np.array(rewards) #r_t
        discounts = self.gamma ** np.arange(L)
        discounted_rewards = rewards * discounts # this is gamma^t r_t
        
        return state_values, log_probs, rewards, discounted_rewards, dones
    
    def learn(self, state_values, log_probs, rewards, discounted_rewards, dones):

        # Update Critic use MSE
        # Update Actor by maximizing A_t * log(a_t|s_t)

        L = len(discounted_rewards)
        with torch.no_grad():
            G = []
            return_value = 0
            if self.mode == 'MC':
                for i in range(L-1, -1, -1):
                    return_value = rewards[i] + self.gamma * (1-dones[i].detach().numpy()) * return_value
                    G.append(return_value)
                G = G[::-1]
                G = torch.tensor(G, dtype=torch.float).view(-1, 1).to(self.device)
            else:
                rewards = torch.tensor(rewards, dtype=torch.float).view(-1, 1).to(self.device)
                G = rewards + self.gamma * (1-dones) * state_values[1:, :]
            
        Critic_Loss = 0.5*(state_values[:-1, :] - G).pow(2).mean()
        
        with torch.no_grad():
            if self.use_critic:
                G = G - state_values[:-1, :] # advantage
            if self.normalize:
                G = (G - G.mean()) / (G.std() + 0.00001) # normalized advantage
                
        Actor_Loss = -log_probs * G
        Actor_Loss = Actor_Loss.mean()
        
        if self.share:
            Loss = Actor_Loss + Critic_Loss
            self.optimizer.zero_grad()
            Loss.backward()
            self.optimizer.step()
        else:
            self.critic_optimizer.zero_grad()
            Critic_Loss.backward()
            self.critic_optimizer.step()
            self.actor_optimizer.zero_grad()
            Actor_Loss.backward()
            self.actor_optimizer.step()