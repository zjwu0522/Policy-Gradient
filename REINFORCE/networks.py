import torch
import torch.nn as nn
import torch.nn.functional as F

class Actor(nn.Module):

    def __init__(self, state_size, action_size, hidden=[128, 16]):
        super(Actor, self).__init__()
        self.fc1 = nn.Linear(state_size, hidden[0])
        self.fc2 = nn.Linear(hidden[0], hidden[1])
        self.fc3 = nn.Linear(hidden[1], action_size)

    def forward(self, state):
        x = state
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        log_probs = F.log_softmax(self.fc3(x), dim=1)
        return log_probs
    
class Critic(nn.Module):

    def __init__(self, state_size, hidden=[128, 16]):
        super(Critic, self).__init__()
        self.fc1 = nn.Linear(state_size, hidden[0])
        self.fc2 = nn.Linear(hidden[0], hidden[1])
        self.fc3 = nn.Linear(hidden[1], 1)

    def forward(self, state):
        x = state
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        values = self.fc3(x)
        return values
    
class Actor_Critic(nn.Module):

    def __init__(self, state_size, action_size, hidden=[128, 16]):
        super(Actor_Critic, self).__init__()
        self.fc1 = nn.Linear(state_size, hidden[0])
        self.fc2 = nn.Linear(hidden[0], hidden[1])
        self.fc3 = nn.Linear(hidden[1], action_size)
        self.fc4 = nn.Linear(hidden[1], 1)

    def forward(self, state):
        x = state
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        log_probs = F.log_softmax(self.fc3(x), dim=1)
        values = self.fc4(x)
        return log_probs, values