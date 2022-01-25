import torch

#device, CPU or the GPU of your choice
GPU = 0
DEVICE = torch.device("cuda:{}".format(GPU) if torch.cuda.is_available() else "cpu")

#environment names
RAM_DISCRETE_ENV_NAME = 'LunarLander-v2'
RAM_CONTINUOUS_ENV_NAME = 'LunarLanderContinuous-v2'
VISUAL_ENV_NAME = 'Pong-v0'
CONSTANT = 90

#Agent parameters
LEARNING_RATE = 0.005
GAMMA = 0.99
BETA = 0
EPS = 0.2
TAU = 0.99
MODE = 'TD'
SHARE = False
CRITIC = True
NORMALIZE = False
HIDDEN_DISCRETE = [128]


#Training parameters
RAM_NUM_EPISODE = 2000
VISUAL_NUM_EPISODE = 5000
SCALE = 1
MAX_T = 1000
NUM_FRAME = 2
N_UPDATE = 10
UPDATE_FREQUENCY = 4 
