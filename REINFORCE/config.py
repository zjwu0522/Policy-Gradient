import torch

#device, CPU or the GPU of your choice
GPU = 0
DEVICE = torch.device("cuda:{}".format(GPU) if torch.cuda.is_available() else "cpu")

#environment names
RAM_ENV_NAME = 'LunarLander-v2'
VISUAL_ENV_NAME = 'Pong-v0'
CONSTANT = 90

#Agent parameters
LEARNING_RATE = 0.005
GAMMA = 0.99
CRITIC = False
NORMALIZE = True
SHARE = False
MODE = 'MC'

#Training parameters
RAM_NUM_EPISODE = 2000
VISUAL_NUM_EPISODE = 3000
SCALE = 0.01
MAX_T = 2000
NUM_FRAME = 2
BATCH_SIZE = 128