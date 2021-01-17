import torch

LOAD = True            # loading neural networks from file
FILENAME_FOR_LOADING="tennis_trained"
SAVE = False            # saving neural networks to file
FILENAME_FOR_SAVING="tennis_trained"
PLOT=False
PLOTNAME="tennis_trained.png"
TRAINMODE = False
VAL_ENV_SOLVED = 3.0

MAX_NR_EPISODES= 10000          # max number episodes
BUFFER_SIZE = int(1e6)          # replay buffer size
BATCH_SIZE = 512                # minibatch size
GAMMA = 0.99                    # discount factor
TAU = 0.05                      # for soft update of target parameters
LR_ACTOR = 0.0005                 # learning rate of the actor
LR_CRITIC = 0.0005                # learning rate of the critic
WEIGHT_DECAY = 0.0              # L2 weight decay
UPDATE_EVERY = 2                # weight update frequency
SEED=0

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
