# Main code

from __future__ import print_function
import os
import torch
import torch.multiprocessing as mp
# import envs and model
from a3c_envs import create_atari_env
from a3c_model import ActorCritic
from a3c_train import train
from a3c_test import test
import a3c_custom_optim

# Gathering all the parameters (that we can modify to explore)
class Params():
    def __init__(self):
        self.lr = 0.0001 # learning rate
        self.gamma = 0.99 # gamme
        self.tau = 1. 
        self.seed = 1
        self.num_processes = 16
        self.num_steps = 20
        self.max_episode_length = 10000
        self.env_name = 'Breakout-v0'

# Main run
os.environ['OMP_NUM_THREADS'] = '1' # 1 thread per core
params = Params() # get all out parameters and initialize them
torch.manual_seed(params.seed) # set the seed
env = create_atari_env(params.env_name) # get the environment, create an optimized env using universe
shared_model = ActorCritic(env.observation_space.shape[0], env.action_space) # model shared by every agent and store it in the computer
shared_model.share_memory()
optimizer = a3c_custom_optim.SharedAdam(shared_model.parameters(), lr=params.lr) # link optimizer to shared model act on the shared model
optimizer.share_memory() # store the optimizer in the memory
processes = []
p = mp.Process(target=test, args=(params.num_processes, params, shared_model)) # runs a funciton on an independent thread (from torch)
p.start()
processes.append(p)
for rank in range(0, params.num_processes):
    p = mp.Process(target=train, args=(rank, params, shared_model, optimizer))
    p.start()
    processes.append(p)
for p in processes:
    p.join()
