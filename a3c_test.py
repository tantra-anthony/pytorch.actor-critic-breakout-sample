# Test Agent

import torch
import torch.nn.functional as F
from a3c_envs import create_atari_env
from a3c_model import ActorCritic
from torch.autograd import Variable
import time
from collections import deque

# rank is to desync the test agent
def test(rank, params, shared_model):
    # desynchronising the agents
    torch.manual_seed(params.seed + rank)
    # create the environment
    env = create_atari_env(params.env_name, video=True)
    env.seed(params.seed + rank)
    model = ActorCritic(env.observation_space.shape[0], env.action_space)
    # since this is test mode, then we need to evaluate the model
    model.eval()
    state = env.reset()
    state = torch.from_numpy(state)
    # initialize all the required parameters
    reward_sum = 0
    done = True
    # start time to measure the time of computations
    start_time = time.time()
    actions = deque(maxlen = 100)
    episode_length = 0
    while True:
        episode_length += 1
        if done:
            # reload last state of the model
            model.load_state_dict(shared_model.state_dict())
            # reinitialize the cell and hidden states
            cx = Variable(torch.zeros(1, 256), volatile=True)
            hx = Variable(torch.zeros(1, 256), volatile=True)
        else:
            # we keep the same cell and hidden states
            # while making sure they are torch variables
            cx = Variable(cx.data, volatile=True)
            hx = Variable(hx.data, volatile=True)
        # get the predictions of the model
        # output of critic, output of actor, hidden and cell states
        value, action_value, (hx, cx) = model((Variable(state.unsqueeze(0), volatile=True), (hx, cx)))
        prob = F.softmax(action_value)
        # immediately play the action because there is no need to train
        action = prob.max(1)[1].data.numpy()
        state, reward, done, _ = env.step(action[0, 0])
        reward_sum += reward
        if done: # when the game is done
            print("Time {}, episode reward {}, episode length {}".format(time.strftime("%Hh %Mm %Ss", time.gmtime(time.time() - start_time)), reward_sum, episode_length))
            # reinitialize everything after game is done
            reward_sum = 0
            episode_length = 0
            actions.clear()
            state = env.reset()
            # do a break of 60 seconds to let the other agents practice
            time.sleep(60)
        # get new state
        state = torch.from_numpy(state)

