# create function to train

import torch
import torch.nn.functional as F
from a3c_model import ActorCritic
from torch.autograd import Variable
from a3c_envs import create_atari_env

def ensure_shared_grads(model, shared_model):
    for param, shared_param in zip(model.parameters(), shared_model.parameters()):
        if shared_param.grad is not None:
            return
        shared_param._grad = param.grad
        
# shared_model is what agent will get to explore the environment
def train(rank, params, shared_model, optimizer):
    # have to desynchronize every training agent
    # use rank to shift each seed, n agents mean rank is 0 to n
    torch.manual_seed(params.seed + rank) # desync each traiing agent
    # create environment for breakout
    env = create_atari_env(params.env_name)
    # align seed of the environment on the agent
    # each agent has it's own copy of the environment
    # we need to align each of the agent on one specific environment
    # associate different seed to each agent so they can have separate env
    env.seed(params.seed + rank)
    # create a3c model
    model = ActorCritic(env.observation_space.shape[0], env.action_space) # insert thru env
    # get state of env
    # state is 1 by 42 by 42 (1 is black)
    state = env.reset()
    # convert into torch tensors
    state = torch.from_numpy(state)
    # done is when game is over
    done = True
    episode_length = 0 # increment the episode_length
    while True:
        episode_length += 1
        model.load_state_dict(shared_model.state_dict())
        if done:
            # reinitialize the hidden and cell states
            # since output is 256 we need 256 zeroes
            cx = Variable(torch.zeros(1, 256))
            hx = Variable(torch.zeros(1, 256))
        else:
            # keep data
            cx = Variable(cx.date)
            hx = Variable(hx.data)
        values = [] # value of the critic
        log_probs = []
        rewards = []
        entropies = []
        # loop over exploration steps
        for step in range(params.num_steps):
            # get predictions of the model, apply it to the input
            # get the values of the v function
            value, action_values, (hx, cx) = model((Variable(state.unsqueeze(0)), (hx, cx))) # model need to be unsqueezed
            # get the probabiliteis using softmax
            prob = F.softmax(action_values)
            # remember entropy is the minus of the sum of the product log prob times prob
            log_prob = F.log_softmax(action_values)
            entropy = -(log_prob * prob).sum(1)
            # append to entropies
            entropies.append(entropy)
            
            action = prob.multinomial().data # take a random draw of the actions available
            log_prob = log_prob.gather(1, Variable(action)) # associate with the action
            # append to values and log_probs
            values.append(value)
            log_probs.append(log_prob)
            # by reaching a new state, we get a reward, refer to the env code
            state, reward, done = env.step(action.numpy())
            # make sure agent is not stuck in a state
            # limit the time by limiting max_episode_length
            done = (done or episode_length >= params.max_episode_length)
            # make sure reward between -1 and +1
            reward = max(min(reward, 1), -1)
            # check if game is done and then restart environment
            if done:
                episode_length = 0
                state = env.reset()
            # remember that state is an image in the form of a numpy array
            state = torch.from_numpy(state)
            # append reward to the rewards now
            rewards.append(reward)
            if done: # stop exploration if done
                break
        # cumulative reward
        R = torch.zeros(1, 1)
        if not done: # cumulative reward is the output of model in prev state
            value, _, _ = model((Variable(state.unsqueeze(0)), (hx, cx)))
            R = value.data
        values.append(Variable(R))
        # calculate loss now
        # remember we have 2 types of loss
        policy_loss = 0
        value_loss = 0
        R = Variable(R) # must be torch as we're comparing gradient R is a term of value loss
        # initialise the GAE generalised advantage estimation (advantage of action in state compared to another state)
        gae = torch.zeros(1, 1) # A(a, s) = Q(a, s) - V(s)
        # stochastic gradient descent
        # reversed is so that we can move back in time
        for i in reversed(range(len(rewards))):
            R = params.gamma * R + rewards[i] # we will get R = r_0 + gamma * r_1 + gamma^2 * r_2 + ... + gamma^(n-1) * r_(n-1) + gamma^nb_steps * V(last state)
            # compute the advantage of reward against the value
            advantage = R - values[i]
            # get the value loss Q*(a*, s) = V*(s)
            value_loss = value_loss + 0.5 * advantage.pow(2) # loss generated by the predictions of the V function output by the critic
            # use GAE for policy loss, temporal diff of state value
            TD = rewards[i] + params.gamma * values[i + 1].data - values[i].data
            gae = gae * params.gamma * params.tau + TD # gae = sum_i (gamme*tau)^i * TD(i)
            # we can now finally calculate policy loss
            # log of probability of the entropy are negative values
            # we maximise the probability of playing the action that will maximise the advantage
            # purpose of entropy is to prevent falling too quickly into a trap
            # where all actions 0 but one is 1, entropy is to control that from happening
            policy_loss = policy_loss - log_probs[i] * Variable(gae) - 0.01 * entropies[i] # policy_loss = - sum_i log(pi_i) + 0.01*R_i (entropy)
        # apply stochastic gradient descent
        optimizer.zero_grad()
        # give more importance to policy loss as its smaller than value loss
        (policy_loss + 0.5 * value_loss).backward()
        # prevent gradient from generating very large values
        # 40 is such that the norm of the gradient stays between 0 and 40
        torch.nn.utils.clip_grad_norm(model.parameters(), 40)
        # make sure model and share_model share the same grad
        ensure_shared_grads(model, shared_model)
        # now optimize to reduce the losses
        optimizer.step()
            
            